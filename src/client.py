"""
Phase C2: Federated Learning Client
FlowerClient subclasses for performing local training with DP protection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from imblearn.over_sampling import SMOTE
from opacus import PrivacyEngine
import flwr as fl
from flwr.common import NDArrays, Scalar

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data import TBChestXrayDataset, get_train_transform, get_val_transform
from src.partition import get_client_partition
from src.models import get_model


class FlowerClient(fl.client.NumPyClient):
    """
    Federated Learning client for local training.
    Handles data loading, model training with DP, and weight synchronization.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = config.LEARNING_RATE,
        device: str = config.DEVICE,
        dp_enabled: bool = config.DP_ENABLED,
    ):
        """
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_dataloader: Training dataloader for this client
            val_dataloader: Validation dataloader for this client
            learning_rate: Learning rate for local SGD
            device: Device to train on ('cpu' or 'cuda')
            dp_enabled: Whether to apply differential privacy
        """
        self.client_id = client_id
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.device = device
        self.dp_enabled = dp_enabled
        
        # Move model to device
        self.model = self.model.to(device)

        # Loss only; optimizers are created inside training paths.
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_privacy_metrics: Dict[str, float] = {}
        
        # Metrics
        self.local_epochs_trained = 0
        self.training_loss_history = []
        self.validation_loss_history = []
        self.validation_acc_history = []
        
        logger.info(
            f"Client {client_id} initialized: "
            f"train_batches={len(train_dataloader)}, "
            f"val_batches={len(val_dataloader)}, "
            f"dp={dp_enabled}"
        )
    
    def get_parameters(self, config: Dict) -> NDArrays:
        """
        Extract model weights as list of numpy arrays (for transmission to server).
        
        Args:
            config: Configuration dict from server
        
        Returns:
            List of parameter arrays
        """
        # Return independent NumPy snapshots to avoid accidental shared-memory mutation.
        return [param.detach().cpu().numpy().copy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model weights from numpy arrays received from server.
        
        Args:
            parameters: List of parameter arrays from server
        """
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), parameters):
                # Clone input arrays so client training cannot mutate server-side NumPy buffers.
                new_tensor = torch.tensor(new_param, device=self.device)
                param.data.copy_(new_tensor)
    
    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Perform local training for one round.
        
        Args:
            parameters: Model weights from server
            config: Configuration dict (may contain num_epochs, learning_rate, etc.)
        
        Returns:
            Tuple of (updated_weights, num_samples_trained, metrics_dict)
        """
        # Set weights
        self.set_parameters(parameters)
        
        # Get training config (epochs, batch size, etc.)
        num_epochs = config.get("num_epochs", config.get("local_epoch", 1))
        avg_loss, num_samples_trained = self._fit_feature_smote(num_epochs)

        self.training_loss_history.append(avg_loss)
        self.local_epochs_trained += num_epochs
        
        # Get privacy budget if DP enabled
        privacy_dict = {}
        if self.last_privacy_metrics:
            privacy_dict = self.last_privacy_metrics
            logger.info(
                f"Client {self.client_id} privacy budget: "
                f"ε={privacy_dict.get('epsilon', 0):.4f}, δ={privacy_dict.get('delta', 0)}"
            )
        
        return (
            self.get_parameters({}),  # Updated weights
            num_samples_trained,  # Number of samples trained on
            {
                "loss": avg_loss,
                **privacy_dict,  # Include privacy metrics
                "local_epochs": num_epochs,
            },
        )

    def _fit_standard(self, num_epochs: int) -> Tuple[float, int]:
        """Standard local training loop on image batches."""
        total_loss = 0.0
        epoch_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_x, batch_y in self.train_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item() * len(batch_y)
                epoch_loss += batch_loss
                epoch_samples += len(batch_y)

            epoch_loss /= max(epoch_samples, 1)
            total_loss += epoch_loss

            logger.info(
                f"Client {self.client_id}, Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss:.4f}"
            )

        avg_loss = total_loss / max(num_epochs, 1)
        return avg_loss, epoch_samples

    def _extract_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d features from ResNet50 before final FC layer."""
        m = self.model
        if hasattr(m, "_module"):
            m = m._module

        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _fit_feature_smote(self, num_epochs: int) -> Tuple[float, int]:
        """
        Paper-aligned training:
        1) Extract frozen backbone features.
        2) Apply SMOTE in feature space.
        3) Train classifier head on resampled features.
        4) If enabled, apply DP only to the classifier head.
        """
        self.last_privacy_metrics = {}
        base_model = self.model._module if hasattr(self.model, "_module") else self.model
        if not hasattr(base_model, "fc"):
            logger.warning(
                f"Client {self.client_id}: feature-space SMOTE currently implemented for ResNet-style model only; "
                f"fallback to standard training."
            )
            return self._fit_standard(num_epochs)

        self.model.eval()
        feature_batches: List[np.ndarray] = []
        label_batches: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y in self.train_dataloader:
                batch_x = batch_x.to(self.device)
                feats = self._extract_resnet_features(batch_x)
                feature_batches.append(feats.detach().cpu().numpy())
                label_batches.append(batch_y.detach().cpu().numpy())

        if not feature_batches:
            logger.warning(f"Client {self.client_id}: empty local dataset, skipping fit")
            return 0.0, 0

        X_feats = np.concatenate(feature_batches, axis=0)
        y_labels = np.concatenate(label_batches, axis=0)

        logger.info(f"Client {self.client_id} - Before SMOTE: {np.bincount(y_labels, minlength=2)}")

        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_feats, y_labels)
            logger.info(f"Client {self.client_id} - After SMOTE: {np.bincount(y_resampled, minlength=2)}")
        except Exception as e:
            logger.warning(f"Client {self.client_id}: skip SMOTE due to tiny local data/non-IID partition: {e}")
            X_resampled, y_resampled = X_feats, y_labels

        # Create balanced feature dataloader on CPU, move to device in training loop.
        tensor_x = torch.tensor(X_resampled, dtype=torch.float32)
        tensor_y = torch.tensor(y_resampled, dtype=torch.long)
        smote_dataset = TensorDataset(tensor_x, tensor_y)
        smote_loader = DataLoader(smote_dataset, batch_size=max(8, int(config.BATCH_SIZE)), shuffle=True)

        classifier_head = base_model.fc.to(self.device)
        classifier_head.train()
        optimizer = optim.SGD(classifier_head.parameters(), lr=self.learning_rate, momentum=0.9)

        train_loader = smote_loader
        dp_head = None
        privacy_engine = None
        if self.dp_enabled and len(smote_dataset) > 0:
            try:
                privacy_engine = PrivacyEngine()
                dp_head, optimizer, train_loader = privacy_engine.make_private(
                    module=classifier_head,
                    optimizer=optimizer,
                    data_loader=smote_loader,
                    noise_multiplier=config.DP_NOISE_MULTIPLIER,
                    max_grad_norm=config.DP_MAX_GRAD_NORM,
                )
            except Exception as e:
                logger.warning(f"Client {self.client_id}: DP-head wrap failed, continue without DP for this round: {e}")
                dp_head = None
                privacy_engine = None

        train_head = dp_head if dp_head is not None else classifier_head

        total_loss = 0.0
        num_samples = len(smote_dataset)

        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0
            epoch_seen = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = train_head(batch_x)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_count = batch_y.size(0)
                epoch_loss_sum += loss.item() * batch_count
                epoch_seen += batch_count

            epoch_loss = epoch_loss_sum / max(epoch_seen, 1)
            total_loss += epoch_loss
            logger.info(
                f"Client {self.client_id}, Epoch {epoch + 1}/{num_epochs} (feature-SMOTE): loss={epoch_loss:.4f}"
            )

        # If DP wrapper was used, copy trained weights back to base classifier head.
        if dp_head is not None:
            cleaned_state = {
                k.replace("_module.", ""): v
                for k, v in dp_head.state_dict().items()
            }
            classifier_head.load_state_dict(cleaned_state, strict=False)

        if privacy_engine is not None:
            delta = 1e-5
            try:
                epsilon = float(privacy_engine.accountant.get_epsilon(delta=delta))
                self.last_privacy_metrics = {"epsilon": epsilon, "delta": delta}
            except Exception as e:
                logger.warning(f"Client {self.client_id}: failed to compute epsilon after DP training: {e}")

        self.model.train()
        avg_loss = total_loss / max(num_epochs, 1)
        return avg_loss, int(num_samples)
    
    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on validation set.
        
        Args:
            parameters: Model weights
            config: Configuration dict
        
        Returns:
            Tuple of (loss, num_samples, metrics_dict)
        """
        # Set weights
        self.set_parameters(parameters)
        
        # Evaluation loop
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in self.val_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)
                
                total_loss += loss.item() * len(batch_y)
                total_correct += (logits.argmax(1) == batch_y).sum().item()
                total_samples += len(batch_y)
        
        self.model.train()
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        
        self.validation_loss_history.append(avg_loss)
        self.validation_acc_history.append(accuracy)
        
        logger.info(
            f"Client {self.client_id} validation: loss={avg_loss:.4f}, acc={accuracy:.4f}"
        )
        
        return avg_loss, total_samples, {"accuracy": accuracy}


def create_client(
    client_id: int,
    train_dataset: TBChestXrayDataset,
    val_dataset: TBChestXrayDataset,
    batch_size: int = config.BATCH_SIZE,
    model_name: str = "resnet50",
    learning_rate: float = config.LEARNING_RATE,
    device: str = config.DEVICE,
    dp_enabled: bool = config.DP_ENABLED,
) -> FlowerClient:
    """
    Factory function to create a FlowerClient.
    
    Args:
        client_id: Client identifier
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for dataloaders
        model_name: Model architecture ('resnet50' or 'vgg16')
        learning_rate: Learning rate
        device: Compute device
        dp_enabled: Whether to use differential privacy
    
    Returns:
        Initialized FlowerClient
    """
    # Create model
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return FlowerClient(
        client_id=client_id,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=learning_rate,
        device=device,
        dp_enabled=dp_enabled,
    )


if __name__ == "__main__":
    # Test client creation
    train_ds = TBChestXrayDataset(config.TB_ORGANIZED_ROOT, split="train", transform=get_train_transform())
    val_ds = TBChestXrayDataset(config.TB_ORGANIZED_ROOT, split="val", transform=get_val_transform())
    
    client = create_client(0, train_ds, val_ds)
    print(f"Client created: {client}")
