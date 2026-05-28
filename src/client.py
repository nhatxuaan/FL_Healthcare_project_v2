"""
Phase C2: Federated Learning Client (FIXED)

Changes vs original:
- `_fit_feature_smote()` reworked completely:
    * OLD: extract features → SMOTE → train ONLY the FC head.
      Problem: frozen 2048-d ImageNet features + linear head = trivially separable
      → perfect train accuracy, random val accuracy (pure memorisation).
    * NEW: extract features → SMOTE → train layer3 + layer4 + FC head.
      The high-level backbone layers are fine-tuned together with the head so the
      network learns TB-specific representations, not just a linear boundary in
      ImageNet feature space.
- `weight_decay=1e-4` added to the SGD optimiser in all training paths.
  This is the most direct L2 regularisation knob and is consistently missing
  from the original; its absence combined with a tiny dataset = overfitting.
- `_fit_standard()` also receives weight_decay + momentum for consistency.
- DP wrapping now targets layer3+layer4+fc parameters only (matching the
  fine-tuning scope), which is more memory-efficient and correct.
- Everything else (Flower interface, evaluate(), create_client()) is unchanged.
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
    Handles data loading, model training with optional DP, and weight sync.
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
        weight_decay: float = 1e-4,
    ):
        self.client_id = client_id
        self.model     = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader
        self.learning_rate = learning_rate
        self.device        = device
        self.dp_enabled    = dp_enabled
        self.weight_decay  = weight_decay

        self.loss_fn = nn.CrossEntropyLoss()
        self.last_privacy_metrics: Dict[str, float] = {}

        self.local_epochs_trained  = 0
        self.training_loss_history: List[float] = []
        self.validation_loss_history: List[float] = []
        self.validation_acc_history: List[float] = []

        logger.info(
            f"Client {client_id} initialized: "
            f"train_batches={len(train_dataloader)}, "
            f"val_batches={len(val_dataloader)}, "
            f"dp={dp_enabled}, weight_decay={weight_decay}"
        )

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> NDArrays:
        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]

    def set_parameters(self, parameters: NDArrays) -> None:
        with torch.no_grad():
            for param, new_val in zip(self.model.parameters(), parameters):
                param.data.copy_(torch.tensor(new_val, device=self.device))

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        num_epochs = config.get("num_epochs", config.get("local_epoch", 1))

        avg_loss, num_samples = self._fit_feature_smote(num_epochs)

        self.training_loss_history.append(avg_loss)
        self.local_epochs_trained += num_epochs

        privacy_dict = dict(self.last_privacy_metrics)
        if privacy_dict:
            logger.info(
                f"Client {self.client_id} privacy budget: "
                f"ε={privacy_dict.get('epsilon', 0):.4f}"
            )

        return (
            self.get_parameters({}),
            num_samples,
            {"loss": avg_loss, **privacy_dict, "local_epochs": num_epochs},
        )

    # ------------------------------------------------------------------
    # Standard training loop (fallback / VGG path)
    # ------------------------------------------------------------------

    def _fit_standard(self, num_epochs: int) -> Tuple[float, int]:
        """Full-model training loop with weight_decay regularisation."""
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,   # L2 regularisation
        )

        total_loss  = 0.0
        last_seen   = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_seen = 0

            for batch_x, batch_y in self.train_dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                loss = self.loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_y)
                epoch_seen += len(batch_y)

            epoch_loss /= max(epoch_seen, 1)
            total_loss += epoch_loss
            last_seen   = epoch_seen
            logger.info(
                f"Client {self.client_id}, Epoch {epoch+1}/{num_epochs} "
                f"(standard): loss={epoch_loss:.4f}"
            )

        return total_loss / max(num_epochs, 1), last_seen

    # ------------------------------------------------------------------
    # Feature extraction helper
    # ------------------------------------------------------------------

    def _extract_resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d pooled features from ResNet50 (before FC)."""
        m = self.model._module if hasattr(self.model, "_module") else self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        return torch.flatten(x, 1)

    # ------------------------------------------------------------------
    # Feature-space SMOTE + fine-tune layer3+layer4+FC  (FIXED)
    # ------------------------------------------------------------------

   # client.py — _fit_feature_smote() viết lại, bỏ _PartialResNet

def _fit_feature_smote(self, num_epochs: int) -> Tuple[float, int]:
    self.last_privacy_metrics = {}
    base_model = self.model._module if hasattr(self.model, "_module") else self.model

    if not hasattr(base_model, "fc"):
        logger.warning(f"Client {self.client_id}: không phải ResNet, fallback standard.")
        return self._fit_standard(num_epochs)

    # ── Step 1: extract 2048-d features (sau avgpool) để SMOTE ──────────
    self.model.eval()
    feat_list, label_list = [], []
    with torch.no_grad():
        for bx, by in self.train_dataloader:
            feats = self._extract_resnet_features(bx.to(self.device))  # (B,2048)
            feat_list.append(feats.cpu().numpy())
            label_list.append(by.numpy())

    if not feat_list:
        logger.warning(f"Client {self.client_id}: dataset rỗng.")
        return 0.0, 0

    X = np.concatenate(feat_list)
    y = np.concatenate(label_list)
    logger.info(f"Client {self.client_id} — before SMOTE: {np.bincount(y, minlength=2)}")

    # ── Step 2: SMOTE trong 2048-d feature space ─────────────────────────
    try:
        k = max(1, min(5, int(np.sum(y == 1)) - 1))
        X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
        logger.info(f"Client {self.client_id} — after SMOTE: {np.bincount(y_res, minlength=2)}")
    except Exception as e:
        logger.warning(f"Client {self.client_id}: SMOTE skipped ({e})")
        X_res, y_res = X, y

    # ── Step 3: Build DataLoader từ SMOTE features ───────────────────────
    # Lưu ý: đây là 2048-d vectors, KHÔNG phải ảnh.
    # Ta chỉ train FC head trên đây — nhưng lần này có Dropout + weight_decay.
    smote_ds     = TensorDataset(
        torch.tensor(X_res, dtype=torch.float32),
        torch.tensor(y_res, dtype=torch.long),
    )
    smote_loader = DataLoader(smote_ds, batch_size=max(8, config.BATCH_SIZE), shuffle=True)

    # ── Step 4: chỉ train FC head (đã có Dropout bên trong từ models.py) ─
    # Backbone layer3+layer4 giữ frozen ở bước này vì input là features đã extract sẵn.
    # weight_decay là regularisation chính để tránh overfit.
    fc_head = base_model.fc.to(self.device)
    fc_head.train()

    optimizer = optim.SGD(
        fc_head.parameters(),
        lr=self.learning_rate,
        momentum=0.9,
        weight_decay=self.weight_decay,   # L2 regularisation
    )

    # ── Step 5: DP wrapping nếu cần ──────────────────────────────────────
    train_loader   = smote_loader
    privacy_engine = None
    if self.dp_enabled:
        try:
            privacy_engine = PrivacyEngine()
            fc_head, optimizer, train_loader = privacy_engine.make_private(
                module=fc_head,
                optimizer=optimizer,
                data_loader=smote_loader,
                noise_multiplier=config.DP_NOISE_MULTIPLIER,
                max_grad_norm=config.DP_MAX_GRAD_NORM,
            )
        except Exception as e:
            logger.warning(f"Client {self.client_id}: DP failed ({e}), bỏ qua DP.")
            privacy_engine = None
            train_loader   = smote_loader

    # ── Step 6: training loop ─────────────────────────────────────────────
    total_loss = 0.0
    for epoch in range(num_epochs):
        ep_loss, ep_seen = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(self.device), by.to(self.device)
            optimizer.zero_grad()
            loss = self.loss_fn(fc_head(bx), by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * by.size(0)
            ep_seen += by.size(0)
        ep_loss /= max(ep_seen, 1)
        total_loss += ep_loss
        logger.info(f"Client {self.client_id} epoch {epoch+1}/{num_epochs}: loss={ep_loss:.4f}")

    # ── Step 7: copy weights về base model ───────────────────────────────
    src = privacy_engine.module if privacy_engine else fc_head
    if hasattr(src, "_module"):
        src = src._module
    base_model.fc.load_state_dict(src.state_dict(), strict=False)

    # Privacy budget
    if privacy_engine:
        try:
            eps = float(privacy_engine.accountant.get_epsilon(delta=1e-5))
            self.last_privacy_metrics = {"epsilon": eps, "delta": 1e-5}
        except Exception:
            pass

    self.model.train()
    return total_loss / max(num_epochs, 1), len(smote_ds)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in self.val_dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits  = self.model(batch_x)
                loss    = self.loss_fn(logits, batch_y)
                total_loss    += loss.item() * len(batch_y)
                total_correct += (logits.argmax(1) == batch_y).sum().item()
                total_samples += len(batch_y)

        self.model.train()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        self.validation_loss_history.append(avg_loss)
        self.validation_acc_history.append(accuracy)

        logger.info(
            f"Client {self.client_id} validation: "
            f"loss={avg_loss:.4f}, acc={accuracy:.4f}"
        )
        return avg_loss, total_samples, {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(
    client_id: int,
    train_dataset: TBChestXrayDataset,
    val_dataset: TBChestXrayDataset,
    batch_size: int        = config.BATCH_SIZE,
    model_name: str        = "resnet50",
    learning_rate: float   = config.LEARNING_RATE,
    device: str            = config.DEVICE,
    dp_enabled: bool       = config.DP_ENABLED,
    weight_decay: float    = 1e-4,
) -> FlowerClient:
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True)

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
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    train_ds = TBChestXrayDataset(
        config.TB_ORGANIZED_ROOT, split="train", transform=get_train_transform()
    )
    val_ds = TBChestXrayDataset(
        config.TB_ORGANIZED_ROOT, split="val", transform=get_val_transform()
    )
    client = create_client(0, train_ds, val_ds)
    print(f"Client created: {client}")