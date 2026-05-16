"""
Simple smoke test for FL pipeline without Ray (which is incompatible with Python 3.13+)
Manually orchestrates FL rounds to validate data loading, model training, and aggregation.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.prepare_data import verify_prepared_dataset
from src.data import TBChestXrayDataset, get_train_transform, get_val_transform
from src.partition import dirichlet_partition
from src.models import get_model
from src.client import FlowerClient
from src.aggregators import aggregate_adaptive, compute_divergence
from src.evaluation import MetricsLogger, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def smoke_test_fl_pipeline(
    num_clients: int = 10,
    num_rounds: int = 2,
    model_name: str = "resnet50",
):
    """
   Simple smoke test that manually runs FL rounds.
    
    Args:
        num_clients: Number of clients to simulate
        num_rounds: Number of FL rounds
        model_name: Model architecture to use
    """
    logger.info("=" * 70)
    logger.info("SMOKE TEST: FEDERATED LEARNING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Config: model={model_name}, clients={num_clients}, rounds={num_rounds}")
    logger.info(f"DP=False (disabled for smoke test), Non-IID α={config.DIRICHLET_ALPHA}\n")
    
    # Phase 1: Verify preparation
    logger.info("PHASE 1: Verifying data preparation...")
    verify_prepared_dataset(config.TB_ORGANIZED_ROOT)
    logger.info("✓ Data verified\n")
    
    # Phase 2: Load and partition training data
    logger.info("PHASE 2: Data partitioning...")
    train_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="train",
        transform=get_train_transform(),
    )
    labels = np.array([label for _, label in train_dataset.samples], dtype=np.int32)
    
    partitions = dirichlet_partition(
        dataset_indices=np.arange(len(train_dataset), dtype=np.int32),
        labels=labels,
        num_clients=num_clients,
        alpha=config.DIRICHLET_ALPHA,
        seed=42,
    )
    logger.info(f"✓ Split {len(train_dataset)} samples into {num_clients} clients\n")
    
    # Phase 3: Initialize val and test datasets
    logger.info("PHASE 3: Loading validation and test sets...")
    val_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="val",
        transform=get_val_transform(),
    )
    test_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="test",
        transform=get_val_transform(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    logger.info(f"✓ Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    # Phase 4: Initialize global model
    logger.info("PHASE 4: Initializing global model...")
    global_model = get_model(
        model_name=model_name,
        pretrained=True,
        freeze_backbone=True,
        num_classes=2,
    )
    global_model = global_model.to(config.DEVICE)
    # Keep server parameters as immutable snapshots for each round comparison.
    global_params = [param.detach().cpu().numpy().copy() for param in global_model.parameters()]
    logger.info("✓ Global model initialized\n")
    
    # Phase 5: FL training loop
    logger.info("PHASE 5: Running FL rounds...")
    metrics_logger = MetricsLogger(output_dir=config.RESULTS_DIR)
    
    div_history = []
    algo_history = []
    loss_history = []
    
    for round_num in range(num_rounds):
        logger.info(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Client training
        client_results = []
        
        for client_id in range(num_clients):
            # Get client data (skip clients with 0 samples)
            client_indices = partitions[client_id]
            if len(client_indices) == 0:
                logger.info(f"  Client {client_id}: skipped (0 samples)")
                continue
            
            client_subset = torch.utils.data.Subset(train_dataset, client_indices)
            client_train_loader = DataLoader(
                client_subset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )
            
            # Create client model
            client_model = get_model(
                model_name=model_name,
                pretrained=(client_id == 0),  # Only pretrain for first client to save time
                freeze_backbone=True,
                num_classes=2,
            )
            client_model = client_model.to(config.DEVICE)
            
            # Create client
            client = FlowerClient(
                client_id=str(client_id),
                model=client_model,
                train_dataloader=client_train_loader,
                val_dataloader=val_loader,
                learning_rate=config.LEARNING_RATE,
                device=config.DEVICE,
                dp_enabled=False,
            )
            
            # Set global parameters on client
            client.set_parameters(global_params)
            
            # Train client locally
            updated_params, num_samples, metrics = client.fit(
                parameters=global_params,
                config={"num_epochs": 1}
            )
            
            client_results.append((updated_params, num_samples, metrics))
            logger.info(
                f"  Client {client_id}: {num_samples} samples, "
                f"loss={metrics['loss']:.4f}"
            )
        
        # Server aggregation
        if not client_results:
            logger.warning("No client results available")
            continue
        
        client_weights = [params for params, _, _ in client_results]
        sample_counts = [num for _, num, _ in client_results]
        
        # Compute divergence and aggregate
        divergence = compute_divergence(client_weights, global_params)
        div_history.append(divergence)
        
        aggregated, algorithm_used, _ = aggregate_adaptive(
            client_weights,
            global_params,
            tau=config.TAU_STATIC,
            learning_rate=config.LEARNING_RATE,
            client_sample_counts=sample_counts,
        )
        
        global_params = aggregated
        algo_history.append(algorithm_used)
        
        # Compute round loss from client metrics
        total_loss = sum(
            metrics['loss'] * num for _, num, metrics in client_results
        )
        avg_loss = total_loss / sum(sample_counts)
        loss_history.append(avg_loss)
        
        # Extract epsilon from client metrics if DP is enabled
        privacy_epsilon = None
        if config.DP_ENABLED and client_results:
            # Get epsilon from first client that has it
            for _, _, metrics in client_results:
                if 'epsilon' in metrics:
                    privacy_epsilon = metrics['epsilon']
                    break
        
        logger.info(f"Round {round_num + 1} Summary:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Divergence: {divergence:.6f}")
        logger.info(f"  Algorithm: {algorithm_used}")
        logger.info(f"  Tau: {config.TAU_STATIC}")
        if privacy_epsilon is not None:
            logger.info(f"  Privacy: ε={privacy_epsilon:.4f}")
        
        # Log to CSV
        metrics_logger.log_round(
            round_num=round_num + 1,
            loss=avg_loss,
            divergence=divergence,
            tau=config.TAU_STATIC,
            algorithm=algorithm_used,
            epsilon=privacy_epsilon,
            num_clients=len(client_results),
            split="train",
        )
    
    # Phase 6: Evaluate on test set
    logger.info("\nPHASE 6: Evaluating on test set...")
    final_model = get_model(
        model_name=model_name,
        pretrained=True,
        freeze_backbone=False,
        num_classes=2,
    )
    final_model = final_model.to(config.DEVICE)
    
    # Set global parameters on final model
    temp_client = FlowerClient(
        client_id="test",
        model=final_model,
        train_dataloader=val_loader,
        val_dataloader=val_loader,
        learning_rate=config.LEARNING_RATE,
        device=config.DEVICE,
        dp_enabled=False,
    )
    temp_client.set_parameters(global_params)
    final_model.load_state_dict(temp_client.model.state_dict())
    
    test_loss, test_accuracy = evaluate_model(
        model=final_model,
        dataloader=test_loader,
        device=config.DEVICE,
    )
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Accuracy: {test_accuracy:.6f}")
    
    # Log final results
    metrics_logger.log_round(
        round_num=num_rounds + 1,
        loss=test_loss,
        accuracy=test_accuracy,
        split="test",
    )
    metrics_logger.save_to_csv()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SMOKE TEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Clients: {num_clients}")
    logger.info(f"Rounds: {num_rounds}")
    logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info("=" * 70 + "\n")
    
    return test_loss, test_accuracy


if __name__ == "__main__":
    test_loss, test_acc = smoke_test_fl_pipeline(
        num_clients=10,
        num_rounds=2,
        model_name="resnet50",
    )
    print(f"\n✓ Smoke test complete!")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
