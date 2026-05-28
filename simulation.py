"""
Phase E: Federated Learning Simulation
Main runner for Flower-based FL simulation with adaptive aggregation.

Pipeline:
  1. Data preparation (split, organize, resize)
  2. Data partitioning (Dirichlet non-IID)
  3. Client factory with DP-enabled training
  4. Server strategy with FedAvg/FedSGD switching
  5. Simulation execution (flwr.simulation)
  6. Metrics logging and export
"""

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar

import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.prepare_data import prepare_tb_dataset, verify_prepared_dataset
from src.data import TBChestXrayDataset, create_dataloaders, get_train_transform, get_val_transform, custom_collate_fn
from src.partition import dirichlet_partition
from src.models import get_model
from src.client import FlowerClient
from src.strategy import AdaptiveAggregationStrategy
# IMPORT THÊM HÀM ĐÁNH GIÁ CHI TIẾT TỪ BƯỚC 1
from src.evaluation import MetricsLogger, print_metrics_summary, evaluate_model_detailed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_simulation_data():
    """Phase 1: Prepare TB dataset (split, organize, resize)."""
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("=" * 70)
    
    organized_dir = config.TB_ORGANIZED_ROOT
    if organized_dir.exists() and len(list(organized_dir.glob("*/*/*"))) > 100:
        logger.info(f"Data already organized at {organized_dir}. Skipping preparation.")
    else:
        logger.info(f"Preparing TB dataset from {config.TB_DATA_ROOT}")
        prepare_tb_dataset(
            raw_dir=config.TB_DATA_ROOT,
            output_dir=config.TB_ORGANIZED_ROOT,
            target_size=(224, 224),
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
    
    verify_prepared_dataset(config.TB_ORGANIZED_ROOT)
    logger.info("✓ Data preparation complete\n")
    return organized_dir


def partition_data(num_clients: int = config.NUM_CLIENTS_BASELINE):
    """Phase 2: Partition training data using Dirichlet(α=0.5)."""
    logger.info("=" * 70)
    logger.info("PHASE 2: DATA PARTITIONING")
    logger.info("=" * 70)
    
    train_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="train",
        transform=get_train_transform(),
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Partitioning into {num_clients} clients with Dirichlet(α={config.DIRICHLET_ALPHA})")
    
    labels = np.array([label for _, label in train_dataset.samples], dtype=np.int32)
    
    partitions = dirichlet_partition(
        dataset_indices=np.arange(len(train_dataset), dtype=np.int32),
        labels=labels,
        num_clients=num_clients,
        alpha=config.DIRICHLET_ALPHA,
        seed=42,
    )
    
    logger.info(f"✓ Partitioning complete: {len(partitions)} clients\n")
    return partitions, train_dataset


def create_client_fn(
    partitions: List[np.ndarray],
    train_dataset: TBChestXrayDataset,
    model_name: str = "resnet50",
    dp_enabled: bool = config.DP_ENABLED,
) -> Callable[[str], FlowerClient]:
    """Create client factory function for Flower simulation."""
    
    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        client_indices = partitions[cid_int]
        subset = torch.utils.data.Subset(train_dataset, client_indices)
        
        if len(subset) == 0:
            logger.warning(f"Client {cid}: No training data assigned (empty partition)")
            train_loader = DataLoader(
                subset,
                batch_size=config.BATCH_SIZE,
                sampler=torch.utils.data.SequentialSampler(subset),
                num_workers=0,
            )
        else:
            train_loader = DataLoader(
                subset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
            )
        
        val_loader = DataLoader(
            TBChestXrayDataset(
                root_dir=config.TB_ORGANIZED_ROOT,
                split="val",
                transform=get_val_transform(),
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        
        model = get_model(
            model_name=model_name,
            pretrained=True,
            freeze_backbone=True,
            num_classes=2,
        )
        
        client = FlowerClient(
            client_id=cid_int,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            learning_rate=config.LEARNING_RATE,
            device=config.DEVICE,
            dp_enabled=dp_enabled,
        )
        return client
    
    return client_fn


def run_simulation(
    num_clients: int = config.NUM_CLIENTS_BASELINE,
    num_rounds: int = 10,
    model_name: str = "resnet50",
    dp_enabled: bool = config.DP_ENABLED,
    min_available_clients: int = None,
    min_fit_clients: int = None,
    min_evaluate_clients: int = None,
):
    logger.info("=" * 70)
    logger.info("FEDERATED LEARNING SIMULATION")
    logger.info("=" * 70)
    
    if min_available_clients is None:
        min_available_clients = int(0.9 * num_clients)
    if min_fit_clients is None:
        min_fit_clients = num_clients
    if min_evaluate_clients is None:
        min_evaluate_clients = num_clients
    
    # Step 1: Prepare data
    prepare_simulation_data()
    
    # Step 2: Partition data
    partitions, train_dataset = partition_data(num_clients)
    
    # Step 3: Create client factory
    client_fn = create_client_fn(
        partitions=partitions,
        train_dataset=train_dataset,
        model_name=model_name,
        dp_enabled=dp_enabled,
    )
    
    # Step 4: Create strategy
    strategy = AdaptiveAggregationStrategy(
        fraction_fit=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    # Step 5: Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25 if config.DEVICE == "cuda" else 0},
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: EVALUATION & LOGGING WITH PER-CLASS METRICS")
    logger.info("=" * 70)
    
    # Step 6: Evaluate on test set
    test_dataset = TBChestXrayDataset(
        root_dir=config.TB_ORGANIZED_ROOT,
        split="test",
        transform=get_val_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    
    final_model = get_model(
        model_name=model_name,
        pretrained=True,
        freeze_backbone=False,
        num_classes=2,
    )
    
    test_client = client_fn("0")
    if strategy.final_parameters is not None:
        test_client.set_parameters(
            fl.common.parameters_to_ndarrays(strategy.final_parameters)
        )
    
    state_dict = test_client.model.state_dict()
    corrected_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_module."):
            corrected_state_dict[key[8:]] = value
        else:
            corrected_state_dict[key] = value
    
    final_model.load_state_dict(corrected_state_dict, strict=False)
    final_model = final_model.to(config.DEVICE)
    
    # GỌI HÀM ĐÁNH GIÁ CHI TIẾT ĐÃ CẬP NHẬT Ở BƯỚC 1
    test_loss, detailed_report = evaluate_model_detailed(
        model=final_model,
        dataloader=test_loader,
        device=config.DEVICE,
    )
    
    # Bóc tách các thông số tổng hợp và từng lớp từ báo cáo chi tiết
    test_accuracy = detailed_report["accuracy"]
    
    normal_f1 = detailed_report["Normal"]["f1-score"]
    normal_recall = detailed_report["Normal"]["recall"]
    normal_precision = detailed_report["Normal"]["precision"]
    
    tb_f1 = detailed_report["TB"]["f1-score"]
    tb_recall = detailed_report["TB"]["recall"]
    tb_precision = detailed_report["TB"]["precision"]
    
    # In thông số ghi nhận ra Logger hệ thống
    logger.info(f"Final Global Model Test Loss: {test_loss:.6f}")
    logger.info(f"Final Global Model Test Accuracy: {test_accuracy:.6f}")
    logger.info(f"Class [Normal] -> Precision: {normal_precision:.4f}, Recall: {normal_recall:.4f}, F1-Score: {normal_f1:.4f}")
    logger.info(f"Class [TB]     -> Precision: {tb_precision:.4f}, Recall: {tb_recall:.4f}, F1-Score: {tb_f1:.4f}")
    
    # Step 7: Log metrics & Save to CSV
    metrics_logger = MetricsLogger(output_dir=config.RESULTS_DIR)
    
    divergence_history = strategy.get_metrics()["divergence_history"]
    algorithm_history = strategy.get_metrics()["algorithm_history"]
    loss_history = strategy.get_metrics()["loss_history"]
    
    for r in range(len(loss_history)):
        metrics_logger.log_round(
            round_num=r + 1,
            loss=loss_history[r] if r < len(loss_history) else 0,
            divergence=divergence_history[r] if r < len(divergence_history) else 0,
            tau=config.TAU_STATIC,
            algorithm=algorithm_history[r] if r < len(algorithm_history) else "N/A",
            num_clients=num_clients,
            split="train",
        )
    
    metrics_logger.log_round(
        round_num=num_rounds + 1,
        loss=test_loss,
        accuracy=test_accuracy,
        split="test",
    )
    metrics_logger.save_to_csv()
    
    # Cập nhật thông số mở rộng vào dict lưu file kết quả cuối cùng
    test_results = {
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Normal_Precision": normal_precision,
        "Normal_Recall": normal_recall,
        "Normal_F1_Score": normal_f1,
        "TB_Precision": tb_precision,
        "TB_Recall": tb_recall,
        "TB_F1_Score": tb_f1,
        "Model": model_name,
        "Clients": num_clients,
        "Rounds": num_rounds,
        "DP Enabled": dp_enabled,
    }
    metrics_logger.log_test_results(test_results)
    
    # Print summary hiển thị màn hình cuối cùng
    print_metrics_summary(
        metrics={
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "Normal F1-Score": normal_f1,
            "TB F1-Score": tb_f1,
            "TB Recall (Sensitivity)": tb_recall,
            "Training Rounds": num_rounds,
            "Clients": num_clients,
        },
        title="SIMULATION COMPLETE - FINAL DETAILED METRICS"
    )
    
    return history, test_loss, test_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FL simulation baseline")
    parser.add_argument("--num-clients", type=int, default=config.NUM_CLIENTS_BASELINE)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vgg16"])
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    
    args = parser.parse_args()
    dp_enabled = config.DP_ENABLED and not args.no_dp
    
    run_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        model_name=args.model,
        dp_enabled=dp_enabled,
    )