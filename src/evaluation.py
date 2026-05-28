"""
Phase D: Evaluation & Logging
Metrics tracking and CSV export for monitoring baseline performance.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class MetricsLogger:
    """
    Logs metrics per round and exports to CSV.
    Tracks accuracy, loss, divergence, tau, algorithm choice, privacy budget.
    """
    
    def __init__(self, output_dir: Path = config.RESULTS_DIR):
        """
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.output_dir / "rounds.csv"
        self.test_metrics_file = self.output_dir / "test_results.csv"
        
        self.round_data = []
        self.fieldnames = [
            "round",
            "split",
            "loss",
            "accuracy",
            "divergence",
            "tau",
            "algorithm",
            "epsilon",
            "num_clients",
        ]
        
        logger.info(f"MetricsLogger initialized, output: {self.output_dir}")
    
    def log_round(
        self,
        round_num: int,
        loss: float,
        accuracy: float = None,
        divergence: float = None,
        tau: float = None,
        algorithm: str = None,
        epsilon: float = None,
        num_clients: int = None,
        split: str = "train",
    ):
        """
        Log metrics for a training/evaluation round.
        
        Args:
            round_num: Round number
            loss: Loss value
            accuracy: Accuracy (if available)
            divergence: Divergence metric δ_t
            tau: Threshold τ
            algorithm: Algorithm used (FedAvg/FedSGD)
            epsilon: Privacy budget ε
            num_clients: Number of clients
            split: 'train', 'val', or 'test'
        """
        entry = {
            "round": round_num,
            "split": split,
            "loss": round(loss, 6) if loss is not None else "",
            "accuracy": round(accuracy, 6) if accuracy is not None else "",
            "divergence": round(divergence, 6) if divergence is not None else "",
            "tau": round(tau, 6) if tau is not None else "",
            "algorithm": algorithm or "",
            "epsilon": round(epsilon, 6) if epsilon is not None else "",
            "num_clients": num_clients or "",
        }
        
        self.round_data.append(entry)
        logger.debug(f"Logged round {round_num}: {entry}")
    
    def save_to_csv(self):
        """Save collected metrics to CSV file."""
        try:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.round_data)
            logger.info(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def log_test_results(self, results: Dict):
        """
        Log final test set results.
        
        Args:
            results: Dictionary with test metrics
        """
        test_file = self.output_dir / "test_results.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("FINAL TEST SET RESULTS\n")
                f.write("=" * 60 + "\n\n")
                
                for key, value in results.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            logger.info(f"Test results saved to {test_file}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = config.DEVICE,
) -> tuple:
    """
    Evaluate model on dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to evaluate on
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * len(batch_y)
            total_correct += (logits.argmax(1) == batch_y).sum().item()
            total_samples += len(batch_y)
    
    model.train()
    
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    return avg_loss, accuracy


def print_metrics_summary(
    metrics: Dict,
    title: str = "Metrics Summary"
):
    """Pretty-print metrics summary."""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, list):
            # For lists, print summary statistics
            if value and isinstance(value[0], (int, float)):
                print(f"{key}:")
                print(f"  Mean: {np.mean(value):.6f}")
                print(f"  Std:  {np.std(value):.6f}")
                print(f"  Min:  {np.min(value):.6f}")
                print(f"  Max:  {np.max(value):.6f}")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test logging
    logger_obj = MetricsLogger()
    
    # Simulate logging some rounds
    for r in range(5):
        logger_obj.log_round(
            round_num=r + 1,
            loss=0.5 - r * 0.05,
            accuracy=0.8 + r * 0.03,
            divergence=0.2 - r * 0.02,
            tau=0.10,
            algorithm="FedAvg" if r % 2 == 0 else "FedSGD",
            epsilon=2.5,
            num_clients=10,
            split="train",
        )
    
    logger_obj.save_to_csv()
    print(f"Logged metrics to {logger_obj.metrics_file}")
