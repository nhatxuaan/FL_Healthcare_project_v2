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
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class MetricsLogger:
    def __init__(self, output_dir: Path = config.RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "rounds.csv"
        self.round_data = []
        self.fieldnames = ["round", "split", "loss", "accuracy", "divergence", "tau", "algorithm", "epsilon", "num_clients"]

    def log_round(self, round_num: int, loss: float, accuracy: float = None, divergence: float = None, tau: float = None, algorithm: str = None, epsilon: float = None, num_clients: int = None, split: str = "train"):
        entry = {
            "round": round_num, "split": split,
            "loss": round(loss, 6) if loss is not None else "",
            "accuracy": round(accuracy, 6) if accuracy is not None else "",
            "divergence": round(divergence, 6) if divergence is not None else "",
            "tau": round(tau, 6) if tau is not None else "",
            "algorithm": algorithm or "", "epsilon": round(epsilon, 6) if epsilon is not None else "",
            "num_clients": num_clients or "",
        }
        self.round_data.append(entry)

    def save_to_csv(self):
        try:
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.round_data)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def log_test_results(self, results: Dict):
        test_file = self.output_dir / "test_results.txt"
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\nFINAL TEST SET RESULTS\n" + "=" * 60 + "\n\n")
                for key, value in results.items():
                    if key != "report_text":
                        f.write(f"{key}: {value:.6f}\n" if isinstance(value, float) else f"{key}: {value}\n")
                if "report_text" in results:
                    f.write("\n" + results["report_text"])
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


# HÀM GỐC: Trả về đúng 2 tham số để giữ nguyên độ chính xác 89% khi train
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = config.DEVICE) -> tuple:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_samples = 0.0, 0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)
            preds = logits.argmax(1)
            all_labels.extend(batch_y.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            
    model.train()
    return total_loss / max(total_samples, 1), accuracy_score(all_labels, all_preds)


# HÀM MỚI: Chỉ gọi 1 lần duy nhất ở Step 6 để xuất báo cáo chi tiết
def evaluate_model_detailed(model: nn.Module, dataloader: DataLoader, device: str = config.DEVICE) -> tuple:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_samples = 0.0, 0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)
            all_labels.extend(batch_y.detach().cpu().numpy())
            all_preds.extend(logits.argmax(1).detach().cpu().numpy())
            
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    report_text = classification_report(all_labels, all_preds, target_names=['Normal', 'Tuberculosis'], digits=4)
    
    print("\n" + "="*60 + f"\n{'🔬 COMPREHENSIVE CLASSIFICATION REPORT':^60}\n" + "="*60 + f"\n{report_text}\n" + "="*60 + "\n")
    return avg_loss, accuracy, report_text


def print_metrics_summary(metrics: Dict, title: str = "Metrics Summary"):
    print("\n" + "=" * 60 + f"\n{title:^60}\n" + "=" * 60)
    for key, value in metrics.items():
        if key == "report_text": continue
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("=" * 60 + "\n")