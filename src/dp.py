"""
Phase C1: Differential Privacy Integration
Using PyTorch Opacus PrivacyEngine to add Gaussian noise to gradients.
Provides σ²=0.5 (ε=2.5) privacy budget as specified in paper.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class DPPrivacyEngine:
    """
    Wrapper around Opacus PrivacyEngine for easy DP-SGD integration.
    Handles gradient clipping, noise addition, and privacy accounting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader,
        noise_multiplier: float = config.DP_NOISE_MULTIPLIER,
        max_grad_norm: float = config.DP_MAX_GRAD_NORM,
        target_epsilon: float = config.DP_TARGET_EPSILON,
        target_delta: float = 1e-5,
    ):
        """
        Initialize DP engine.
        
        Args:
            model: PyTorch model to protect
            optimizer: Optimizer instance
            train_dataloader: DataLoader for privacy accounting
            noise_multiplier: Noise multiplier σ. For σ²=0.5, noise_multiplier = sqrt(0.5) ≈ 0.707
            max_grad_norm: Gradient clipping norm (L2 clipping threshold)
            target_epsilon: Target privacy budget (per paper: ε=2.5)
            target_delta: Target failure probability (default 1e-5)
        """
        self.model = model
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        # Initialize Opacus PrivacyEngine
        try:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_dataloader = (
                self.privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                )
            )
            logger.info(
                f"PrivacyEngine initialized with noise_multiplier={noise_multiplier}, "
                f"max_grad_norm={max_grad_norm}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PrivacyEngine: {e}")
            raise
    
    def get_privacy_budget(self) -> Tuple[float, float]:
        """
        Get current privacy budget (epsilon, delta) accounting.
        
        Returns:
            Tuple of (epsilon, delta) privacy budget
        """
        try:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.target_delta)
            return epsilon, self.target_delta
        except Exception as e:
            logger.warning(f"Could not get privacy budget: {e}")
            return float('inf'), self.target_delta
    
    def make_private_with_eps(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader,
        target_epsilon: float,
        target_delta: float = 1e-5,
        num_epochs: int = 1,
    ) -> Tuple[nn.Module, optim.Optimizer]:
        """
        Alternative initialization targeting specific epsilon value.
        Auto-adjusts noise_multiplier to achieve target ε.
        
        Args:
            model: Model to privatize
            optimizer: Optimizer
            train_dataloader: Data loader
            target_epsilon: Target privacy budget
            target_delta: Target failure probability
            num_epochs: Number of epochs for privacy accounting
        
        Returns:
            Tuple of (private_model, private_optimizer)
        """
        privacy_engine = PrivacyEngine()
        private_model, private_optimizer, private_dataloader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=num_epochs,
            )
        )
        
        epsilon_achieved = privacy_engine.accountant.get_epsilon(delta=target_delta)
        logger.info(
            f"PrivacyEngine configured for target ε={target_epsilon}, "
            f"achieved ε={epsilon_achieved:.4f}"
        )
        
        return private_model, private_optimizer


def wrap_model_with_dp(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader,
    dp_enabled: bool = config.DP_ENABLED,
    noise_multiplier: float = config.DP_NOISE_MULTIPLIER,
    max_grad_norm: float = config.DP_MAX_GRAD_NORM,
) -> Tuple[nn.Module, optim.Optimizer, Optional[DPPrivacyEngine]]:
    """
    Convenience function to optionally wrap model with DP.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        train_dataloader: Data loader
        dp_enabled: Whether to enable DP
        noise_multiplier: Noise multiplier
        max_grad_norm: Gradient clipping norm
    
    Returns:
        Tuple of (model, optimizer, dp_engine or None)
    """
    if not dp_enabled:
        logger.info("DP disabled - model will train without privacy protection")
        return model, optimizer, None
    
    dp_engine = DPPrivacyEngine(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    
    return dp_engine.model, dp_engine.optimizer, dp_engine


if __name__ == "__main__":
    import torch.utils.data as data_utils
    from src.models import get_model
    
    # Test DP wrapping
    model = get_model("resnet50", num_classes=2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy dataloader
    dummy_data = torch.randn(32, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (32,))
    dataset = data_utils.TensorDataset(dummy_data, dummy_labels)
    dataloader = data_utils.DataLoader(dataset, batch_size=8)
    
    # Wrap with DP
    private_model, private_optimizer, dp_engine = wrap_model_with_dp(
        model, optimizer, dataloader, dp_enabled=True
    )
    
    if dp_engine:
        epsilon, delta = dp_engine.get_privacy_budget()
        print(f"Privacy budget: ε={epsilon:.4f}, δ={delta}")
