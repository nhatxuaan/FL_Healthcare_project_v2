"""
Phase B: Model Definition & Transfer Learning
ResNet50 and VGG16 with ImageNet pre-trained weights, modified for binary TB classification.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, VGG16_Weights
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def create_resnet50(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = config.FREEZE_BACKBONE,
) -> nn.Module:
    """
    Create ResNet50 with ImageNet pre-trained weights.
    Modifies the final FC layer for the target number of classes.
    
    Args:
        num_classes: Number of output classes (2 for TB binary classification)
        pretrained: Whether to load ImageNet pre-trained weights
        freeze_backbone: Whether to freeze feature extraction layers
    
    Returns:
        ResNet50 model ready for training
    """
    logger.info("Loading ResNet50 with ImageNet pre-trained weights...")
    
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
    
    # Modify final FC layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    logger.info(f"ResNet50 modified: final FC layer = {in_features} -> {num_classes}")
    
    # Freeze backbone if needed (only train FC)
    if freeze_backbone:
        # Freeze all convolutional and batch norm layers in the backbone
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        logger.info("ResNet50 backbone frozen: conv1, bn1, layer1-4 (only FC layer trainable)")
    
    return model


def create_vgg16(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = config.FREEZE_BACKBONE,
) -> nn.Module:
    """
    Create VGG16 with ImageNet pre-trained weights.
    Modifies the final classifier for the target number of classes.
    
    Args:
        num_classes: Number of output classes (2 for TB binary classification)
        pretrained: Whether to load ImageNet pre-trained weights
        freeze_backbone: Whether to freeze feature extraction layers
    
    Returns:
        VGG16 model ready for training
    """
    logger.info("Loading VGG16 with ImageNet pre-trained weights...")
    
    if pretrained:
        weights = VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    else:
        model = models.vgg16(weights=None)
    
    # Modify final classifier for binary classification
    # VGG16 classifier: Linear(512, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, num_classes)
    # We replace the last layer
    original_fc = model.classifier[-1]
    in_features = original_fc.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    logger.info(f"VGG16 modified: final classifier layer = {in_features} -> {num_classes}")
    
    # Freeze backbone (features) if needed
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("VGG16 feature extraction layers frozen (only classifier trainable)")
    
    return model


def get_model(
    model_name: str = "resnet50",
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = config.FREEZE_BACKBONE,
) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        model_name: One of ['resnet50', 'vgg16']
        num_classes: Number of output classes
        pretrained: Whether to load pre-trained weights
        freeze_backbone: Whether to freeze feature extraction layers
    
    Returns:
        Requested model
    """
    if model_name.lower() == "resnet50":
        return create_resnet50(num_classes, pretrained, freeze_backbone)
    elif model_name.lower() == "vgg16":
        return create_vgg16(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from ['resnet50', 'vgg16']")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count and return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    
    # Test model creation and parameter counting
    for model_name in ["resnet50", "vgg16"]:
        model = get_model(model_name, num_classes=2, pretrained=True, freeze_backbone=True)
        trainable_params = count_trainable_parameters(model)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n{model_name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
