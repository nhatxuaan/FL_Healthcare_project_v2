"""
Phase B: Model Definition & Transfer Learning (FIXED)

Changes vs original:
- FC head now includes Dropout (p=0.4) before the linear layer for both
  ResNet50 and VGG16. Without dropout, a 2048→2 linear layer sitting on top
  of frozen ImageNet features can perfectly separate any small dataset
  (capacity >> data), which is the primary structural cause of overfitting.
- Backbone freeze strategy changed from "freeze everything" to "freeze only
  layer1 & layer2 (low-level edge/texture detectors), fine-tune layer3 & layer4
  (high-level semantic features)". Chest X-ray pathology is encoded in those
  high-level features, which were originally learned on RGB natural images —
  they need to be adapted to greyscale-converted medical images.
- `freeze_backbone=True` now means partial freeze (layer1+layer2 only).
  Pass `freeze_backbone=False` to fine-tune the entire network.
- All other functions (get_model, count_trainable_parameters) are unchanged.
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
    num_classes: int        = config.NUM_CLASSES,
    pretrained: bool        = True,
    freeze_backbone: bool   = config.FREEZE_BACKBONE,
    dropout_p: float        = 0.4,
) -> nn.Module:
    """
    ResNet50 with ImageNet weights, Dropout head, and partial backbone freeze.

    Architecture of the new classification head:
        AveragePool(2048) → Dropout(0.4) → Linear(2048, num_classes)

    Freeze strategy when freeze_backbone=True:
        Frozen  : conv1, bn1, layer1, layer2  (low-level edge/texture)
        Trainable: layer3, layer4, fc          (high-level semantics + classifier)

    This lets the network adapt its high-level representations to chest X-ray
    while keeping most parameters fixed, reducing overfitting on small datasets.

    Args:
        num_classes   : Output classes (2 for TB binary classification)
        pretrained    : Whether to load ImageNet weights
        freeze_backbone: True = partial freeze (layer1+2); False = all trainable
        dropout_p     : Dropout probability before FC (default 0.4)
    """
    logger.info("Loading ResNet50 with ImageNet pre-trained weights...")

    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet50(weights=weights)

    in_features = model.fc.in_features   # 2048

    # --- New head: Dropout + Linear ---
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    logger.info(
        f"ResNet50 head: Dropout({dropout_p}) → Linear({in_features}, {num_classes})"
    )

    if freeze_backbone:
        # Freeze only low-level layers; keep high-level layers trainable
        _freeze_layers = [model.conv1, model.bn1, model.layer1, model.layer2]
        for module in _freeze_layers:
            for p in module.parameters():
                p.requires_grad = False

        _trainable_layers = [model.layer3, model.layer4, model.fc]
        for module in _trainable_layers:
            for p in module.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger.info(
            f"ResNet50 partial freeze: conv1+bn1+layer1+layer2 frozen, "
            f"layer3+layer4+fc trainable | "
            f"trainable params: {trainable:,} / {total:,}"
        )
    else:
        logger.info("ResNet50: all layers trainable (no freeze)")

    return model


def create_vgg16(
    num_classes: int        = config.NUM_CLASSES,
    pretrained: bool        = True,
    freeze_backbone: bool   = config.FREEZE_BACKBONE,
    dropout_p: float        = 0.5,
) -> nn.Module:
    """
    VGG16 with ImageNet weights, extra Dropout in classifier, partial feature freeze.

    VGG16 already has Dropout(0.5) in its default classifier. We reinforce this
    by keeping those drops AND replacing the final layer with Dropout + Linear.

    Freeze strategy when freeze_backbone=True:
        Frozen  : features[0..20]  (first 4 conv blocks)
        Trainable: features[21..]  (last conv block) + classifier

    Args:
        num_classes   : Output classes
        pretrained    : Whether to load ImageNet weights
        freeze_backbone: True = partial freeze; False = all trainable
        dropout_p     : Dropout probability before the final linear layer
    """
    logger.info("Loading VGG16 with ImageNet pre-trained weights...")

    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.vgg16(weights=weights)

    # Replace the final classifier layer with Dropout + Linear
    in_features = model.classifier[-1].in_features   # 4096
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    logger.info(
        f"VGG16 head: Dropout({dropout_p}) → Linear({in_features}, {num_classes})"
    )

    if freeze_backbone:
        # Freeze first 4 conv blocks (indices 0-20 in features Sequential)
        # Trainable: block 5 (indices 21-28) + classifier
        for i, layer in enumerate(model.features):
            for p in layer.parameters():
                p.requires_grad = (i >= 21)

        for p in model.classifier.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger.info(
            f"VGG16 partial freeze: features[0-20] frozen, "
            f"features[21+]+classifier trainable | "
            f"trainable params: {trainable:,} / {total:,}"
        )
    else:
        logger.info("VGG16: all layers trainable (no freeze)")

    return model


def get_model(
    model_name: str       = "resnet50",
    num_classes: int      = config.NUM_CLASSES,
    pretrained: bool      = True,
    freeze_backbone: bool = config.FREEZE_BACKBONE,
) -> nn.Module:
    """
    Factory: return model by name. Signature unchanged from original.

    Args:
        model_name    : 'resnet50' or 'vgg16'
        num_classes   : Number of output classes
        pretrained    : Use ImageNet pre-trained weights
        freeze_backbone: Partial freeze (True) or full fine-tune (False)
    """
    name = model_name.lower()
    if name == "resnet50":
        return create_resnet50(num_classes, pretrained, freeze_backbone)
    elif name == "vgg16":
        return create_vgg16(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from ['resnet50', 'vgg16']")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ["resnet50", "vgg16"]:
        model = get_model(name, num_classes=2, pretrained=True, freeze_backbone=True)
        trainable = count_trainable_parameters(model)
        total     = sum(p.numel() for p in model.parameters())
        print(f"\n{name}:")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params   : {total - trainable:,}")
        dummy  = torch.randn(2, 3, 224, 224)
        output = model(dummy)
        print(f"  Output shape    : {output.shape}")