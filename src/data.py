"""
Phase A2: PyTorch Data Module
TBChestXrayDataset class with ImageNet augmentation/normalization.
"""

from pathlib import Path
from typing import Optional, Callable, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles both tensor and PIL Image inputs, converting PIL to tensor if needed.
    This is a fallback for compatibility with SMOTE augmented data.
    """
    from torchvision import transforms as T
    
    images, labels = zip(*batch)
    
    # Convert to tensor if needed
    processed_images = []
    pil_to_tensor = T.ToTensor()  # Fallback converter
    
    for img in images:
        if isinstance(img, torch.Tensor):
            processed_images.append(img)
        elif isinstance(img, Image.Image):
            # Convert PIL to tensor
            logger.warning(f"Converting PIL image to tensor (should have been done by transform)")
            tensor_img = pil_to_tensor(img)
            processed_images.append(tensor_img)
        else:
            raise TypeError(f"Unexpected image type in collate: {type(img)}")
    
    images_tensor = torch.stack(processed_images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return images_tensor, labels_tensor


class TBChestXrayDataset(Dataset):
    """
    Custom PyTorch Dataset for TB Chest X-ray images.
    
    Loads organized TB dataset (train/val/test splits with Normal/Tuberculosis classes).
    Applies ImageNet normalization and optional augmentation.
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        include_augmented: bool = False,
    ):
        """
        Args:
            root_dir: Path to organized dataset folder (e.g., data/tb_organized)
            split: One of 'train', 'val', 'test'
            transform: Optional image transformation pipeline
            target_transform: Optional label transformation
            include_augmented: Whether to include pixel-space augmented samples from disk
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be one of 'train', 'val', 'test', got {split}")
        
        # Class mapping
        self.class_to_idx = {
            "Normal": 0,
            "Tuberculosis": 1,
        }
        
        # Collect all images
        self.samples = []  # List of (image_path, label) tuples
        
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            images = sorted(class_dir.glob("*.png"))
            for img_path in images:
                self.samples.append((img_path, class_idx))
        
        # For training split, optionally load augmented (legacy pixel-space SMOTE) samples.
        if split == "train" and include_augmented:
            augmented_tb_dir = split_dir / "Tuberculosis_augmented"
            if augmented_tb_dir.exists():
                augmented_images = sorted(augmented_tb_dir.glob("*.png"))
                for img_path in augmented_images:
                    self.samples.append((img_path, 1))  # class_idx for TB = 1
                logger.info(
                    f"Loaded {len(augmented_images)} augmented (SMOTE) TB images from "
                    f"{augmented_tb_dir}"
                )
            else:
                logger.info(
                    f"No augmented TB data found at {augmented_tb_dir}. "
                    f"Run prepare_data.py to generate SMOTE augmentation."
                )
        
        logger.info(
            f"Loaded {len(self.samples)} total images from {split} split "
            f"({split_dir})"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            Tuple of (image_tensor, label) where:
                - image_tensor: shape (3, 224, 224), normalized with ImageNet stats
                - label: int (0 for Normal, 1 for TB)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")  # Convert grayscale to RGB if needed
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        # Apply target transform
        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """
    Training augmentation pipeline (with augmentation).
    Per ImageNet preprocessing: resize, crop, horizontal flip, normalize.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    """
    Validation/test augmentation pipeline (no augmentation, only resize & normalize).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])


def create_dataloaders(
    root_dir: Path = config.TB_ORGANIZED_ROOT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for TB dataset.
    
    Args:
        root_dir: Path to organized dataset
        batch_size: Batch size (default 32 per paper)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (useful for GPU)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets with appropriate transforms
    train_dataset = TBChestXrayDataset(
        root_dir, split="train", transform=get_train_transform()
    )
    val_dataset = TBChestXrayDataset(
        root_dir, split="val", transform=get_val_transform()
    )
    test_dataset = TBChestXrayDataset(
        root_dir, split="test", transform=get_val_transform()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,  # Custom collate to handle both tensors and PIL images
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,  # Custom collate to handle both tensors and PIL images
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,  # Custom collate to handle both tensors and PIL images
    )
    
    logger.info(
        f"DataLoaders created: train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader)} batches"
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test of dataset loading
    dataset = TBChestXrayDataset(
        config.TB_ORGANIZED_ROOT, split="train", transform=get_train_transform()
    )
    print(f"Dataset size: {len(dataset)}")
    
    img, label = dataset[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample label: {label} (0=Normal, 1=TB)")
