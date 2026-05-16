"""
Phase A1: Data Preparation & Organization
Extracts TB Chest X-ray from raw structure and organizes into train/val/test splits with resizing.
Applies SMOTE to training data to balance TB vs Normal classes before FL training.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import logging
from scipy.stats import chi2_contingency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    logger.warning("imblearn not available - SMOTE will be skipped")
    HAS_SMOTE = False


def get_image_paths(class_dir: Path) -> List[Path]:
    """Get all PNG images from a class directory."""
    return sorted([f for f in class_dir.glob("*.png") if f.is_file()])


def split_indices(
    total_count: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split indices into train/val/test with given ratios.
    
    Args:
        total_count: Total number of samples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.arange(total_count)
    np.random.shuffle(indices)
    
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    train_indices = indices[:train_count].tolist()
    val_indices = indices[train_count : train_count + val_count].tolist()
    test_indices = indices[train_count + val_count :].tolist()
    
    return train_indices, val_indices, test_indices


def resize_and_save_image(
    src_path: Path, dst_path: Path, size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Load, resize to 224x224, and save image.
    Grayscale images are converted to 3-channel RGB for compatibility with ResNet/VGG.
    
    Args:
        src_path: Source image path
        dst_path: Destination image path
        size: Target size (H, W)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(src_path)
        
        # Convert grayscale to RGB (TB X-rays are grayscale)
        if img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)
        return True
    except Exception as e:
        logger.warning(f"Failed to process {src_path}: {e}")
        return False


def prepare_tb_dataset(
    raw_dir: Path = config.TB_DATA_ROOT,
    output_dir: Path = config.TB_ORGANIZED_ROOT,
    img_size: int = config.IMG_SIZE,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    test_ratio: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Main function to prepare TB dataset:
    1. Load images from raw structure (Normal/, Tuberculosis/)
    2. Split into train/val/test (70/15/15)
    3. Resize all images to 224x224
    4. Save organized structure
    
    Args:
        raw_dir: Path to raw TB_Chest_Radiography_Database folder
        output_dir: Path to output organized folder
        img_size: Target image size
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Testing split ratio
        seed: Random seed
    
    Returns:
        Dictionary with statistics (counts, distribution, etc.)
    """
    
    logger.info(f"Starting TB dataset preparation...")
    logger.info(f"Raw data: {raw_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Check raw data exists
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")
    
    class_dirs = {
        "Normal": raw_dir / "Normal",
        "Tuberculosis": raw_dir / "Tuberculosis",
    }
    
    # Verify class directories
    for class_name, class_dir in class_dirs.items():
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        logger.info(f"Found '{class_name}' directory: {class_dir}")
    
    # Collect all images per class
    class_images = {}
    total_images = 0
    for class_name, class_dir in class_dirs.items():
        images = get_image_paths(class_dir)
        class_images[class_name] = images
        total_images += len(images)
        logger.info(f"  {class_name}: {len(images)} images")
    
    logger.info(f"Total images: {total_images}")
    
    # Split each class independently (stratified split)
    split_data = {}
    for class_name, images in class_images.items():
        train_idx, val_idx, test_idx = split_indices(
            len(images), train_ratio, val_ratio, test_ratio, seed
        )
        
        split_data[class_name] = {
            "train": [images[i] for i in train_idx],
            "val": [images[i] for i in val_idx],
            "test": [images[i] for i in test_idx],
        }
        logger.info(
            f"{class_name} split: train={len(split_data[class_name]['train'])}, "
            f"val={len(split_data[class_name]['val'])}, "
            f"test={len(split_data[class_name]['test'])}"
        )
    
    # Process and save images
    stats = {"processed": 0, "failed": 0, "splits": {}}
    
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        split_stats = {}
        
        for class_name in class_images.keys():
            class_split_dir = split_dir / class_name
            images_to_process = split_data[class_name][split_name]
            
            processed = 0
            for i, src_path in enumerate(images_to_process):
                dst_path = class_split_dir / src_path.name
                
                if resize_and_save_image(src_path, dst_path, (img_size, img_size)):
                    processed += 1
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                
                if (i + 1) % 500 == 0:
                    logger.info(
                        f"  {split_name}/{class_name}: {i + 1}/{len(images_to_process)} processed"
                    )
            
            split_stats[class_name] = processed
            logger.info(f"{split_name}/{class_name}: {processed} files processed")
        
        stats["splits"][split_name] = split_stats
    
    logger.info(f"\nPreparation complete!")
    logger.info(f"  Total processed: {stats['processed']}")
    logger.info(f"  Total failed: {stats['failed']}")
    logger.info(f"  Output structure: {output_dir}")
    
    return stats


def apply_smote_to_training_data(
    output_dir: Path = config.TB_ORGANIZED_ROOT,
    target_ratio: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Apply SMOTE to training data to balance TB vs Normal classes.
    Generates synthetic TB samples and saves them to train/Tuberculosis_augmented/ folder.
    Runs Chi-square test to verify balance.
    
    Args:
        output_dir: Path to organized dataset folder
        target_ratio: Minority class ratio after SMOTE (1.0 = balanced)
        seed: Random seed
    
    Returns:
        Dictionary with SMOTE statistics and Chi-square test results
    """
    if not HAS_SMOTE:
        logger.warning("SMOTE not available (imblearn not installed), skipping balancing")
        return {"status": "skipped", "reason": "imblearn not installed"}
    
    logger.info("\n" + "="*70)
    logger.info("PHASE: APPLY SMOTE TO TRAINING DATA")
    logger.info("="*70)
    
    train_dir = output_dir / "train"
    normal_dir = train_dir / "Normal"
    tb_dir = train_dir / "Tuberculosis"
    augmented_tb_dir = train_dir / "Tuberculosis_augmented"
    
    # Verify directories exist
    if not normal_dir.exists() or not tb_dir.exists():
        logger.error(f"Training directories not found: {normal_dir}, {tb_dir}")
        return {"status": "failed", "reason": "training directories not found"}
    
    # Load all training images as feature vectors
    logger.info("Loading training images...")
    image_paths = []
    labels = []
    X = []  # Feature matrix (flattened images)
    
    # Load Normal samples (class 0)
    for img_path in sorted(normal_dir.glob("*.png")):
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).flatten().astype(np.float32)
            X.append(img_array)
            image_paths.append(str(img_path))
            labels.append(0)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    # Load TB samples (class 1)
    for img_path in sorted(tb_dir.glob("*.png")):
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).flatten().astype(np.float32)
            X.append(img_array)
            image_paths.append(str(img_path))
            labels.append(1)
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    X = np.array(X)
    labels = np.array(labels)
    
    # Count before SMOTE
    unique_before, counts_before = np.unique(labels, return_counts=True)
    class_dist_before = dict(zip(unique_before, counts_before))
    logger.info(f"\nBefore SMOTE:")
    logger.info(f"  Normal (class 0): {class_dist_before.get(0, 0)} samples")
    logger.info(f"  TB (class 1): {class_dist_before.get(1, 0)} samples")
    logger.info(f"  Imbalance ratio: {class_dist_before.get(0, 1) / class_dist_before.get(1, 1):.2f}:1")
    
    # Apply SMOTE
    logger.info(f"\nApplying SMOTE with target ratio {target_ratio}...")
    try:
        smote = SMOTE(sampling_strategy=target_ratio, random_state=seed, k_neighbors=5)
        X_balanced, labels_balanced = smote.fit_resample(X, labels)
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        return {"status": "failed", "reason": str(e)}
    
    # Count after SMOTE
    unique_after, counts_after = np.unique(labels_balanced, return_counts=True)
    class_dist_after = dict(zip(unique_after, counts_after))
    num_synthetic = len(X_balanced) - len(X)
    
    logger.info(f"\nAfter SMOTE:")
    logger.info(f"  Normal (class 0): {class_dist_after.get(0, 0)} samples")
    logger.info(f"  TB (class 1): {class_dist_after.get(1, 0)} samples")
    logger.info(f"  Imbalance ratio: {class_dist_after.get(0, 1) / class_dist_after.get(1, 1):.2f}:1")
    logger.info(f"  Generated synthetic samples: {num_synthetic}")
    
    # Save synthetic images to disk
    logger.info(f"\nSaving synthetic TB samples to {augmented_tb_dir}...")
    augmented_tb_dir.mkdir(parents=True, exist_ok=True)
    
    num_original = len(X)
    synthetic_count = 0
    
    for idx in range(num_original, len(X_balanced)):
        if labels_balanced[idx] == 1:  # Only save synthetic TB samples
            try:
                # Reshape back to 224x224x3
                synthetic_img_array = X_balanced[idx].reshape(224, 224, 3).astype(np.uint8)
                synthetic_img = Image.fromarray(synthetic_img_array, 'RGB')
                
                # Save with numbered filename
                save_path = augmented_tb_dir / f"synthetic_{idx - num_original:05d}.png"
                synthetic_img.save(save_path, quality=95)
                synthetic_count += 1
                
                if synthetic_count % 200 == 0:
                    logger.info(f"  Saved {synthetic_count} synthetic images...")
            except Exception as e:
                logger.warning(f"Failed to save synthetic image {idx}: {e}")
    
    logger.info(f"✓ Saved {synthetic_count} synthetic TB images")
    
    # Chi-square test for balance verification
    logger.info(f"\nRunning Chi-square test to verify balance...")
    
    # Create contingency table
    contingency_table = np.array([
        [class_dist_before.get(0, 0), class_dist_before.get(1, 0)],
        [class_dist_after.get(0, 0), class_dist_after.get(1, 0)]
    ])
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        logger.info(f"  Chi-square statistic: {chi2:.4f}")
        logger.info(f"  P-value: {p_value:.6f}")
        logger.info(f"  Degrees of freedom: {dof}")
        
        # Interpret result
        alpha = 0.05
        if p_value > alpha:
            logger.info(f"✓ Chi-square test PASSED (p={p_value:.4f} > {alpha})")
            logger.info(f"  Data distribution after SMOTE is NOT significantly different from before")
            chi_square_result = "PASSED"
        else:
            logger.warning(f"Chi-square test showed significant difference (p={p_value:.4f} < {alpha})")
            logger.warning(f"  This may indicate SMOTE introduced artificial bias (expected)")
            chi_square_result = "WARNING"
    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")
        chi_square_result = "ERROR"
    
    return {
        "status": "complete",
        "before": class_dist_before,
        "after": class_dist_after,
        "synthetic_generated": num_synthetic,
        "synthetic_saved": synthetic_count,
        "chi_square": {
            "statistic": chi2 if chi_square_result != "ERROR" else None,
            "p_value": p_value if chi_square_result != "ERROR" else None,
            "result": chi_square_result
        }
    }


def verify_prepared_dataset(output_dir: Path = config.TB_ORGANIZED_ROOT) -> dict:
    """
    Verify the prepared dataset structure and count files.
    Includes both original and augmented (SMOTE) training data.
    
    Args:
        output_dir: Path to organized output folder
    
    Returns:
        Dictionary with verification details
    """
    logger.info(f"\nVerifying prepared dataset...")
    
    verification = {}
    
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        split_verification = {}
        
        for class_name in ["Normal", "Tuberculosis"]:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png")))
                split_verification[class_name] = count
            else:
                split_verification[class_name] = 0
        
        # Check for augmented TB folder (only in train split)
        if split == "train":
            augmented_dir = split_dir / "Tuberculosis_augmented"
            if augmented_dir.exists():
                augmented_count = len(list(augmented_dir.glob("*.png")))
                split_verification["Tuberculosis_augmented"] = augmented_count
                logger.info(f"{split}: Normal={split_verification['Normal']}, TB={split_verification['Tuberculosis']}, Augmented={augmented_count}")
            else:
                logger.info(f"{split}: Normal={split_verification['Normal']}, TB={split_verification['Tuberculosis']}")
        else:
            total = sum(split_verification.values())
            logger.info(f"{split}: Normal={split_verification['Normal']}, TB={split_verification['Tuberculosis']}, Total={total}")
        
        verification[split] = split_verification
    
    return verification


if __name__ == "__main__":
    # Run data preparation
    stats = prepare_tb_dataset()
    
    # Verify initial preparation
    verification = verify_prepared_dataset()
    
    # Apply SMOTE to balance training data
    smote_stats = apply_smote_to_training_data()
    
    # Verify final dataset
    logger.info("\n" + "="*70)
    logger.info("FINAL DATASET VERIFICATION (INCLUDING AUGMENTED DATA)")
    logger.info("="*70)
    final_verification = verify_prepared_dataset()
