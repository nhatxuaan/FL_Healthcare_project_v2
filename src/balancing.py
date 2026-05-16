"""
SMOTE-based Data Balancing for FL Healthcare
Applies SMOTE to balance TB vs Normal class at client level
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class SMOTEBalancer:
    """
    Apply SMOTE to balance imbalanced datasets at client level.
    Generates synthetic TB samples to match Normal sample count.
    """
    
    def __init__(self, target_ratio=1.0, random_state=42):
        """
        Args:
            target_ratio: Ratio of minority to majority class after SMOTE (default 1.0 = balanced)
            random_state: Random seed for reproducibility
        """
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.smote = None
    
    def apply_smote(self, image_paths, labels):
        """
        Apply SMOTE to balance image dataset.
        
        Args:
            image_paths: List of full paths to images
            labels: List of labels (0=Normal, 1=TB)
        
        Returns:
            balanced_image_paths: List of paths (original + references to synthetic)
            balanced_labels: Corresponding labels
            synthetic_images: Dict of synthetic image arrays (key: synthetic index, value: numpy array)
        """
        
        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        logger.info(f"Before SMOTE: {class_dist}")
        
        # If already balanced, return as-is
        if len(unique) < 2 or min(counts) / max(counts) > self.target_ratio:
            logger.warning("Dataset already balanced or single class, skipping SMOTE")
            return list(image_paths), list(labels), {}
        
        # Load images as feature vectors (flatten pixel values)
        X = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_array = np.array(img).flatten().astype(np.float32)
            X.append(img_array)
        X = np.array(X)
        
        logger.info(f"Feature matrix shape: {X.shape} (images flattened to vectors)")
        
        # Apply SMOTE with sampling strategy to balance classes
        sampling_strategy = self.target_ratio
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        
        try:
            X_balanced, y_balanced = smote.fit_resample(X, labels)
        except Exception as e:
            logger.error(f"SMOTE failed: {e}. Returning original data.")
            return list(image_paths), list(labels), {}
        
        # Identify synthetic samples (indices >= len(original))
        num_original = len(image_paths)
        synthetic_indices = np.arange(num_original, len(X_balanced))
        
        logger.info(f"Generated {len(synthetic_indices)} synthetic samples via SMOTE")
        logger.info(f"After SMOTE: {np.bincount(y_balanced)}")
        
        # Create balanced image paths list
        balanced_paths = list(image_paths)
        synthetic_images = {}
        
        for idx in synthetic_indices:
            # Store synthetic image array
            synthetic_key = f"synthetic_{idx}"
            # Reshape back from flattened vector
            synthetic_img = X_balanced[idx].reshape(224, 224, 3).astype(np.uint8)
            synthetic_images[idx] = synthetic_img
            balanced_paths.append(synthetic_key)
        
        balanced_labels = list(y_balanced)
        
        logger.info(f"Balanced dataset: {len(balanced_paths)} samples (original: {num_original})")
        
        return balanced_paths, balanced_labels, synthetic_images


def balance_client_dataset(image_paths, labels, random_state=42):
    """
    Helper function to apply SMOTE to a client's dataset.
    
    Args:
        image_paths: List of file paths
        labels: List of labels
        random_state: For reproducibility
    
    Returns:
        balanced_paths: Balanced image paths
        balanced_labels: Balanced labels
        synthetic_images: Synthetic image cache
    """
    balancer = SMOTEBalancer(target_ratio=1.0, random_state=random_state)
    return balancer.apply_smote(image_paths, labels)
