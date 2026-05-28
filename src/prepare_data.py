"""
Phase A1: Data Preparation & Organization (FIXED)

Changes vs original:
- SMOTE is now applied ONLY to training fold, AFTER the train/val/test split is finalized.
  This eliminates the data-leakage where synthetic interpolations of training images
  were placed next to val/test neighbours drawn from the same original pool.
- Pixel-space SMOTE (flatten 224x224x3 → 150k-d vector) is replaced with
  feature-space SMOTE using a frozen ResNet50 backbone (2048-d).
  Interpolating in a pretrained feature space produces semantically coherent
  synthetic samples instead of blurry pixel averages.
- The old `apply_smote_to_training_data()` that operated on the already-saved
  folder structure is replaced by `apply_feature_smote_to_train_split()`, which
  computes features on-the-fly and saves synthetic images back to disk only AFTER
  the split is locked.
- Chi-square verification is retained.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import logging
from scipy.stats import chi2_contingency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    logger.warning("imblearn not available — SMOTE will be skipped")
    HAS_SMOTE = False

try:
    import torch
    import torchvision.transforms as T
    from torchvision import models
    from torchvision.models import ResNet50_Weights
    HAS_TORCH = True
except ImportError:
    logger.warning("torch not available — feature-space SMOTE will be skipped")
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_paths(class_dir: Path) -> List[Path]:
    return sorted([f for f in class_dir.glob("*.png") if f.is_file()])


def split_indices(
    total_count: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split indices into train/val/test.
    Stratification is handled at the call site (per-class split), so this
    function operates on a single class at a time.
    """
    np.random.seed(seed)
    indices = np.arange(total_count)
    np.random.shuffle(indices)

    train_count = int(total_count * train_ratio)
    val_count   = int(total_count * val_ratio)

    train_idx = indices[:train_count].tolist()
    val_idx   = indices[train_count : train_count + val_count].tolist()
    test_idx  = indices[train_count + val_count :].tolist()

    return train_idx, val_idx, test_idx


def resize_and_save_image(
    src_path: Path, dst_path: Path, size: Tuple[int, int] = (224, 224)
) -> bool:
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)
        return True
    except Exception as e:
        logger.warning(f"Failed to process {src_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 1: organize raw images into train / val / test folders
# ---------------------------------------------------------------------------

def prepare_tb_dataset(
    raw_dir: Path  = config.TB_DATA_ROOT,
    output_dir: Path = config.TB_ORGANIZED_ROOT,
    img_size: int    = config.IMG_SIZE,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float   = config.VAL_SPLIT,
    test_ratio: float  = config.TEST_SPLIT,
    seed: int          = config.RANDOM_SEED,
) -> dict:
    """
    Organize raw TB dataset into stratified train/val/test splits.
    SMOTE is NOT applied here — it runs in a separate step after this function
    returns, ensuring zero leakage between splits.
    """
    logger.info("Starting TB dataset preparation...")
    logger.info(f"Raw data : {raw_dir}")
    logger.info(f"Output   : {output_dir}")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")

    class_dirs = {
        "Normal":       raw_dir / "Normal",
        "Tuberculosis": raw_dir / "Tuberculosis",
    }
    for cname, cdir in class_dirs.items():
        if not cdir.exists():
            raise FileNotFoundError(f"Class directory not found: {cdir}")
        logger.info(f"Found '{cname}' directory: {cdir}")

    # Collect per-class image lists
    class_images: Dict[str, List[Path]] = {}
    for cname, cdir in class_dirs.items():
        images = get_image_paths(cdir)
        class_images[cname] = images
        logger.info(f"  {cname}: {len(images)} images")

    # --- Stratified split: each class is split independently ---
    split_data: Dict[str, Dict[str, List[Path]]] = {}
    for cname, images in class_images.items():
        tr_idx, va_idx, te_idx = split_indices(
            len(images), train_ratio, val_ratio, test_ratio, seed
        )
        split_data[cname] = {
            "train": [images[i] for i in tr_idx],
            "val":   [images[i] for i in va_idx],
            "test":  [images[i] for i in te_idx],
        }
        logger.info(
            f"{cname} — train: {len(split_data[cname]['train'])}, "
            f"val: {len(split_data[cname]['val'])}, "
            f"test: {len(split_data[cname]['test'])}"
        )

    # --- Copy and resize images to output structure ---
    stats = {"processed": 0, "failed": 0, "splits": {}}

    for split_name in ["train", "val", "test"]:
        split_stats = {}
        for cname in class_images:
            class_split_dir = output_dir / split_name / cname
            images_to_copy  = split_data[cname][split_name]
            processed = 0
            for i, src in enumerate(images_to_copy):
                dst = class_split_dir / src.name
                if resize_and_save_image(src, dst, (img_size, img_size)):
                    processed += 1
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                if (i + 1) % 500 == 0:
                    logger.info(f"  {split_name}/{cname}: {i+1}/{len(images_to_copy)} done")
            split_stats[cname] = processed
            logger.info(f"{split_name}/{cname}: {processed} files saved")
        stats["splits"][split_name] = split_stats

    logger.info(
        f"\nPreparation complete — processed: {stats['processed']}, "
        f"failed: {stats['failed']}"
    )
    return stats


# ---------------------------------------------------------------------------
# Step 2: feature-space SMOTE on the training split ONLY
# ---------------------------------------------------------------------------

def _build_feature_extractor(device: str = "cpu"):
    """
    Return a frozen ResNet50 backbone (no FC head) for feature extraction.
    Produces 2048-d embeddings per image.
    """
    backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the classification head — keep everything up to avgpool
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def _extract_features(
    image_paths: List[Path],
    backbone,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract 2048-d ResNet50 features for a list of image paths.
    Images are preprocessed with the standard ImageNet pipeline.
    """
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_feats = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception as e:
                logger.warning(f"Could not load {p}: {e}")
                # Use zeros as a fallback so indices stay aligned
                imgs.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = backbone(batch_tensor)           # (B, 2048, 1, 1)
            feats = feats.view(feats.size(0), -1)   # (B, 2048)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def apply_feature_smote_to_train_split(
    output_dir: Path = config.TB_ORGANIZED_ROOT,
    target_ratio: float = 1.0,
    seed: int = config.RANDOM_SEED,
    device: str = "cpu",
) -> dict:
    """
    Apply SMOTE in ResNet50 feature space to balance the TRAINING split only.

    Why feature-space instead of pixel-space:
      - Pixel-space SMOTE on 224x224x3 = 150,528-d vectors produces blurry
        averages that carry no valid medical signal.
      - Interpolating in 2048-d pretrained feature space produces synthetic
        embeddings that lie on the real image manifold, yielding genuinely
        plausible representations of TB pathology.

    The function saves synthetic TB images (decoded from feature-space
    neighbours via nearest-real-image proxy) to train/Tuberculosis_augmented/.
    Because this step runs AFTER the split is locked, there is zero leakage
    into val or test sets.

    NOTE: Feature-space SMOTE cannot perfectly reconstruct pixel images from
    embeddings. Instead, for each synthetic embedding we save the REAL nearest-
    neighbour TB image with a slight brightness/contrast jitter applied. This is
    a conservative but leak-free strategy that keeps all saved images medically
    valid while still expanding the minority class.
    """
    if not HAS_SMOTE:
        logger.warning("imblearn not installed — skipping SMOTE")
        return {"status": "skipped", "reason": "imblearn not installed"}
    if not HAS_TORCH:
        logger.warning("torch not installed — skipping feature-space SMOTE")
        return {"status": "skipped", "reason": "torch not installed"}

    logger.info("\n" + "=" * 70)
    logger.info("PHASE: FEATURE-SPACE SMOTE ON TRAINING SPLIT ONLY")
    logger.info("=" * 70)

    train_dir      = output_dir / "train"
    normal_dir     = train_dir / "Normal"
    tb_dir         = train_dir / "Tuberculosis"
    augmented_dir  = train_dir / "Tuberculosis_augmented"

    if not normal_dir.exists() or not tb_dir.exists():
        logger.error("Training class directories not found — run prepare_tb_dataset() first")
        return {"status": "failed", "reason": "training directories not found"}

    # --- Collect image paths and labels for training split only ---
    normal_paths = sorted(normal_dir.glob("*.png"))
    tb_paths     = sorted(tb_dir.glob("*.png"))

    all_paths  = list(normal_paths) + list(tb_paths)
    all_labels = [0] * len(normal_paths) + [1] * len(tb_paths)

    logger.info(f"Training split — Normal: {len(normal_paths)}, TB: {len(tb_paths)}")

    if len(tb_paths) == 0 or len(normal_paths) == 0:
        logger.error("One class is empty — cannot apply SMOTE")
        return {"status": "failed", "reason": "empty class"}

    # --- Extract features from TRAINING images only ---
    logger.info(f"Extracting ResNet50 features (device={device})...")
    backbone = _build_feature_extractor(device)
    X_feats  = _extract_features(all_paths, backbone, device)
    y_labels = np.array(all_labels)
    logger.info(f"Feature matrix: {X_feats.shape}")

    # --- Compute class balance ---
    unique, counts = np.unique(y_labels, return_counts=True)
    dist_before = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"Before SMOTE: {dist_before}")

    if min(counts) / max(counts) >= target_ratio:
        logger.info("Dataset already balanced — skipping SMOTE")
        return {"status": "skipped", "reason": "already balanced"}

    # --- Apply SMOTE in feature space ---
    logger.info("Applying SMOTE in 2048-d feature space...")
    smote = SMOTE(
        sampling_strategy=target_ratio,
        random_state=seed,
        k_neighbors=min(5, len(tb_paths) - 1),  # guard for tiny TB sets
    )
    try:
        X_balanced, y_balanced = smote.fit_resample(X_feats, y_labels)
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        return {"status": "failed", "reason": str(e)}

    unique_after, counts_after = np.unique(y_balanced, return_counts=True)
    dist_after   = dict(zip(unique_after.tolist(), counts_after.tolist()))
    num_synthetic = int(len(X_balanced) - len(X_feats))
    logger.info(f"After SMOTE: {dist_after} ({num_synthetic} synthetic TB embeddings generated)")

    # --- Proxy strategy: save jittered nearest-real-TB-image for each synthetic point ---
    # We find the real TB training image whose feature vector is closest to the
    # synthetic embedding, then apply mild colour jitter to distinguish it from the
    # original. This guarantees every saved image is medically valid.
    logger.info("Saving synthetic TB samples (nearest-real-TB proxy with jitter)...")
    augmented_dir.mkdir(parents=True, exist_ok=True)

    tb_feat_indices = np.where(y_labels == 1)[0]
    tb_features     = X_feats[tb_feat_indices]          # (N_tb, 2048)
    tb_image_paths  = [all_paths[i] for i in tb_feat_indices]

    synthetic_start = len(X_feats)
    synthetic_feats = X_balanced[synthetic_start:]       # only the new ones
    synthetic_labels_check = y_balanced[synthetic_start:]

    jitter_transform = T.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02
    )

    saved_count = 0
    for idx, (syn_feat, syn_label) in enumerate(
        zip(synthetic_feats, synthetic_labels_check)
    ):
        if syn_label != 1:
            continue  # should always be 1 for TB; guard just in case

        # Find nearest real TB image in feature space (L2)
        dists   = np.linalg.norm(tb_features - syn_feat, axis=1)
        nearest = int(np.argmin(dists))
        src_img_path = tb_image_paths[nearest]

        try:
            img = Image.open(src_img_path).convert("RGB")
            img = jitter_transform(img)
            save_path = augmented_dir / f"synthetic_{idx:05d}.png"
            img.save(save_path, quality=95)
            saved_count += 1
            if saved_count % 200 == 0:
                logger.info(f"  Saved {saved_count} synthetic images...")
        except Exception as e:
            logger.warning(f"Failed to save synthetic image {idx}: {e}")

    logger.info(f"✓ Saved {saved_count} synthetic TB images to {augmented_dir}")

    # --- Chi-square test for balance verification ---
    contingency = np.array([
        [dist_before.get(0, 0), dist_before.get(1, 0)],
        [dist_after.get(0, 0),  dist_after.get(1, 0)],
    ])
    chi2_result = "ERROR"
    chi2_val = p_val = None
    try:
        chi2_val, p_val, dof, _ = chi2_contingency(contingency)
        alpha = 0.05
        chi2_result = "PASSED" if p_val > alpha else "WARNING"
        logger.info(
            f"Chi-square test: χ²={chi2_val:.4f}, p={p_val:.6f}, "
            f"dof={dof} → {chi2_result}"
        )
    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")

    return {
        "status":            "complete",
        "before":            dist_before,
        "after":             dist_after,
        "synthetic_generated": num_synthetic,
        "synthetic_saved":   saved_count,
        "chi_square": {
            "statistic": chi2_val,
            "p_value":   p_val,
            "result":    chi2_result,
        },
    }


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_prepared_dataset(output_dir: Path = config.TB_ORGANIZED_ROOT) -> dict:
    logger.info("\nVerifying prepared dataset...")
    verification = {}
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        split_v   = {}
        for cname in ["Normal", "Tuberculosis"]:
            cdir = split_dir / cname
            split_v[cname] = len(list(cdir.glob("*.png"))) if cdir.exists() else 0
        if split == "train":
            aug_dir = split_dir / "Tuberculosis_augmented"
            if aug_dir.exists():
                split_v["Tuberculosis_augmented"] = len(list(aug_dir.glob("*.png")))
        total = sum(v for k, v in split_v.items() if k != "Tuberculosis_augmented")
        logger.info(f"{split}: {split_v}  (real total={total})")
        verification[split] = split_v
    return verification


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: build clean split — no SMOTE here
    stats = prepare_tb_dataset()

    # Verify the raw split before touching it with SMOTE
    logger.info("\n--- Split before SMOTE ---")
    verify_prepared_dataset()

    # Step 2: apply feature-space SMOTE on training split ONLY
    device = "cuda" if (HAS_TORCH and __import__("torch").cuda.is_available()) else "cpu"
    smote_stats = apply_feature_smote_to_train_split(device=device)

    # Final verification
    logger.info("\n--- Final dataset (including augmented) ---")
    verify_prepared_dataset()

    logger.info(f"\nSMOTE stats: {smote_stats}")