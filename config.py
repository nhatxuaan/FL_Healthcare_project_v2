"""
Centralized configuration for FL Healthcare baseline experiment.
Based on base_paper_summary.md specifications.
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
TB_DATA_ROOT = DATA_ROOT / "tb" / "TB_Chest_Radiography_Database"
TB_ORGANIZED_ROOT = DATA_ROOT / "tb_organized"  # After preprocessing
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
TB_ORGANIZED_ROOT.mkdir(parents=True, exist_ok=True)

# ==================== DATA CONFIGURATION ====================
# TB Chest X-ray dataset parameters
DATASET_NAME = "TB_Chest_Xray"
NUM_CLASSES = 2  # Binary: Normal (0), TB (1)
CLASS_NAMES = ["Normal", "Tuberculosis"]
IMG_SIZE = 224  # Per paper: resize to 224x224
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==================== TRAINING HYPERPARAMETERS ====================
# Per paper: "Optimal Hyperparameters: Batch Size = 32, Learning Rate = 0.01"
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS_PER_ROUND = 2  # Restored for DP+GPU test closer to paper baseline
NUM_ROUNDS = 10  # Number of federated learning rounds

# ==================== FEDERATED LEARNING CONFIGURATION ====================
# Baseline: static tau (Improvement #2 will make it dynamic)
TAU_STATIC = 0.10  # Divergence threshold per paper

# Non-IID distribution
DIRICHLET_ALPHA = 0.5  # Non-IID heterogeneity parameter

# Number of clients
NUM_CLIENTS_BASELINE = 10  # Paper baseline
NUM_CLIENTS_IMPROVED = 15  # Will test with 15-20 later

# Simulation rounds
NUM_ROUNDS_BASELINE = 10  # Baseline test rounds
NUM_ROUNDS_FULL = 20  # Full run

# ==================== DIFFERENTIAL PRIVACY CONFIGURATION ====================
# Per paper: DP with Gaussian noise, sigma^2 = 0.5, epsilon = 2.5, ~96.5% accuracy
DP_ENABLED = True  # Re-enabled for DP+GPU test
DP_SIGMA_SQUARED = 0.5  # Variance of Gaussian noise
DP_EPSILON = 2.5  # Privacy budget

# Opacus PrivacyEngine parameters
DP_NOISE_MULTIPLIER = (DP_SIGMA_SQUARED ** 0.5)  # Convert sigma^2 to sigma for noise_multiplier
DP_MAX_GRAD_NORM = 1.0  # Gradient clipping norm
DP_TARGET_EPSILON = DP_EPSILON

# ==================== MODEL CONFIGURATION ====================
# Transfer learning models per paper
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"  # ImageNet pre-trained weights
FREEZE_BACKBONE = True  # Freeze feature extraction, train only FC layers

# Model architectures
MODELS_TO_TEST = ["resnet50", "vgg16"]  # Both per baseline requirement

# ==================== LOGGING & MONITORING ====================
LOG_LEVEL = "INFO"
VERBOSE = True

# CSV logging
METRICS_CSV = RESULTS_DIR / "baseline_metrics.csv"
ROUNDS_CSV = RESULTS_DIR / "rounds.csv"  # Per-round metrics

# ==================== DEVICE & PRECISION ====================
DEVICE = "cuda"  # Use GPU-enabled PyTorch build in .venv312
DTYPE = "float32"  # PyTorch dtype

# ==================== EXPERIMENT METADATA ====================
EXPERIMENT_NAME = "baseline_fedavg_fedsgd_static_tau"
DESCRIPTION = "Reproduce baseline paper: FedAvg/FedSGD with static tau=0.10, TB dataset, 10 clients, DP enabled"
PAPER_REFERENCE = "A privacy-enhanced framework for collaborative Big Data analysis in healthcare using adaptive federated learning aggregation (Haripriya et al., 2025)"

# ==================== RANDOM SEED ====================
RANDOM_SEED = 42
