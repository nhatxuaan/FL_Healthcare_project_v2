# 🏥 Privacy-Preserving Federated Learning for Healthcare

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FFD21E)

## 📖 Overview
This project implements a scalable, distributed Deep Learning framework for medical image classification using **Federated Learning (FL)**. 
The architecture is heavily inspired by the baseline paper: *"A privacy-enhanced framework for collaborative Big Data analysis in healthcare using adaptive federated learning aggregation."*

By utilizing the **Flower (`flwr`)** framework, this system allows multiple autonomous nodes (representing hospitals) to collaboratively train a global Convolutional Neural Network (CNN) without ever sharing raw, sensitive patient data with the central server.

## ✨ Novel Contributions (Project Improvements)
While based on the baseline paper, our team proposes three major algorithmic and systemic improvements:
1. **Borderline-SMOTE Integration:** Upgrading from standard SMOTE to Borderline-SMOTE at the local client level. This focuses oversampling on "dangerous" borderline cases (critical in medical imaging) while avoiding noise generation, ensuring strict data privacy without cross-client leakage.
2. **Dynamic Aggregation Threshold ($\tau$):** Replacing the static divergence threshold ($\tau=0.10$) with a dynamic $\tau$. The threshold dynamically adjusts based on the convergence rate ($\Delta$loss/round), optimizing the switch between FedAvg and FedSGD algorithms for better global accuracy.
3. **Enhanced Scalability:** Expanding the federated network simulation from 10 to 15-20 autonomous clients. We address GPU memory constraints by implementing partial participation configurations.

## 🚀 Key Features
* **Federated Architecture:** Robust Client-Server communication using the Flower framework.
* **Deep Learning for Medical Imaging:** Utilizes Transfer Learning with **ResNet** and **VGG16** for high-accuracy image classification.
* **Strict Data Privacy:** Raw X-ray and MRI images remain strictly localized at the hospital level. Only mathematical model weights are transmitted.
* **Cloud Simulation Ready:** Fully compatible with Google Colab's GPU environment for simulating large-scale multi-client networks.

## 🗄️ Datasets

Current workspace data is configured for:
1. [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/data)

Planned future extension datasets:
1. [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
2. [Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)

## 📂 Project Structure
```text
FL_Healthcare_Project/
│
├── config.py              # Central experiment configuration
├── simulation.py          # Full Flower simulation entrypoint (baseline run)
├── smoke_test.py          # Quick local validation script (2-round default)
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── src/
│   ├── client.py           # Client-side training logic & parameter serialization
│   ├── aggregators.py      # FedAvg, FedSGD aggregation & divergence computation
│   ├── models.py           # ResNet50 model with backbone freeze
│   ├── strategy.py         # Adaptive strategy (FedAvg/FedSGD switching)
│   ├── data.py             # Dataset & DataLoader definitions
│   ├── partition.py        # Non-IID client partition logic
│   ├── prepare_data.py     # Raw TB -> organized train/val/test preprocessing
│   ├── evaluation.py       # Training/evaluation utilities
│   └── dp.py               # Differential Privacy integration
│
├── data/
│   ├── tb/                 # RAW dataset root
│   │   └── TB_Chest_Radiography_Database/
│   │       ├── Normal/
│   │       └── Tuberculosis/
│   └── tb_organized/       # Preprocessed & organized data (auto-created)
│       ├── train/          # Training data split (70% of total)
│       │   ├── Normal/     # Normal X-ray images
│       │   └── Tuberculosis/ # TB X-ray images
│       ├── val/            # Validation data split (15% of total)
│       │   ├── Normal/
│       │   └── Tuberculosis/
│       └── test/           # Test data split (15% of total)
│           ├── Normal/
│           └── Tuberculosis/
│
├── results/                # Output metrics & logs (auto-created)
│   └── rounds.csv          # Per-round results: divergence, loss, accuracy, time
│
├── .gitignore              # Ignored files (venv, raw data, outputs)
└── docs/
    └── base_paper_summary.md  # Baseline paper reference & parameters
```

### Data Directory Details
- **`data/tb/`** - Raw dataset root. Expected source path for preprocessing is `data/tb/TB_Chest_Radiography_Database/` with class folders `Normal/` and `Tuberculosis/`
- **`data/tb_organized/`** - Created **after preprocessing** with organized folder structure:
  - `train/` - 70% of data split by class (Normal, Tuberculosis)
  - `val/` - 15% of data split by class
  - `test/` - 15% of data split by class

Note: These folders are auto-generated and should be in `.gitignore` since they contain preprocessed data.

## ⚙️ Installation & Setup (Complete Guide for Others)

### Prerequisites
- **Python 3.10+** (recommended: 3.12 for Flower simulation with Ray)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Kaggle API** (optional, only if you want to re-download raw dataset)
- **~500MB disk space** for datasets (after preprocessing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/viethoangp/FL_Healthcare_Project.git
cd FL_Healthcare_Project
```

### Step 2: Create & Activate Virtual Environment
On **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

On **Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected packages:**
- `flower==1.27.0` - Federated Learning framework
- `torch==2.11.0` - Deep Learning backend (CPU version)
- `torchvision==0.16.0` - Vision utilities & pretrained models
- `scikit-learn>=1.6.0` - ML utilities & metrics
- `opacus>=1.5.0` - Differential Privacy library
- `pandas>=2.0.0` - Data manipulation

### Step 4: Download Dataset from Kaggle

Download the dataset zip file from Kaggle and extract it into the `data/` folder.
After extraction, the folder should be:
    data/tb/TB_Chest_Radiography_Database/...

### Step 5: Prepare Dataset

Run the preprocessing script after extracting the raw zip:

python src/prepare_data.py

This will:
1. Read raw TB images from `data/tb/TB_Chest_Radiography_Database/`
2. Resize images to 224x224 and convert to RGB
3. Create organized train/val/test splits in `data/tb_organized/`
4. Keep class folders in each split (Normal & Tuberculosis)
5. Split ratio: 70% train / 15% validation / 15% test

**After completion, your `data/` folder structure will be:**
```
data/
├── tb/
│   └── TB_Chest_Radiography_Database/
│       ├── Normal/
│       └── Tuberculosis/
└── tb_organized/          # PROCESSED: organized splits by class
    ├── train/
    │   ├── Normal/
    │   └── Tuberculosis/
    ├── val/
    │   ├── Normal/
    │   └── Tuberculosis/
    └── test/
        ├── Normal/
        └── Tuberculosis/
```

---

## 🏃‍♂️ How to Run the Federated Learning Pipeline

### Option 1: Run Complete Baseline (Default - Recommended)
Execute the full Flower simulation with 10 rounds:

```bash
python simulation.py --num-rounds 10 --model resnet50
```

**What this does:**
- Initializes 10 federated clients with local TB Chest X-ray data
- Runs 10 communication rounds with FedAvg/FedSGD adaptive switching
- Logs divergence metrics, loss, accuracy, and elapsed time per round
- Exports results to `results/rounds.csv` with columns:
  - `round`, `divergence`, `num_clients`, `global_loss`, `train_accuracy`, `test_accuracy`, `elapsed_time`, `epsilon` (if DP enabled)
- Final test accuracy printed at pipeline completion


### Option 2: Quick Sanity Test (2 Rounds)
To quickly verify setup without full training:

```bash
python smoke_test.py
```

**Runtime:** ~240 seconds (~4 minutes)

### Option 3: Run with Differential Privacy (DP) Enabled
For privacy-preserving federated learning with Gaussian noise injection:

1. Edit `config.py`:
   ```python
   DP_ENABLED = True          # Enable DP
   DP_EPSILON = 2.5           # Privacy budget per round
   DP_SIGMA_SQUARED = 0.5     # Gaussian noise variance
   ```

2. Run pipeline:
   ```bash
   python simulation.py --num-rounds 10 --model resnet50
   ```

**Note:** DP reduces accuracy slightly; privacy-utility tradeoff is configurable via `/epsilon` and `/sigma`.

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Federated Learning
NUM_CLIENTS_BASELINE = 10   # Number of hospital clients
NUM_ROUNDS_BASELINE = 10    # Communication rounds
BATCH_SIZE = 32             # Batch size per client
LEARNING_RATE = 0.01        # SGD learning rate
NUM_EPOCHS_PER_ROUND = 1    # Local epochs per round

# Aggregation Strategy
TAU_STATIC = 0.10           # τ: Switch FedAvg→FedSGD when divergence > τ

# Model
FREEZE_BACKBONE = True      # Freeze ResNet50 conv1, bn1, layer1-4
MODELS_TO_TEST = ["resnet50", "vgg16"]

# Dataset
DIRICHLET_ALPHA = 0.5       # Non-IID distribution (lower α = more heterogeneous)
RANDOM_SEED = 42            # Reproducibility

# Privacy (Differential Privacy)
DP_ENABLED = True           # Enable Gaussian noise injection
DP_EPSILON = 2.5            # Privacy budget
DP_SIGMA_SQUARED = 0.5      # Noise variance

# Logging
LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR
ROUNDS_CSV = RESULTS_DIR / "rounds.csv"
```

---

## Citation & References

Haripriya, R., Khare, N., Pandey, M., and Biswas, S. A privacy-enhanced framework for collaborative big data analysis in healthcare using adaptive federated learning aggregation. Journal of Big Data, 12(1), 113, 2025. DOI: https://doi.org/10.1186/s40537-025-01169-8

**Frameworks & Libraries:**
- **Flower (Federated Learning):** https://flower.dev
- **PyTorch:** Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS 2019

---
