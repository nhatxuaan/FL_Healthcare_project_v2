# Base Paper Summary & Project Context for GitHub Copilot

[cite_start]**Title:** A privacy-enhanced framework for collaborative Big Data analysis in healthcare using adaptive federated learning aggregation 
**Authors:** Haripriya et al. (2025) [cite_start][cite: 1718, 1740]
**Role of Copilot:** Act as a Senior AI Researcher implementing this paper using PyTorch and the `flwr` (Flower) framework.

## 1. Datasets & Preprocessing
[cite_start]The project utilizes three distinct medical datasets distributed in a **Non-IID** manner across clients to simulate real-world healthcare heterogeneity[cite: 2214, 2284].
* [cite_start]**TB Chest X-ray:** 7,000 images (2 classes: Normal, TB)[cite: 1980].
* [cite_start]**Brain Tumor MRI:** 3,624 images (4 classes: No Tumor, Meningioma, Glioma, Pituitary)[cite: 1980].
* [cite_start]**Diabetic Retinopathy (APTOS):** 3,662 images (5 classes: No DR, Mild, Moderate, Severe, Proliferate DR)[cite: 1980].
* [cite_start]**Image Preprocessing:** All images MUST be resized to exactly $224 \times 224$ pixels to be compatible with ResNet and VGG16[cite: 1980, 2014].
* [cite_start]**Data Split:** 70% Training, 15% Validation, 15% Testing[cite: 2082].

## 2. Models & Transfer Learning
* [cite_start]**Architectures:** ResNet (specifically ResNet50) and VGG16[cite: 1724, 1826].
* [cite_start]**Initialization:** Transfer learning using ImageNet pre-trained weights (`IMAGENET1K_V1`)[cite: 1725, 1767]. 
* [cite_start]**Fine-tuning Strategy:** Preserve foundational feature-extraction layers and modify the fully connected (FC) layers to match the specific number of medical classes (2, 4, or 5)[cite: 2085, 2086].
* [cite_start]**Loss Function:** Categorical Cross-Entropy Loss[cite: 2101].
* [cite_start]**Optimal Hyperparameters:** Batch Size = 32 (for faster convergence), Learning Rate = 0.01 (for stable convergence)[cite: 2604, 2607].

## 3. Core Algorithm: Adaptive Aggregation (Base Paper Logic)
[cite_start]The paper's main novelty is dynamically switching between `FedAvg` and `FedSGD` based on the divergence of client model updates[cite: 1726, 1727, 2213].

* [cite_start]**Divergence Metric ($\delta_t$):** Computed as the average Euclidean distance between local models and the global model[cite: 2237]:
    [cite_start]`delta_t = (1 / K) * sum(|| w_t^k - w_{t-1} ||_2)` [cite: 2238]
* [cite_start]**Static Threshold ($\tau$):** The base paper uses a mathematically optimized static threshold of **$\tau = 0.10$**[cite: 2281].
* **Aggregation Logic:**
    * [cite_start]If `delta_t > 0.10`: Use **FedSGD** (Gradient-based aggregation for high divergence)[cite: 2240].
    * [cite_start]If `delta_t <= 0.10`: Use **FedAvg** (Parameter averaging for communication efficiency)[cite: 2241].

## 4. Privacy Preservation
* [cite_start]**Differential Privacy (DP):** Gaussian noise $\mathcal{N}(0, \sigma^2)$ is added to the gradients during backpropagation[cite: 2184, 2188].
* **Optimal Noise Scale:** $\sigma^2 = 0.5$, which yields a privacy budget of $\epsilon = 2.5$. This provides the optimal balance between accuracy (approx 96.5%) and privacy[cite: 2208, 2822, 2823].

---

## 5. OUR TEAM'S CUSTOM IMPROVEMENTS (CRITICAL OVERRIDES)
When generating code, Copilot MUST prioritize these 3 custom improvements over the base paper's methodology:

1.  **Improvement #1 (Data Imbalance):** The base paper mentions SMOTE[cite: 2034]. We will specifically use **`BorderlineSMOTE(kind='borderline-1')`** from `imblearn` applied LOCALLY at each client before training to handle borderline medical cases safely without data leakage[cite: 1686, 1687, 1688].
2.  **Improvement #2 (Dynamic Threshold $\tau$):** Instead of the static $\tau = 0.10$ used by the authors[cite: 1660, 2281], we will implement a **Dynamic Threshold** based on the convergence rate ($\Delta loss$ per round). If `abs(loss_prev - loss_curr)` is small, we decrease $\tau$ (favoring FedAvg); otherwise, we increase $\tau$[cite: 1691, 1692, 1693, 1694].
3.  **Improvement #3 (Scalability):** The base paper evaluates 10 clients[cite: 2214]. We will scale this to **15 and 20 clients** using `flwr.simulation.start_simulation()` to test robustness[cite: 1697, 1703, 2991].