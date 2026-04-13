## Label Efficiency, Robustness, and Representation Geometry in Self-Supervised Learning

This repository contains the implementation and experiments for a systematic empirical study of representation learning methods under varying supervision levels.

We evaluate **Supervised Learning, SimCLR, BYOL, and MAE** on CIFAR-10, focusing on:

- Label efficiency across multiple supervision regimes  
- Robustness under label noise  
- Representation geometry analysis  

---

### Key Contributions

- **Label Efficiency Score (LES):**  
  A continuous metric that integrates performance across label fractions using a logarithmic scale.

- **Unified Evaluation Framework:**  
  All methods are evaluated under identical conditions for fair comparison.

- **Noise Robustness Analysis:**  
  Systematic evaluation under controlled label corruption.

- **Representation Geometry Study:**  
  Analysis using effective rank, intra-class variance, inter-class distance, and separation ratio.

---

### Methods Implemented

- **Supervised Learning:** ResNet50 baseline  
- **SimCLR:** Contrastive learning  
- **BYOL:** Non-contrastive self-supervised learning  
- **MAE:** Pretrained Vision Transformer (Masked Autoencoder)

---

### Project Structure

```

ssl_label_efficiency/
│
├── datasets/              # Data loading and preprocessing
├── models/                # Model architectures (ResNet, etc.)
├── training/              # Training loops
├── evaluation/            # Linear probing and feature extraction
├── scripts/               # Main experiment scripts
│   ├── evaluate_simclr.py
│   ├── evaluate_byol.py
│   ├── evaluate_mae.py
│   ├── evaluate_noise.py
│   └── run_representation_geometry.py
│
├── analysis/              # LES computation and plotting
├── results/               # Saved results and plots
├── main.py                # Supervised training + scaling study
└── README.md

````

---

### Installation

```bash
git clone https://github.com/your-username/ssl-label-efficiency.git
cd ssl-label-efficiency

conda create -n ssl_project python=3.10
conda activate ssl_project

pip install -r requirements.txt
````

---

### Experiments

#### 1. Supervised Scaling Study

```bash
python main.py
```

Runs supervised training across label fractions:

```
[1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
```

---

#### 2. Self-Supervised Evaluation

```bash
python -m scripts.evaluate_simclr
python -m scripts.evaluate_byol
python -m scripts.evaluate_mae
```

---

#### 3. Label Efficiency Analysis

```bash
python -m analysis.plot_label_efficiency
```

Outputs:

* Label efficiency curves
* LES and ΔLES values

---

#### 4. Noise Robustness

```bash
python -m scripts.evaluate_noise
```

Evaluates performance under:

```
Noise levels: 0%, 20%, 40%
```

---

#### 5. Representation Geometry

```bash
python -m scripts.run_representation_geometry
```

Computes:

* Effective Rank
* Intra-class Variance
* Inter-class Distance
* Separation Ratio

---

### Key Results

#### Label Efficiency (LES)

| Method     | LES    | ΔLES    |
| ---------- | ------ | ------- |
| Supervised | 0.4138 | —       |
| SimCLR     | 0.4462 | +0.0323 |
| BYOL       | 0.3111 | -0.1028 |
| MAE        | 0.9486 | +0.5347 |

---

### Observations

* **SimCLR** improves label efficiency in low-label regimes
* **BYOL** shows stable but limited performance
* **MAE** achieves high performance due to large-scale pretraining
* **Supervised learning** degrades significantly with fewer labels

---

### Insights

* Label efficiency depends on **representation quality**, not just accuracy
* Contrastive methods provide **robust and transferable features**
* Representation geometry reveals trade-offs between:

  * Diversity (effective rank)
  * Structure (class separation)

---

### Limitations

* Experiments are limited to CIFAR-10
* MAE uses pretrained weights (not trained from scratch)
* Minimal hyperparameter tuning for fairness

---

### Future Work

* Extend to ImageNet-scale datasets
* Train MAE from scratch for fair comparison
* Explore theoretical links between geometry and performance

---

### Acknowledgment

This work was conducted as part of a research-oriented course project focused on understanding self-supervised learning through systematic experimentation.
