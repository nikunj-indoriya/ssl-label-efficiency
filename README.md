# Label Efficiency, Robustness, and Representation Geometry in Self-Supervised Learning

> A systematic empirical study comparing self-supervised and supervised representation learning on CIFAR-10 across varying label regimes, label noise levels, and feature-space geometry metrics.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Reproducing Experiments](#reproducing-experiments)
- [Figures](#figures)
- [Methods](#methods)
- [Metrics](#metrics)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

Self-supervised learning (SSL) promises to learn useful representations from unlabelled data, reducing dependence on expensive manual annotations. This project answers three concrete empirical questions:

1. **Label Efficiency** — How does performance scale with the number of labelled examples for supervised vs. SSL methods?
2. **Noise Robustness** — How much does accuracy degrade when training labels are randomly corrupted?
3. **Representation Geometry** — What structural properties of the learned feature space explain the observed differences?

We evaluate four methods under identical experimental conditions on CIFAR-10:

| Method | Paradigm | Backbone |
|---|---|---|
| Supervised | End-to-end cross-entropy | ResNet-50 |
| SimCLR | Contrastive (NT-Xent loss) | ResNet-50 |
| BYOL | Non-contrastive (EMA target network) | ResNet-50 |
| MAE | Masked image modelling (ViT encoder) | ViT-Base/16 |

---

## Key Results

### Label Efficiency Score (LES)

The LES is the area under the accuracy-vs-log(label fraction) curve, normalised to [0, 1]. It summarises a method's average performance across *all* label budgets in a single number.

| Method | LES | ΔLES vs Supervised |
|---|---|---|
| Supervised | 0.4138 | — |
| SimCLR | 0.4462 | **+0.0323** |
| BYOL | 0.3111 | −0.1028 |
| MAE | 0.9486 | **+0.5347** |

### Accuracy at Individual Label Fractions

| Label Fraction | Supervised | SimCLR | BYOL | MAE |
|---|---|---|---|---|
| 100% | 0.6833 | 0.4531 | 0.3109 | **0.9498** |
| 50% | 0.5817 | 0.4520 | 0.3110 | **0.9500** |
| 20% | 0.4181 | 0.4513 | 0.3115 | **0.9490** |
| 10% | 0.3981 | 0.4507 | 0.3143 | **0.9491** |
| 5% | 0.3920 | 0.4474 | 0.3096 | **0.9497** |
| 1% | 0.1705 | **0.4266** | 0.3100 | **0.9440** |

> **Key takeaway:** Supervised accuracy collapses at low label fractions (0.17 at 1%), while SimCLR and MAE representations remain largely stable — confirming the label-efficiency advantage of SSL.

### Noise Robustness (10% label fraction)

Robustness score = accuracy at noisy labels / accuracy at clean labels.

| Method | 0% Noise | 20% Noise | 40% Noise | Robustness Score |
|---|---|---|---|---|
| Supervised | 0.3889 | 0.3835 | 0.3192 | 0.8208 |
| SimCLR | 0.4552 | 0.4465 | 0.4224 | 0.9279 |
| BYOL | 0.3125 | 0.3066 | 0.3087 | 0.9878 |
| MAE | **0.9492** | **0.9489** | **0.9490** | **0.9998** |

> **Key takeaway:** SSL methods are substantially more robust to label noise because their representations are learned from the data distribution — not from the (potentially corrupted) labels.

### Representation Geometry

Features are L2-normalised before computing all metrics.

| Method | Effective Rank | Intra-class Var | Inter-class Dist | Separation |
|---|---|---|---|---|
| Supervised | 25.67 | 0.4390 | 0.7953 | 1.8115 |
| SimCLR | 22.94 | 0.5311 | 0.5250 | 0.9886 |
| BYOL | 5.77 | 0.3938 | 0.5459 | 1.3861 |
| MAE | **180.56** | 0.5503 | 0.6571 | 1.1940 |

> **Key takeaway:** MAE (pretrained ViT) learns extremely high-rank, diverse representations. Supervised learning achieves the best class separation ratio despite lower rank — its representations are tuned specifically for the classification task. BYOL exhibits near-representational collapse (rank 5.77).

---

## Project Structure

```
ssl_label_efficiency/
│
├── datasets/
│   ├── cifar.py              # CIFAR-10 loaders for all experimental settings
│   ├── simclr_dataset.py     # Two-view dataset for SimCLR / BYOL training
│   ├── mae_dataset.py        # 224×224 loader for MAE pre-training
│   └── utils.py              # Stratified sub-sampling + label noise injection
│
├── models/
│   └── resnet.py             # ResNet-50 with detachable backbone / classifier
│
├── methods/
│   ├── simclr.py             # SimCLR encoder + projection head
│   ├── byol.py               # BYOL online/target network with EMA update
│   ├── mae.py                # MAE ViT-Base/16 encoder + lightweight MLP decoder
│   ├── losses.py             # NT-Xent contrastive loss (SimCLR)
│   ├── byol_loss.py          # BYOL regression loss (negative cosine similarity)
│   └── augmentations.py      # SimCLR stochastic augmentation pipeline
│
├── training/
│   ├── train_supervised.py   # Supervised train / evaluate loops
│   ├── train_simclr.py       # SimCLR contrastive pre-training loop
│   ├── train_byol.py         # BYOL pre-training loop with cosine EMA schedule
│   └── train_mae.py          # MAE proxy reconstruction pre-training loop
│
├── evaluation/
│   ├── extract_features.py   # Frozen-encoder feature extraction (all model types)
│   └── linear_probe.py       # Linear classifier training + evaluation
│
├── scripts/
│   ├── run_simclr.py                  # Pre-train SimCLR encoder
│   ├── run_byol.py                    # Pre-train BYOL encoder
│   ├── run_mae.py                     # Pre-train MAE encoder (from scratch)
│   ├── evaluate_simclr.py             # Linear probe across label fractions (SimCLR)
│   ├── evaluate_byol.py               # Linear probe across label fractions (BYOL)
│   ├── evaluate_mae.py                # Linear probe across label fractions (MAE)
│   ├── evaluate_noise.py              # Label noise robustness experiment
│   ├── run_representation_analysis.py # Spectral / covariance analysis
│   └── run_representation_geometry.py # Geometry metrics (rank, intra, inter, sep.)
│
├── analysis/
│   ├── representation_analysis.py  # Covariance, eigenspectrum, effective rank
│   ├── representation_geometry.py  # Intra/inter-class distance, separation ratio
│   ├── plot_label_efficiency.py    # LES computation + label efficiency plot
│   └── plot_all_results.py         # All six publication figures
│
├── main.py      # Supervised baseline: trains across all label fractions
└── test.py      # Sanity check: verify stratified subset sampling
```

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA (recommended).

```bash
git clone https://github.com/nikunj-indoriya/ssl-label-efficiency.git
cd ssl-label-efficiency

conda create -n ssl_project python=3.10
conda activate ssl_project

pip install torch torchvision timm tqdm matplotlib numpy
```

CIFAR-10 is downloaded automatically to `./data/` on first run.

---

## Reproducing Experiments

Run the steps below **in order**. Each step produces checkpoint files consumed by the next.

### Step 1 — Supervised Baseline

Trains ResNet-50 from scratch at six label fractions and saves the full-data checkpoint.

```bash
python main.py
```

Output: `supervised_model.pth`

---

### Step 2 — SSL Pre-training

Each script trains a self-supervised encoder and saves only the backbone weights.

```bash
# SimCLR (20 epochs, ResNet-50, batch size 256)
python -m scripts.run_simclr

# BYOL (40 epochs, ResNet-50, cosine EMA schedule)
python -m scripts.run_byol

# MAE from scratch (20 epochs, ViT-Base/16, proxy reconstruction loss)
# Note: evaluate_mae.py uses a pretrained ViT instead — this is optional.
python -m scripts.run_mae
```

Outputs: `simclr_encoder.pth`, `byol_encoder.pth`, `mae_encoder.pth`

---

### Step 3 — Label Efficiency Evaluation

Linear-probe evaluation for each SSL method across all label fractions.

```bash
python -m scripts.evaluate_simclr
python -m scripts.evaluate_byol
python -m scripts.evaluate_mae   # uses pretrained ViT-Base/16 from timm
```

---

### Step 4 — Label Noise Robustness

Evaluates all four methods under 0%, 20%, and 40% symmetric label noise at 10% label fraction.

```bash
python -m scripts.evaluate_noise
```

---

### Step 5 — Representation Geometry

Computes effective rank, intra-class variance, inter-class distance, and separation ratio.

```bash
python -m scripts.run_representation_geometry
```

Spectral / covariance analysis (Supervised + SimCLR only):

```bash
python -m scripts.run_representation_analysis
```

---

### Step 6 — Generate All Figures

```bash
# All six figures (label efficiency, LES, ΔLES, noise curves, robustness, geometry)
python -m analysis.plot_all_results

# Label efficiency curve + LES scores only
python -m analysis.plot_label_efficiency
```

Output files: `fig1_label_efficiency.png`, `fig2_les_scores.png`, `fig3_delta_les.png`,
`fig4_noise_curves.png`, `fig5_robustness.png`, `fig6_geometry.png`

---

### Sanity Check

Verify that the stratified sub-sampling utility produces balanced class distributions:

```bash
python test.py
```

---

## Figures

| Figure | Description |
|---|---|
| `fig1_label_efficiency.png` | Accuracy vs label fraction (log x-axis) for all four methods |
| `fig2_les_scores.png` | Label Efficiency Score bar chart |
| `fig3_delta_les.png` | ΔLES relative to the supervised baseline |
| `fig4_noise_curves.png` | Accuracy vs noise level (0–40%) for all methods |
| `fig5_robustness.png` | Robustness score (acc_noisy / acc_clean) bar chart |
| `fig6_geometry.png` | Grouped bar chart of the four representation geometry metrics |

---

## Methods

### Supervised Baseline
Standard end-to-end training with cross-entropy loss. ResNet-50 is trained from random initialisation at each label fraction. Serves as the reference point for all LES and ΔLES comparisons.

### SimCLR
Contrastive pre-training using the NT-Xent (InfoNCE) loss. Two independently augmented views of each image form a positive pair; all other pairs in the batch are negatives. The projection head (2-layer MLP) is used only during pre-training and discarded at evaluation.

### BYOL
Non-contrastive pre-training with an online network (encoder → projector → predictor) and an EMA-updated target network (encoder → projector). The predictor asymmetry prevents representational collapse without requiring negative pairs. The EMA momentum τ is annealed via a cosine schedule from 0.99 toward 1.0.

### MAE (ViT)
Masked image modelling with a ViT-Base/16 encoder. During pre-training, 75% of patch tokens are randomly removed before the encoder, and a lightweight decoder reconstructs the masked regions. For evaluation, a pretrained ImageNet ViT from `timm` is used as the encoder proxy, as training ViT-Base from scratch on CIFAR-10 is data-limited.

---

## Metrics

**Label Efficiency Score (LES)**
Area under the accuracy-vs-log(label fraction) curve, normalised by |log(f_min)|. Computed via the trapezoidal rule. Captures average performance across all label budgets in a single scalar.

**ΔLES**
LES(method) − LES(supervised). Positive values indicate the SSL method outperforms supervised training averaged across all label fractions.

**Robustness Score**
acc_noisy / acc_clean. Measures how much accuracy is retained when training labels are corrupted. Values close to 1.0 indicate high robustness.

**Effective Rank**
exp(H(p)) where H is the Shannon entropy of the normalised eigenvalue distribution of the feature covariance matrix. Quantifies how many dimensions are meaningfully used by the representation.

**Intra-class Variance**
Mean within-class spread of L2-normalised features. Lower values indicate tighter, more compact class clusters.

**Inter-class Distance**
Mean pairwise L2 distance between class centroids. Higher values indicate better class separation.

**Separation Ratio**
Inter-class distance / intra-class variance. A single composite metric for linear separability; higher is better.

---

## Limitations & Future Work

**Current limitations:**
- Experiments are limited to CIFAR-10 (50K training images, 10 classes).
- MAE evaluation uses a pretrained ImageNet ViT rather than a from-scratch CIFAR-10 MAE, which inflates its scores relative to the other methods.
- SSL pre-training uses shortened schedules (20–40 epochs) compared to standard protocols (200–1000 epochs), which may underestimate SimCLR and BYOL performance.
- No hyperparameter search; all methods use default learning rates for fairness.

**Future directions:**
- Scale experiments to ImageNet-100 or full ImageNet.
- Train MAE from scratch with a pixel-level decoder for a truly fair comparison.
- Investigate the theoretical link between effective rank and linear-probe accuracy.
- Extend the noise robustness study to asymmetric and instance-dependent noise models.
- Explore semi-supervised fine-tuning (not just linear probing) for the SSL encoders.

---

## Acknowledgements

This work was conducted as a research-oriented course project on self-supervised learning. The following open-source libraries and works were instrumental:

- [PyTorch](https://pytorch.org/) — deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) — ViT-Base/16 model and pretrained weights
- [SimCLR](https://arxiv.org/abs/2002.05709) — Chen et al., ICML 2020
- [BYOL](https://arxiv.org/abs/2006.07733) — Grill et al., NeurIPS 2020
- [MAE](https://arxiv.org/abs/2111.06377) — He et al., CVPR 2022
