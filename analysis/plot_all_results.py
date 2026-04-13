"""
analysis/plot_all_results.py
-----------------------------
Comprehensive results visualisation for the SSL label-efficiency study.

Generates six publication-quality figures summarising all experimental results:

  Fig 1 — Label Efficiency Curves
          Accuracy vs label fraction (log x-axis) for all four methods.
          Reveals how rapidly each method degrades as labels are reduced.

  Fig 2 — LES Bar Plot
          Label Efficiency Score (area under the label-efficiency curve) for
          each method. A single scalar summarising label efficiency.

  Fig 3 — ΔLES Bar Plot
          Gain of each SSL method over the supervised baseline.
          Positive ΔLES = SSL method is more label-efficient than supervised.

  Fig 4 — Noise Robustness Curves
          Accuracy vs label noise level (0%, 20%, 40%) for all methods.
          Shows how robust each method is to label corruption.

  Fig 5 — Robustness Score Bar Plot
          acc_noisy / acc_clean ratio for each method — a scalar robustness
          summary (higher = less degraded by noise).

  Fig 6 — Representation Geometry Bar Chart
          Four geometry metrics (Effective Rank, Intra-class Variance,
          Inter-class Distance, Separation Ratio) grouped by method,
          enabling direct comparison of feature space quality.

All plots are saved as high-resolution PNGs (300 dpi) and the display is
closed after saving to avoid blocking execution in headless environments.
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA (FINAL RUNS)
# =========================

# Label fractions swept in the main experiment.
fractions = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

# Linear-probe / fine-tune accuracy at each label fraction for all methods.
supervised = np.array([0.6833, 0.5817, 0.4181, 0.3981, 0.3920, 0.1705])
simclr     = np.array([0.4531, 0.4520, 0.4513, 0.4507, 0.4474, 0.4266])
byol       = np.array([0.3109, 0.3110, 0.3115, 0.3143, 0.3096, 0.3100])
mae        = np.array([0.9498, 0.9500, 0.9490, 0.9491, 0.9497, 0.9440])

# Precomputed Label Efficiency Scores (area under accuracy-vs-log-fraction curve,
# normalised by |log(f_min)|).
les_scores = {
    "Supervised": 0.4138,
    "SimCLR":     0.4462,
    "BYOL":       0.3111,
    "MAE":        0.9486
}

# ΔLES: improvement of each SSL method over the supervised LES.
delta_les = {
    "SimCLR": 0.0323,
    "BYOL":  -0.1028,
    "MAE":    0.5347
}

# Accuracy at noise levels 0%, 20%, 40% (label_fraction=0.1).
noise_levels = np.array([0, 20, 40])

noise_supervised = np.array([0.3889, 0.3835, 0.3192])
noise_simclr     = np.array([0.4552, 0.4465, 0.4224])
noise_byol       = np.array([0.3125, 0.3066, 0.3087])
noise_mae        = np.array([0.9492, 0.9489, 0.9490])

# Robustness scores: acc_noisy / acc_clean for each method.
robustness_scores = {
    "Supervised": 0.8208,
    "SimCLR":     0.9279,
    "BYOL":       0.9878,
    "MAE":        0.9998
}

# Representation geometry metrics for each method:
# [Effective Rank, Intra-class Variance, Inter-class Distance, Separation Ratio]
geometry = {
    "Supervised": [25.67,  0.4390, 0.7953, 1.8115],
    "SimCLR":     [22.94,  0.5311, 0.5250, 0.9886],
    "BYOL":       [ 5.77,  0.3938, 0.5459, 1.3861],
    "MAE":        [180.56, 0.5503, 0.6571, 1.1940],
}

# =========================
# STYLE
# =========================
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12


# =========================
# 1. LABEL EFFICIENCY CURVE
# =========================
# Shows how accuracy changes as the label budget is reduced.
# A flat curve (SSL) vs steeply dropping curve (supervised) indicates
# better label efficiency.
plt.figure()
plt.plot(fractions, supervised, marker='o', label="Supervised")
plt.plot(fractions, simclr,     marker='o', label="SimCLR")
plt.plot(fractions, byol,       marker='o', label="BYOL")
plt.plot(fractions, mae,        marker='o', label="MAE")

plt.xscale("log")
plt.xlabel("Label Fraction (log scale)")
plt.ylabel("Accuracy")
plt.title("Label Efficiency Comparison")
plt.legend()
plt.grid(True)
plt.savefig("fig1_label_efficiency.png", dpi=300)
plt.close()


# =========================
# 2. LES BAR PLOT
# =========================
# LES (Label Efficiency Score) summarises the area under the label-efficiency
# curve. Higher bars → better average performance across all label fractions.
plt.figure()
names  = list(les_scores.keys())
values = list(les_scores.values())

plt.bar(names, values)
plt.ylabel("Normalized LES")
plt.title("Label Efficiency Score (LES)")
plt.savefig("fig2_les_scores.png", dpi=300)
plt.close()


# =========================
# 3. DELTA LES
# =========================
# ΔLES measures the improvement of each SSL method over the supervised LES.
# Bars above zero → SSL method is more label-efficient than supervised training.
plt.figure()
names  = list(delta_les.keys())
values = list(delta_les.values())

plt.bar(names, values)
plt.axhline(0, linestyle='--')   # reference line at Δ=0 (no improvement)
plt.ylabel("ΔLES over Supervised")
plt.title("Relative Label Efficiency Gain")
plt.savefig("fig3_delta_les.png", dpi=300)
plt.close()


# =========================
# 4. NOISE ROBUSTNESS CURVES
# =========================
# Shows how accuracy degrades as training label noise increases from 0→40%.
# A flat line indicates high robustness; a steep drop indicates fragility.
plt.figure()
plt.plot(noise_levels, noise_supervised, marker='o', label="Supervised")
plt.plot(noise_levels, noise_simclr,     marker='o', label="SimCLR")
plt.plot(noise_levels, noise_byol,       marker='o', label="BYOL")
plt.plot(noise_levels, noise_mae,        marker='o', label="MAE")

plt.xlabel("Noise Level (%)")
plt.ylabel("Accuracy")
plt.title("Robustness to Label Noise")
plt.legend()
plt.grid(True)
plt.savefig("fig4_noise_curves.png", dpi=300)
plt.close()


# =========================
# 5. ROBUSTNESS SCORES
# =========================
# acc_noisy / acc_clean ratio. A score close to 1.0 means the method is
# almost unaffected by 40% label noise — desirable for real-world use cases.
plt.figure()
names  = list(robustness_scores.keys())
values = list(robustness_scores.values())

plt.bar(names, values)
plt.ylabel("Robustness Score")
plt.title("Noise Robustness Comparison")
plt.savefig("fig5_robustness.png", dpi=300)
plt.close()


# =========================
# 6. REPRESENTATION GEOMETRY
# =========================
# Grouped bar chart: four geometry metrics (x-axis groups) with one bar per
# method. Allows direct comparison of feature space quality across methods.
metrics = ["Rank", "Intra-Var", "Inter-Dist", "Separation"]
methods = list(geometry.keys())

data = np.array(list(geometry.values()))   # shape: (num_methods, num_metrics)

x     = np.arange(len(metrics))
width = 0.2   # width of each individual bar

plt.figure()

for i, method in enumerate(methods):
    # Offset each method's bars horizontally so they appear side by side.
    plt.bar(x + i*width, data[i], width, label=method)

# Centre the x-tick labels under each group of four bars.
plt.xticks(x + width, metrics)
plt.title("Representation Geometry Comparison")
plt.legend()
plt.savefig("fig6_geometry.png", dpi=300)
plt.close()


print("All plots saved successfully.")
