import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATA (FINAL RUNS)
# =========================

fractions = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

supervised = np.array([0.6833, 0.5817, 0.4181, 0.3981, 0.3920, 0.1705])
simclr     = np.array([0.4531, 0.4520, 0.4513, 0.4507, 0.4474, 0.4266])
byol       = np.array([0.3109, 0.3110, 0.3115, 0.3143, 0.3096, 0.3100])
mae        = np.array([0.9498, 0.9500, 0.9490, 0.9491, 0.9497, 0.9440])

# LES
les_scores = {
    "Supervised": 0.4138,
    "SimCLR": 0.4462,
    "BYOL": 0.3111,
    "MAE": 0.9486
}

delta_les = {
    "SimCLR": 0.0323,
    "BYOL": -0.1028,
    "MAE": 0.5347
}

# Noise robustness
noise_levels = np.array([0, 20, 40])

noise_supervised = np.array([0.3889, 0.3835, 0.3192])
noise_simclr     = np.array([0.4552, 0.4465, 0.4224])
noise_byol       = np.array([0.3125, 0.3066, 0.3087])
noise_mae        = np.array([0.9492, 0.9489, 0.9490])

robustness_scores = {
    "Supervised": 0.8208,
    "SimCLR": 0.9279,
    "BYOL": 0.9878,
    "MAE": 0.9998
}

# Geometry
geometry = {
    "Supervised": [25.67, 0.4390, 0.7953, 1.8115],
    "SimCLR":     [22.94, 0.5311, 0.5250, 0.9886],
    "BYOL":       [5.77,  0.3938, 0.5459, 1.3861],
    "MAE":        [180.56,0.5503, 0.6571, 1.1940],
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
plt.figure()
plt.plot(fractions, supervised, marker='o', label="Supervised")
plt.plot(fractions, simclr, marker='o', label="SimCLR")
plt.plot(fractions, byol, marker='o', label="BYOL")
plt.plot(fractions, mae, marker='o', label="MAE")

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
plt.figure()
names = list(les_scores.keys())
values = list(les_scores.values())

plt.bar(names, values)
plt.ylabel("Normalized LES")
plt.title("Label Efficiency Score (LES)")
plt.savefig("fig2_les_scores.png", dpi=300)
plt.close()


# =========================
# 3. DELTA LES
# =========================
plt.figure()
names = list(delta_les.keys())
values = list(delta_les.values())

plt.bar(names, values)
plt.axhline(0, linestyle='--')
plt.ylabel("ΔLES over Supervised")
plt.title("Relative Label Efficiency Gain")
plt.savefig("fig3_delta_les.png", dpi=300)
plt.close()


# =========================
# 4. NOISE ROBUSTNESS CURVES
# =========================
plt.figure()
plt.plot(noise_levels, noise_supervised, marker='o', label="Supervised")
plt.plot(noise_levels, noise_simclr, marker='o', label="SimCLR")
plt.plot(noise_levels, noise_byol, marker='o', label="BYOL")
plt.plot(noise_levels, noise_mae, marker='o', label="MAE")

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
plt.figure()
names = list(robustness_scores.keys())
values = list(robustness_scores.values())

plt.bar(names, values)
plt.ylabel("Robustness Score")
plt.title("Noise Robustness Comparison")
plt.savefig("fig5_robustness.png", dpi=300)
plt.close()


# =========================
# 6. REPRESENTATION GEOMETRY
# =========================
metrics = ["Rank", "Intra-Var", "Inter-Dist", "Separation"]
methods = list(geometry.keys())

data = np.array(list(geometry.values()))

x = np.arange(len(metrics))
width = 0.2

plt.figure()

for i, method in enumerate(methods):
    plt.bar(x + i*width, data[i], width, label=method)

plt.xticks(x + width, metrics)
plt.title("Representation Geometry Comparison")
plt.legend()
plt.savefig("fig6_geometry.png", dpi=300)
plt.close()

print("All plots saved successfully.")