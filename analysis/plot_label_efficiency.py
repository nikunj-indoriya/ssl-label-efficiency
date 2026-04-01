import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------

fractions = np.array([1.0, 0.1, 0.05, 0.01])

# Supervised results
supervised = np.array([0.6599, 0.4047, 0.3027, 0.2027])

# SimCLR results (aligning with same fractions)
simclr = np.array([0.5020, 0.4974, 0.4876, 0.4318])
byol = np.array([0.3256, 0.3266, 0.3192, 0.3167])

# -----------------------------
# Plotting
# -----------------------------

plt.figure(figsize=(7, 5))

plt.plot(fractions, supervised, marker='o', linewidth=2, label='Supervised')
plt.plot(fractions, simclr, marker='o', linewidth=2, label='SimCLR')
plt.plot(fractions, byol, marker='o', linewidth=2, label='BYOL')

plt.xscale('log')
plt.xlabel('Label Fraction (log scale)')
plt.ylabel('Accuracy')
plt.title('Label Efficiency Comparison: Supervised vs SimCLR vs BYOL')

plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()

plt.savefig('label_efficiency_comparison.png')
plt.show()

# -----------------------------
# LES Calculation
# -----------------------------

def compute_les(fractions, accuracies):
    log_f = np.log(fractions)
    sorted_idx = np.argsort(log_f)
    log_f = log_f[sorted_idx]
    accuracies = accuracies[sorted_idx]
    return np.trapezoid(accuracies, log_f)

les_supervised = compute_les(fractions, supervised)
les_simclr = compute_les(fractions, simclr)
les_byol = compute_les(fractions, byol)

print(f"Supervised LES: {les_supervised:.4f}")
print(f"SimCLR LES: {les_simclr:.4f}")
print(f"BYOL LES: {les_byol:.4f}")