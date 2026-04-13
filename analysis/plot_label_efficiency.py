import numpy as np
import matplotlib.pyplot as plt

fractions = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

supervised = np.array([0.6833, 0.5817, 0.4181, 0.3981, 0.3920, 0.1705])
simclr = np.array([0.4531, 0.4520, 0.4513, 0.4507, 0.4474, 0.4266])
byol = np.array([0.3109, 0.3110, 0.3115, 0.3143, 0.3096, 0.3100])
mae = np.array([0.9498, 0.9500, 0.9490, 0.9491, 0.9497, 0.9440])

log_f = np.log(fractions)

# sort
idx = np.argsort(log_f)
log_f = log_f[idx]

def compute_les(acc):
    acc = acc[idx]
    return np.trapezoid(acc, log_f) / abs(log_f.min())

les_sup = compute_les(supervised)
les_sim = compute_les(simclr)
les_byol = compute_les(byol)
les_mae = compute_les(mae)

print("Supervised LES:", les_sup)
print("SimCLR LES:", les_sim)
print("BYOL LES:", les_byol)
print("MAE LES:", les_mae)

print("\nΔLES (SimCLR):", les_sim - les_sup)
print("ΔLES (BYOL):", les_byol - les_sup)
print("ΔLES (MAE):", les_mae - les_sup)

# Plot
plt.figure()
plt.plot(fractions, supervised, marker='o', label='Supervised')
plt.plot(fractions, simclr, marker='o', label='SimCLR')
plt.plot(fractions, byol, marker='o', label='BYOL')
plt.plot(fractions, mae, marker='o', label='MAE')

plt.xscale('log')
plt.xlabel('Label Fraction (log scale)')
plt.ylabel('Accuracy')
plt.title('Label Efficiency Comparison')

plt.legend()
plt.grid(True)
plt.savefig('label_efficiency_comparison.png')
plt.show()