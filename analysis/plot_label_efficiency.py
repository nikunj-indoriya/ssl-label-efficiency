import numpy as np
import matplotlib.pyplot as plt

fractions = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

supervised = np.array([0.6360, 0.6027, 0.4655, 0.4050, 0.3264, 0.2264])
simclr = np.array([0.4502, 0.4514, 0.4495, 0.4506, 0.4461, 0.4409])
byol = np.array([0.3120, 0.3134, 0.3095, 0.3134, 0.3106, 0.2922])
mae = np.array([0.9505, 0.9500, 0.9486, 0.9487, 0.9487, 0.9450])

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