import matplotlib.pyplot as plt

fractions = [1.0, 0.1, 0.05, 0.01]
accuracies = [0.6599, 0.4047, 0.3027, 0.2027]

plt.figure()
plt.plot(fractions, accuracies, marker='o')

plt.xscale('log')
plt.xlabel('Label Fraction (log scale)')
plt.ylabel('Accuracy')
plt.title('Label Efficiency Curve (Supervised Baseline)')

plt.grid(True)
plt.savefig('label_efficiency.png')
plt.show()

import numpy as np

fractions = np.array([1.0, 0.1, 0.05, 0.01])
accuracies = np.array([0.6599, 0.4047, 0.3027, 0.2027])

log_f = np.log(fractions)

# Sort (important for integration)
sorted_idx = np.argsort(log_f)
log_f = log_f[sorted_idx]
accuracies = accuracies[sorted_idx]

les = np.trapezoid(accuracies, log_f)

print("Label Efficiency Score (LES):", les)