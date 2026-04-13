"""
analysis/plot_label_efficiency.py
----------------------------------
Label Efficiency Score (LES) computation and label-efficiency curve plotting.

This script computes and visualises the Label Efficiency Score for each
method — a single scalar that summarises performance across the full range
of label fractions.

Label Efficiency Score (LES)
-----------------------------
The LES is the area under the accuracy-vs-log(fraction) curve, normalised by
the absolute range of the log-fraction axis:

    LES = ∫ acc(log f) d(log f) / |log f_min|

Computed via the trapezoidal rule (`np.trapezoid`). A higher LES means the
method maintains better accuracy across all label fractions — i.e. it is more
label-efficient.

ΔLES (delta LES)
----------------
Gain of each SSL method over the supervised baseline:
    ΔLES(method) = LES(method) - LES(supervised)

Positive ΔLES means the SSL method outperforms supervised learning on average
across all tested label fractions.

Accuracy values used here are the final experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Final experimental results ----
# Accuracies at label fractions [1.0, 0.5, 0.2, 0.1, 0.05, 0.01].
fractions = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

supervised = np.array([0.6833, 0.5817, 0.4181, 0.3981, 0.3920, 0.1705])
simclr     = np.array([0.4531, 0.4520, 0.4513, 0.4507, 0.4474, 0.4266])
byol       = np.array([0.3109, 0.3110, 0.3115, 0.3143, 0.3096, 0.3100])
mae        = np.array([0.9498, 0.9500, 0.9490, 0.9491, 0.9497, 0.9440])

# Convert fractions to log scale for the LES integral.
# Using log makes the x-axis spacing between fractions uniform and reflects
# the intuition that doubling data is equally useful regardless of starting size.
log_f = np.log(fractions)

# Sort by ascending log fraction (smallest fraction first) for trapezoid
# integration, which requires a monotonically ordered x-axis.
idx = np.argsort(log_f)
log_f = log_f[idx]


def compute_les(acc):
    """
    Compute the Label Efficiency Score for a given accuracy array.

    Parameters
    ----------
    acc : numpy.ndarray. Accuracy at each fraction in the original order.

    Returns
    -------
    float. Normalised area under the accuracy-vs-log(fraction) curve.
    """
    # Reorder accuracy values to match the sorted log-fraction axis.
    acc = acc[idx]

    # Trapezoidal area under the accuracy-vs-log(fraction) curve.
    # Normalised by |log(f_min)| so LES ∈ [0, 1].
    return np.trapezoid(acc, log_f) / abs(log_f.min())


# ---- Compute LES for each method ----
les_sup  = compute_les(supervised)
les_sim  = compute_les(simclr)
les_byol = compute_les(byol)
les_mae  = compute_les(mae)

print("Supervised LES:", les_sup)
print("SimCLR LES:",     les_sim)
print("BYOL LES:",       les_byol)
print("MAE LES:",        les_mae)

# ---- ΔLES relative to the supervised baseline ----
print("\nΔLES (SimCLR):", les_sim  - les_sup)
print("ΔLES (BYOL):",   les_byol - les_sup)
print("ΔLES (MAE):",    les_mae  - les_sup)

# ---- Plot: accuracy vs label fraction (log x-axis) ----
plt.figure()
plt.plot(fractions, supervised, marker='o', label='Supervised')
plt.plot(fractions, simclr,     marker='o', label='SimCLR')
plt.plot(fractions, byol,       marker='o', label='BYOL')
plt.plot(fractions, mae,        marker='o', label='MAE')

plt.xscale('log')
plt.xlabel('Label Fraction (log scale)')
plt.ylabel('Accuracy')
plt.title('Label Efficiency Comparison')

plt.legend()
plt.grid(True)
plt.savefig('label_efficiency_comparison.png')
plt.show()
