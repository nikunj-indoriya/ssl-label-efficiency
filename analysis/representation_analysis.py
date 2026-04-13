import torch
import numpy as np

def compute_covariance(features):
    features = features - features.mean(dim=0)
    cov = torch.matmul(features.T, features) / (features.shape[0] - 1)
    return cov

def compute_spectrum(cov):
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.sort(eigvals, descending=True).values
    return eigvals.cpu().numpy()

def effective_rank(eigvals):
    eigvals = eigvals / np.sum(eigvals)
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-12))
    return np.exp(entropy)

import matplotlib.pyplot as plt

def plot_spectrum(eigvals, title):
    plt.figure()
    plt.plot(eigvals)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue (log)')
    plt.grid(True)
    plt.show()
    plt.savefig(f"{title}.png")