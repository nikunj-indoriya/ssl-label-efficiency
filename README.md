# SSL Label Efficiency

This repository contains a research project studying label efficiency in self-supervised learning (SSL) for vision.

We evaluate how different SSL paradigms perform under extreme low-label regimes (1%, 5%, 10%) and analyze the geometry of learned representations.

## Methods
- SimCLR (contrastive)
- BYOL (non-contrastive)
- MAE (masked autoencoding)

## Architectures
- ResNet-50
- Vision Transformer (ViT)

## Run everything in order
1. Supervised
python main.py
2. SimCLR 
python -m scripts.evaluate_simclr
3. BYOL
python -m scripts.evaluate_byol
4. MAE 
python -m scripts.evaluate_mae
5. Plot
python -m analysis.plot_label_efficiency