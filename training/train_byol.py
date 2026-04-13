import torch
from tqdm import tqdm
from methods.byol_loss import byol_loss
import math


def train_byol(model, loader, optimizer, device, epoch, total_epochs):

    model.train()
    total_loss = 0

    for step, (x1, x2) in enumerate(tqdm(loader)):

        x1, x2 = x1.to(device), x2.to(device)

        # Forward pass
        p1, p2, z1, z2 = model(x1, x2)

        loss = byol_loss(p1, z2) + byol_loss(p2, z1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Cosine EMA schedule for tau ----
        global_step = epoch * len(loader) + step
        max_steps = total_epochs * len(loader)

        tau = 1 - (1 - 0.99) * (math.cos(math.pi * global_step / max_steps) + 1) / 2

        model.update_target(tau)

        total_loss += loss.item()

    return total_loss / len(loader)