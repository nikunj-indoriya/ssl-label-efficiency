import torch
from tqdm import tqdm
from methods.losses import nt_xent_loss

def train_simclr(model, loader, optimizer, device):

    model.train()
    total_loss = 0

    for x1, x2 in tqdm(loader):

        x1, x2 = x1.to(device), x2.to(device)

        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)