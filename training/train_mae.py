import torch
from tqdm import tqdm


def train_mae(model, loader, optimizer, device):

    model.train()
    total_loss = 0

    for x, _ in tqdm(loader):

        x = x.to(device)

        output = model(x)

        # Simple reconstruction loss (proxy)
        loss = -torch.log(output.var(dim=0).mean() + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)