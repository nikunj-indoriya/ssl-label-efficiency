import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from methods.byol import BYOL
from datasets.simclr_dataset import SimCLRDataset
from training.train_byol import train_byol


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SimCLRDataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    model = BYOL().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_epochs = 40

    for epoch in range(total_epochs):
        loss = train_byol(model, loader, optimizer, device, epoch, total_epochs)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    torch.save(model.online_encoder.state_dict(), "byol_encoder.pth")


if __name__ == "__main__":
    main()