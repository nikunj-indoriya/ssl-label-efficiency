import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from methods.simclr import SimCLR
from datasets.simclr_dataset import SimCLRDataset
from training.train_simclr import train_simclr

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SimCLRDataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    model = SimCLR().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        loss = train_simclr(model, loader, optimizer, device)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")

if __name__ == "__main__":
    main()