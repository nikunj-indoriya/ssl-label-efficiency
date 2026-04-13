import torch
import torch.optim as optim

from methods.mae import MAE
from datasets.mae_dataset import get_mae_loader
from training.train_mae import train_mae


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_mae_loader()

    model = MAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        loss = train_mae(model, loader, optimizer, device)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    torch.save(model.encoder.state_dict(), "mae_encoder.pth")


if __name__ == "__main__":
    main()