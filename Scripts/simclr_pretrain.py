import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Common.data import get_simclr_dataloader
from Common.model import resnet18
#from Common.utils import set_seed, save_config, save_checkpoint, CSVLogger

class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        # Encoder: ResNet-18 without classifier
        self.encoder = resnet18(num_classes=10)
        self.encoder.fc = nn.Identity()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z

def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)

    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill(mask, float("-inf"))

    pos = torch.cat([
        torch.diag(sim, B),
        torch.diag(sim, -B)
    ], dim=0)

    loss = -pos + torch.logsumexp(sim, dim=1)
    return loss.mean()

def main():
    os.makedirs("results/figures", exist_ok = True)
    os.makedirs("results/tables", exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    epochs = 2
    lr = 3e-4
    temperature = 0.5
    loss_history = []

    loader = get_simclr_dataloader(batch_size=4)
    model = SimCLRModel(projection_dim=128)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        weight_decay= 1e-4
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for i, ((x1, x2), _) in enumerate(loader):
            x1 = x1.to(device)
            x2 = x2.to(device)

            z1 = model(x1)
            z2 = model(x2)
            
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"  Batch {i}/{len(loader)} - Loss: {loss.item():.3f}")

            epoch_loss += loss.item()
        
        epoch_loss /= len(loader)
        print(f"Epoch [{epoch}/{epochs}] - SSL Loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

    csv_path = "results/tables/simclr_ssl_mterics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "ssl_loss"])
        for i, l in enumerate(loss_history, start=1):
            writer.writerow([i, l])
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("SSL Loss")
    plt.title("SimCLR Pretraining loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/figures/ssl_loss_curve.png")
    plt.close()

if __name__ == "__main__":
    main()