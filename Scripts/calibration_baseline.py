import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Common.model import get_model
from Common.data import get_dataloaders
from Common.calibration import TemperatureScaler

def fit_temperature(model, dataloader, device):
    # Wrap the trained model with a temperature scaling layer
    scaler = TemperatureScaler(model).to(device)

    # Cross-entropy loss corresponds to NLL for classification
    criterion = nn.CrossEntropyLoss()

    # LBFGS is standard for temperature scaling (1 partameter, smooth loss)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = 0.0
        total = 0
         
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = scaler(images)
            batch_loss = criterion(logits, targets)

            loss += batch_loss
            total += 1

        loss = loss/total
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return scaler.temperature.detach().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load model ----
    model = get_model(num_classes=10)
    ckpt = torch.load(
        "runs/adamw_20ep/best.ckpt",   # adjust if your strong-aug run has a different name
        map_location=device
    )


    # Your checkpoints store raw state_dict
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # ---- Load data ----
    # IMPORTANT: your code does NOT support aug_strength="none"
    # "weak" is the correct test-time choice in your repo
    _, test_loader = get_dataloaders(
        batch_size=128,
        aug_strength="weak"
    )

    T = fit_temperature(model, test_loader, device)
    print("Learned temperature:", T)

    records = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probs = F.softmax(logits/T, dim=1)

            confs, preds = probs.max(dim=1)

            for i in range(len(targets)):
                records.append({
                    "confidence": confs[i].item(),
                    "prediction": preds[i].item(),
                    "label": targets[i].item(),
                    "correct": int(preds[i].item() == targets[i].item())
                })

    df = pd.DataFrame(records)

    num_bins = 10
    bins = np.linspace(0.0, 1.0, num_bins + 1)

    bin_records = []

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        if i == num_bins - 1:
            mask = (df["confidence"] >= bin_lower) & (df["confidence"] <= bin_upper)
        else:
            mask = (df["confidence"] >= bin_lower) & (df["confidence"] < bin_upper)

        bin_df = df[mask]
        
        if len(bin_df) == 0:
            bin_acc = 0.0
            bin_conf = 0.0
        else:
            bin_acc = bin_df["correct"].mean()
            bin_conf = bin_df["confidence"].mean()
        
        bin_records.append({
            "bin": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
            "count": len(bin_df),
            "avg_confidence": bin_conf,
            "accuracy": bin_acc
        })
    
    bin_df = pd.DataFrame(bin_records)

# ---- Reliability Diagram (Calibration Baseline) ----
    x = bin_df["avg_confidence"]
    y = bin_df["accuracy"]

    plt.figure(figsize=(6,6))

    plt.plot([0,1], [0,1], linestyle="--")

    plt.scatter(x,y)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (Calibration Baseline)")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/reliability_baseline.png")
    plt.close()
    
# --- Reliability Diagra, (After Temperature Scaling) --- 

    x = bin_df["avg_confidence"]
    y = bin_df["accuracy"]

    plt.figure(figsize=(6,6))
    plt.plot([0,1], [0,1], linestyle="--")

    plt.scatter(x,y)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (After Temperature Scaling)")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/reliability_temp_scaled.png")
    plt.close()

# ---- Expected Calibration Error (ECE) ----

    ece = 0.0
    total_count = bin_df["count"].sum()

    for _, row in bin_df.iterrows():
        bin_weight = row["count"] / total_count
        ece += bin_weight * abs(row["accuracy"] - row["avg_confidence"])
    print("ECE:", ece)

# ---- Save calibration table ----

    calib_table = pd.DataFrame([
        {"metric": "accuracy", "value": df["confidence"].mean()},
        {"metric": "avg_confience", "value": df["confidence"].mean()},
        {"metric": "ece", "value": ece},
        {"metric": "num_bins", "value": 10},
        {"metric": "num_samples", "value": total_count}
    ])
    
    calib_table.to_csv(
        "results/tables/calibration_baseline.csv",
        index=False
    )

    print(bin_df)
    print("Total samples:", bin_df["count"].sum())
    print("Num samples:", len(df))
    print("Accuracy:", df["correct"].mean())
    print("Avg confidence:", df["confidence"].mean())


if __name__ == "__main__":
    main()
