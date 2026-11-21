"""Train the CNN edge detector on the prepared BSDS500 patches."""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from cnn_model import EdgeDetectionCNN
from dataset_prepare import BSDSDataset, create_dataloaders


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 30


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_positive_weight(dataset: BSDSDataset) -> float:
    """Estimate negative-to-positive pixel frequency using training annotations only."""
    positive_pixels = 0.0
    total_pixels = 0
    for _, mat_path in dataset.samples:
        edge_map = dataset._load_ground_truth(mat_path)
        positive_pixels += float(edge_map.sum())
        total_pixels += edge_map.size

    negative_pixels = total_pixels - positive_pixels
    if positive_pixels <= 0.0:
        raise ValueError("Training annotations contain no positive edge pixels")
    return negative_pixels / positive_pixels


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch and return its mean loss."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, edges in loader:
            images = images.to(device)
            edges = edges.to(device)
            logits = model(images)
            loss = loss_function(logits, edges)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_history(history: list[dict[str, float]], output_path: Path) -> None:
    """Write epoch-level losses to a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(history: list[dict[str, float]], output_path: Path) -> None:
    """Plot and save training and validation loss curves."""
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Training loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.title("CNN edge-detection training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train(epochs: int, project_dir: Path) -> None:
    """Train for the requested number of epochs and save the best artifacts."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = project_dir / "dataset"
    model_dir = project_dir / "models"
    output_dir = project_dir / "outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The existing factory supplies train, validation, and test loaders.
    train_loader, val_loader, _ = create_dataloaders(dataset_root, batch_size=BATCH_SIZE)
    positive_weight = estimate_positive_weight(train_loader.dataset)
    print(f"Using device: {device}")
    print(f"Training positive-class weight: {positive_weight:.6f}")

    model = EdgeDetectionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_model_path = model_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loss = run_epoch(model, train_loader, loss_function, device, optimizer)
        validation_loss = run_epoch(model, val_loader, loss_function, device)
        elapsed = time.perf_counter() - start_time
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss})

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"val loss: {validation_loss:.6f} | "
            f"elapsed: {elapsed:.2f}s"
        )

    save_history(history, output_dir / "training_history.csv")
    save_loss_curve(history, output_dir / "loss_curve.png")
    print(f"Saved best model to {best_model_path}")
    print(f"Saved training history to {output_dir / 'training_history.csv'}")
    print(f"Saved loss curve to {output_dir / 'loss_curve.png'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options, keeping 30 epochs as the default."""
    parser = argparse.ArgumentParser(description="Train the CNN edge detector")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"number of epochs to train (default: {DEFAULT_EPOCHS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    train(arguments.epochs, Path(__file__).parent)
