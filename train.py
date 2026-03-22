"""
StutterNet+ — Training pipeline for Urdu stuttering detection.
Run: python3 train.py
"""
import os
import time
import json
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import StutterNetPlus, count_parameters
from dataset import create_dataloaders


@dataclass
class TrainConfig:
    annotations_path: str = "annotations/annotations.json"
    base_dir: str = "."
    checkpoint_dir: str = "checkpoints"
    num_classes: int = 4
    dropout_rate: float = 0.5
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    val_split: float = 0.1
    seed: int = 42
    focal_gamma: float = 2.0
    patience: int = 20
    elevenlabs_only: bool = True


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class EarlyStopping:
    """Early stopping to halt training when val loss stops improving."""

    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def get_device():
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(train_samples, num_classes):
    """Compute inverse-frequency weights for loss function."""
    counts = Counter(s["label"] for s in train_samples)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        if counts.get(c, 0) > 0:
            weights.append(total / (num_classes * counts[c]))
        else:
            weights.append(0.0)  # no samples for this class
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1) * 100
    return avg_loss, accuracy, all_preds, all_labels


def train(config=None):
    """Main training function."""
    if config is None:
        config = TrainConfig()

    # Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = get_device()
    print("=" * 60)
    print("StutterNet+ Training")
    print("=" * 60)
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, train_samples, val_samples = create_dataloaders(
        config.annotations_path, config.base_dir,
        batch_size=config.batch_size, val_split=config.val_split,
        seed=config.seed, elevenlabs_only=config.elevenlabs_only,
    )

    # Model
    model = StutterNetPlus(
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Loss with class weights
    class_weights = compute_class_weights(train_samples, config.num_classes).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=config.patience)

    # Checkpoint dir
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\nTraining for {config.epochs} epochs (patience={config.patience})")
    print("-" * 75)
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Acc':>6} | {'Val Loss':>10} {'Acc':>6} | {'LR':>9} | {'Best'}")
    print("-" * 75)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = val_loss < best_val_loss
        best_marker = "*" if is_best else ""

        if is_best:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "config": asdict(config),
                "class_names": ["clean", "syllable_repetition", "word_repetition", "block"],
            }
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, "best_model.pt"))

        # Print every epoch (small dataset, fast training)
        print(f"[{epoch:>3}/{config.epochs}] | {train_loss:>10.4f} {train_acc:>5.1f}% | {val_loss:>10.4f} {val_acc:>5.1f}% | {lr:>9.2e} | {best_marker}")

        if early_stopping.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.patience} epochs)")
            break

    # Save last model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }, os.path.join(config.checkpoint_dir, "last_model.pt"))

    elapsed = time.time() - start_time
    print("-" * 75)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/")
    print(f"\nNext step: python3 evaluate.py")


if __name__ == "__main__":
    train()
