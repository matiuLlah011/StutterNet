"""
Phase 1 — Train and evaluate the FluentNet model with improvements.
Each attempt tries a different strategy to improve results.
Prints sample predictions after each attempt.

Run: python3 phase1_train.py --attempt 1
"""
import os
import sys
import time
import json
import datetime
from collections import Counter
from dataclasses import dataclass, asdict

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import StutterNetPlus, count_parameters
from dataset import create_dataloaders, StutterNetDataset


# ─── LABEL NAMES ───────────────────────────────────────────────────
CLASS_NAMES = ["clean", "syllable_repetition", "word_repetition", "block"]


# ─── FOCAL LOSS ────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, focuses on hard ones."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor)
                                 else torch.tensor(alpha, dtype=torch.float32))
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
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ─── EARLY STOPPING ───────────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation loss stops improving."""
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


# ─── DEVICE DETECTION ─────────────────────────────────────────────
def get_device():
    """Pick the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── CLASS WEIGHTS ─────────────────────────────────────────────────
def compute_class_weights(train_samples, num_classes):
    """Inverse-frequency weights so rare classes get more attention."""
    counts = Counter(s["label"] for s in train_samples)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        if counts.get(c, 0) > 0:
            weights.append(total / (num_classes * counts[c]))
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float32)


# ─── TRAIN ONE EPOCH ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch — forward, backward, update weights."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    return running_loss / max(total, 1), correct / max(total, 1) * 100


# ─── VALIDATE ─────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate model on validation set — no gradient computation."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
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
    return running_loss / max(total, 1), correct / max(total, 1) * 100, all_preds, all_labels


# ─── METRICS ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute accuracy, macro precision, recall, F1."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = (y_true == y_pred).mean() * 100

    # Per-class metrics for classes that have samples
    active_classes = sorted(set(y_true))
    precisions, recalls, f1s = [], [], []
    for c in active_classes:
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision": np.mean(precisions) * 100,
        "recall": np.mean(recalls) * 100,
        "f1": np.mean(f1s) * 100,
    }


# ─── SAMPLE PREDICTIONS ──────────────────────────────────────────
def predict_samples(model, device, sample_paths, annotations_path):
    """Run prediction on specific WAV files and print results."""
    import librosa

    # Load annotations to find spectrogram paths
    with open(annotations_path, "r") as f:
        data = json.load(f)
    samples = data["samples"]

    # Map wav_file -> sample info
    wav_to_sample = {}
    for s in samples:
        wav_to_sample[s["wav_file"]] = s

    model.eval()
    results = []
    for wav_path in sample_paths:
        # Find the sample in annotations
        sample_info = wav_to_sample.get(wav_path)
        if sample_info is None:
            # Try matching by filename
            for s in samples:
                if os.path.basename(s["wav_file"]) == os.path.basename(wav_path):
                    sample_info = s
                    break

        if sample_info is None:
            print(f"  WARNING: {wav_path} not found in annotations, skipping")
            continue

        # Load the spectrogram
        spec_path = sample_info["spectrogram_file"]
        spec = np.load(spec_path)  # (257, 701, 1)
        spec_tensor = torch.from_numpy(spec).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(spec_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[pred_class].item() * 100

        true_label = sample_info["label"]
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_class]

        # Determine if stuttered
        is_stuttered = pred_class != 0  # anything not clean = stuttered

        print(f"  File: {os.path.basename(wav_path)}")
        print(f"  Prediction: {pred_name} ({'STUTTERED' if is_stuttered else 'CLEAN'})")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  True label: {true_name}")
        print(f"  {'CORRECT' if pred_class == true_label else 'WRONG'}")
        print()

        results.append({
            "file": os.path.basename(wav_path),
            "prediction": pred_name,
            "confidence": f"{confidence:.1f}%",
            "true_label": true_name,
            "correct": pred_class == true_label,
        })

    return results


# ─── MAIN TRAINING FUNCTION ──────────────────────────────────────
def train_attempt(attempt_num, epochs, batch_size, lr, weight_decay,
                  dropout, focal_gamma, patience, label_smoothing=0.0):
    """Train the model with given hyperparameters and return metrics."""
    device = get_device()
    ann_path = "annotations/annotations.json"

    print("=" * 60)
    print(f"PHASE 1 — ATTEMPT {attempt_num}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hyperparameters:")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"  weight_decay={weight_decay}, dropout={dropout}")
    print(f"  focal_gamma={focal_gamma}, patience={patience}")
    if label_smoothing > 0:
        print(f"  label_smoothing={label_smoothing}")

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create data loaders
    train_loader, val_loader, train_samples, val_samples = create_dataloaders(
        ann_path, ".", batch_size=batch_size, val_split=0.2, seed=42,
    )

    # Build model
    model = StutterNetPlus(num_classes=4, dropout_rate=dropout).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model: {train_p:,} trainable parameters")

    # Loss function with class weights
    class_weights = compute_class_weights(train_samples, 4).to(device)
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    # Checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/phase1_attempt{attempt_num}.pt"

    print(f"\nTraining for up to {epochs} epochs (patience={patience})")
    print("-" * 70)

    best_val_loss = float("inf")
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "config": {"num_classes": 4, "dropout_rate": dropout},
            }, ckpt_path)

        marker = "*" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"[{epoch:>3}/{epochs}] train_loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.1f}% | lr={lr_now:.2e} {marker}")

        if early_stopping.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s. Best val loss: {best_val_loss:.4f}")

    # Load best model and evaluate on VALIDATION SET ONLY (honest evaluation)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on validation data only — NOT training data
    val_ds = StutterNetDataset(val_samples, ".", transform=None, oversample=False)
    eval_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, labels in eval_loader:
            specs = specs.to(device)
            preds = model(specs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_labels, all_preds)
    print(f"\n  (Evaluated on {len(val_samples)} validation samples only)")

    print(f"\n{'=' * 60}")
    print(f"ATTEMPT {attempt_num} RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  Precision: {metrics['precision']:.1f}%")
    print(f"  Recall:    {metrics['recall']:.1f}%")
    print(f"  F1 Score:  {metrics['f1']:.1f}%")

    # Test on 3 sample files
    print(f"\nSample Predictions:")
    print("-" * 40)
    test_files = [
        "samples/syllable_repetition/HARF_001.wav",
        "samples/word_repetition/LAFZ_010.wav",
        "samples/block/BLOCK_005.wav",
    ]
    sample_results = predict_samples(model, device, test_files, ann_path)

    return metrics, sample_results, ckpt_path


# ─── LOG RESULTS TO FILE ─────────────────────────────────────────
def log_results(attempt_name, metrics, sample_results, changes, improved):
    """Append results to results_log.txt."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_log.txt", "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"{attempt_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.1f}%\n")
        f.write(f"Precision: {metrics['precision']:.1f}%\n")
        f.write(f"Recall:    {metrics['recall']:.1f}%\n")
        f.write(f"F1 Score:  {metrics['f1']:.1f}%\n")
        f.write(f"\nSample Predictions:\n")
        for r in sample_results:
            f.write(f"  File: {r['file']}\n")
            f.write(f"  Prediction: {r['prediction']} (Confidence: {r['confidence']})\n")
            f.write(f"  True label: {r['true_label']} — {'CORRECT' if r['correct'] else 'WRONG'}\n\n")
        f.write(f"What changed: {changes}\n")
        f.write(f"Improved: {'YES' if improved else 'NO'}\n")
        f.write(f"{'=' * 60}\n\n")


# ─── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=0,
                        help="Which attempt to run: 1, 2, 3, or 0 for all")
    args = parser.parse_args()

    # Clear results log at start of Phase 1
    if args.attempt in [0, 1]:
        with open("results_log.txt", "w") as f:
            f.write("STUTTERNET+ RESULTS LOG\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

    best_f1 = 0.0
    best_attempt = ""

    # ─── ATTEMPT 1: Hyperparameter tuning ──────────────────────
    if args.attempt in [0, 1]:
        print("\n" + "#" * 60)
        print("# ATTEMPT 1: Hyperparameter Tuning")
        print("# Changes: lr=5e-4, batch_size=8, patience=25, epochs=120")
        print("#" * 60 + "\n")

        m1, s1, _ = train_attempt(
            attempt_num=1,
            epochs=120,
            batch_size=8,
            lr=5e-4,
            weight_decay=5e-4,
            dropout=0.5,
            focal_gamma=2.0,
            patience=25,
        )
        log_results(
            "Phase 1 — Attempt 1: Hyperparameter Tuning",
            m1, s1,
            "Changed lr from 1e-3 to 5e-4, batch_size from 4 to 8, "
            "weight_decay from 1e-3 to 5e-4, patience from 20 to 25, epochs from 100 to 120",
            m1["f1"] > 47.0,  # compare to previous best macro F1 of 47%
        )
        if m1["f1"] > best_f1:
            best_f1 = m1["f1"]
            best_attempt = "Attempt 1"

    # ─── ATTEMPT 2: Better augmentation + lower dropout ────────
    if args.attempt in [0, 2]:
        print("\n" + "#" * 60)
        print("# ATTEMPT 2: Lower dropout + focal_gamma=1.5")
        print("# Changes: dropout=0.3, focal_gamma=1.5, lr=3e-4")
        print("#" * 60 + "\n")

        m2, s2, _ = train_attempt(
            attempt_num=2,
            epochs=120,
            batch_size=8,
            lr=3e-4,
            weight_decay=1e-3,
            dropout=0.3,
            focal_gamma=1.5,
            patience=25,
        )
        log_results(
            "Phase 1 — Attempt 2: Lower Dropout + Focal Gamma",
            m2, s2,
            "Changed dropout from 0.5 to 0.3, focal_gamma from 2.0 to 1.5, "
            "lr to 3e-4, weight_decay back to 1e-3",
            m2["f1"] > best_f1,
        )
        if m2["f1"] > best_f1:
            best_f1 = m2["f1"]
            best_attempt = "Attempt 2"

    # ─── ATTEMPT 3: Warmup + higher weight decay ──────────────
    if args.attempt in [0, 3]:
        print("\n" + "#" * 60)
        print("# ATTEMPT 3: Aggressive regularization + slower LR")
        print("# Changes: lr=2e-4, weight_decay=5e-3, dropout=0.4, focal_gamma=2.5")
        print("#" * 60 + "\n")

        m3, s3, _ = train_attempt(
            attempt_num=3,
            epochs=150,
            batch_size=8,
            lr=2e-4,
            weight_decay=5e-3,
            dropout=0.4,
            focal_gamma=2.5,
            patience=30,
        )
        log_results(
            "Phase 1 — Attempt 3: Aggressive Regularization",
            m3, s3,
            "Changed lr to 2e-4, weight_decay to 5e-3, dropout to 0.4, "
            "focal_gamma to 2.5, patience to 30, epochs to 150",
            m3["f1"] > best_f1,
        )
        if m3["f1"] > best_f1:
            best_f1 = m3["f1"]
            best_attempt = "Attempt 3"

    print("\n" + "=" * 60)
    print(f"PHASE 1 SUMMARY — Best: {best_attempt} with F1={best_f1:.1f}%")
    print("=" * 60)
