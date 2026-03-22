"""
StutterNet — Training Script
Trains the StutterNet model on MFCC features with SMOTE-balanced data.

Training settings (from paper):
  - Loss: CrossEntropyLoss (adapted from binary crossentropy for 3 classes)
  - Optimizer: Adam
  - Learning rate: 0.001
  - Epochs: 50
  - Batch size: 16
  - Early stopping on validation loss

Run: python3 stutternet_train.py
"""
import os
import sys
import time
import datetime

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from stutternet_model import StutterNet, count_parameters


# ─── TRAINING SETTINGS (from paper) ──────────────────────────────
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 10            # Early stopping patience
RANDOM_SEED = 42

# Class names for our 3-class problem
CLASS_NAMES = ["syllable_repetition", "word_repetition", "block"]


# ─── DEVICE DETECTION ─────────────────────────────────────────────
def get_device():
    """Pick best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── METRICS COMPUTATION ─────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute accuracy, macro precision, recall, F1 score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = (y_true == y_pred).mean() * 100

    # Per-class metrics
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


# ─── SAMPLE PREDICTION ───────────────────────────────────────────
def predict_on_samples(model, device, wav_paths):
    """Run StutterNet prediction on raw audio files."""
    import librosa
    import json

    # Need annotations to get true labels
    with open("annotations/annotations.json") as f:
        data = json.load(f)
    id_to_label = {}
    for s in data["samples"]:
        id_to_label[os.path.basename(s["wav_file"])] = s["label"]

    # Label mapping: original (1,2,3) → StutterNet (0,1,2)
    orig_to_snet = {1: 0, 2: 1, 3: 2}

    model.eval()
    results = []

    for wav_path in wav_paths:
        # Extract 40 MFCCs and take mean
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)  # 40-dim vector

        # Convert to tensor and predict
        x = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[pred_class].item() * 100

        pred_name = CLASS_NAMES[pred_class]

        # Get true label
        basename = os.path.basename(wav_path)
        orig_label = id_to_label.get(basename, -1)
        true_class = orig_to_snet.get(orig_label, -1)
        true_name = CLASS_NAMES[true_class] if true_class >= 0 else "unknown"

        print(f"  File: {basename}")
        print(f"  Prediction: {pred_name} (STUTTERED)")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  True label: {true_name}")
        correct = pred_class == true_class
        print(f"  {'CORRECT' if correct else 'WRONG'}")
        print()

        results.append({
            "file": basename,
            "prediction": pred_name,
            "confidence": f"{confidence:.1f}%",
            "true_label": true_name,
            "correct": correct,
        })

    return results


# ─── MAIN TRAINING FUNCTION ──────────────────────────────────────
def main():
    device = get_device()

    print("=" * 60)
    print("STUTTERNET — TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

    # Load pre-extracted features
    feature_file = "stutternet_features.npz"
    if not os.path.exists(feature_file):
        print(f"ERROR: {feature_file} not found. Run stutternet_features.py first.")
        sys.exit(1)

    data = np.load(feature_file, allow_pickle=True)
    X_train = data["X_train"]  # SMOTE-balanced training features
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Training samples: {len(X_train)} (SMOTE balanced)")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")

    # Split training into train/val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )
    print(f"Train/val split: {len(X_tr)} train, {len(X_val)} val")

    # Create PyTorch data loaders
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build the StutterNet model
    model = StutterNet(input_dim=40, num_classes=3).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model parameters: {train_p:,} trainable / {total_p:,} total")

    # Loss function (CrossEntropy replaces binary crossentropy for 3 classes)
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with lr=0.001 as specified in paper
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training with early stopping
    print(f"\nTraining for {EPOCHS} epochs (patience={PATIENCE})...")
    print("-" * 60)

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/stutternet_best.pt"
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1) * 100

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1) * 100

        # Check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, ckpt_path)
        else:
            patience_counter += 1

        marker = "*" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"[{epoch:>2}/{EPOCHS}] train_loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.1f}% {marker}")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")

    # ── Evaluate on test set ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    metrics = compute_metrics(all_labels, all_preds)

    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  Precision: {metrics['precision']:.1f}%")
    print(f"  Recall:    {metrics['recall']:.1f}%")
    print(f"  F1 Score:  {metrics['f1']:.1f}%")

    # ── Sample predictions ────────────────────────────────────
    print(f"\nSample Predictions:")
    print("-" * 40)
    test_files = [
        "samples/syllable_repetition/HARF_005.wav",
        "samples/word_repetition/LAFZ_020.wav",
        "samples/block/BLOCK_010.wav",
    ]
    sample_results = predict_on_samples(model, device, test_files)

    # ── Save results to log ───────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_log.txt", "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"PHASE 3 — StutterNet (MFCC + BiLSTM + RNN)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Architecture: 8-layer (BiLSTM×2 + Dropout + RNN + Dense×2)\n")
        f.write(f"Features: 40 MFCCs (mean across frames)\n")
        f.write(f"SMOTE: Applied to balance training classes\n")
        f.write(f"Training: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}\n")
        f.write(f"\nAccuracy:  {metrics['accuracy']:.1f}%\n")
        f.write(f"Precision: {metrics['precision']:.1f}%\n")
        f.write(f"Recall:    {metrics['recall']:.1f}%\n")
        f.write(f"F1 Score:  {metrics['f1']:.1f}%\n")
        f.write(f"\nSample Predictions:\n")
        for r in sample_results:
            f.write(f"  File: {r['file']}\n")
            f.write(f"  Prediction: {r['prediction']} (Confidence: {r['confidence']})\n")
            f.write(f"  True label: {r['true_label']} — {'CORRECT' if r['correct'] else 'WRONG'}\n\n")
        f.write(f"What changed: New architecture — StutterNet with MFCC features + SMOTE\n")
        f.write(f"{'=' * 60}\n\n")

    print(f"\nResults saved to results_log.txt")
    print(f"Model saved to {ckpt_path}")

    return metrics, sample_results


if __name__ == "__main__":
    main()
