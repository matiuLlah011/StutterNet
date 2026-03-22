"""
Custom — Training Script
Trains the 1D CNN + BiGRU + Attention model on MFCC sequence features.

Key differences from previous phases:
  - Uses full MFCC sequences (not mean-pooled) — preserves temporal info
  - Smaller model (~85K params) — less overfitting on small dataset
  - Mixup augmentation during training for better generalization
  - Label smoothing for softer targets

Run: python3 custom_train.py
"""
import os
import sys
import time
import json
import datetime
from collections import Counter

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from custom_model import CustomStutterDetector, count_parameters


# ─── TRAINING SETTINGS ───────────────────────────────────────────
EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
PATIENCE = 15
LABEL_SMOOTHING = 0.1   # Soft targets to prevent overconfidence
MIXUP_ALPHA = 0.2       # Mixup augmentation strength
RANDOM_SEED = 42

CLASS_NAMES = ["syllable_repetition", "word_repetition", "block"]


# ─── DEVICE ───────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── MIXUP AUGMENTATION ──────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    """
    Mixup: blend two random training samples together.
    This creates virtual training examples between real ones,
    which helps the model generalize better on small datasets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup: weighted combination of two targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── METRICS ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute accuracy, macro precision, recall, F1."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = (y_true == y_pred).mean() * 100

    active_classes = sorted(set(y_true))
    precisions, recalls, f1s = [], [], []
    per_class = {}

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
        per_class[CLASS_NAMES[c]] = {"precision": p, "recall": r, "f1": f1}

    return {
        "accuracy": accuracy,
        "precision": np.mean(precisions) * 100,
        "recall": np.mean(recalls) * 100,
        "f1": np.mean(f1s) * 100,
        "per_class": per_class,
    }


# ─── SAMPLE PREDICTION ───────────────────────────────────────────
def predict_on_samples(model, device, wav_paths):
    """Predict stutter type from raw audio files."""
    import librosa

    # Load annotations for true labels
    with open("annotations/annotations.json") as f:
        data = json.load(f)
    id_to_label = {os.path.basename(s["wav_file"]): s["label"] for s in data["samples"]}
    orig_to_custom = {1: 0, 2: 1, 3: 2}

    model.eval()
    results = []

    for wav_path in wav_paths:
        # Extract features (same as custom_features.py)
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta, delta2]).T  # (frames, 120)

        # Normalize
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        # Pad/truncate to 220 frames
        if features.shape[0] >= 220:
            features = features[:220]
        else:
            padding = np.zeros((220 - features.shape[0], 120))
            features = np.vstack([features, padding])

        # Predict
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[pred_class].item() * 100

        pred_name = CLASS_NAMES[pred_class]
        basename = os.path.basename(wav_path)
        orig_label = id_to_label.get(basename, -1)
        true_class = orig_to_custom.get(orig_label, -1)
        true_name = CLASS_NAMES[true_class] if true_class >= 0 else "unknown"
        correct = pred_class == true_class

        print(f"  File: {basename}")
        print(f"  Prediction: {pred_name} (STUTTERED)")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  True label: {true_name}")
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


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    device = get_device()

    print("=" * 60)
    print("CUSTOM — 1D CNN + BiGRU + Attention")
    print("=" * 60)
    print(f"Device: {device}")

    # Load pre-extracted features
    feature_file = "custom_features.npz"
    if not os.path.exists(feature_file):
        print(f"ERROR: {feature_file} not found. Run custom_features.py first.")
        sys.exit(1)

    data = np.load(feature_file, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Feature shape: {X_train.shape}")

    # Apply oversampling to balance training classes
    counts = Counter(y_train)
    max_count = max(counts.values())
    X_balanced, y_balanced = [], []
    for cls in sorted(counts.keys()):
        mask = y_train == cls
        X_cls = X_train[mask]
        y_cls = y_train[mask]
        # Repeat to match majority class
        repeats = max_count // len(X_cls)
        remainder = max_count % len(X_cls)
        X_balanced.append(np.tile(X_cls, (repeats, 1, 1)))
        X_balanced.append(X_cls[:remainder])
        y_balanced.append(np.tile(y_cls, repeats))
        y_balanced.append(y_cls[:remainder])

    X_train_bal = np.concatenate(X_balanced)
    y_train_bal = np.concatenate(y_balanced)
    print(f"After oversampling: {len(X_train_bal)} train samples")
    print(f"  Class counts: {dict(Counter(y_train_bal))}")

    # Split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_bal, y_train_bal, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train_bal
    )

    # Create data loaders
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

    # Build model
    model = CustomStutterDetector(
        input_features=120, num_classes=3, dropout=DROPOUT
    ).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model: {train_p:,} parameters")

    # Loss with label smoothing (reduces overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/custom_best.pt"
    best_val_loss = float("inf")
    patience_counter = 0
    start = time.time()

    print(f"\nTraining for {EPOCHS} epochs (patience={PATIENCE})...")
    print(f"Using mixup (alpha={MIXUP_ALPHA}) + label smoothing ({LABEL_SMOOTHING})")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        # ── Train with mixup ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Apply mixup augmentation
            mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(mixed_x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            # For accuracy, use original (unmixed) predictions
            with torch.no_grad():
                preds = model(X_batch).argmax(1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1) * 100
        scheduler.step()

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
        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"[{epoch:>2}/{EPOCHS}] train_loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.1f}% | lr={lr_now:.2e} {marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s. Best val loss: {best_val_loss:.4f}")

    # ── Evaluate on test set ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    metrics = compute_metrics(all_labels, all_preds)

    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  Precision: {metrics['precision']:.1f}%")
    print(f"  Recall:    {metrics['recall']:.1f}%")
    print(f"  F1 Score:  {metrics['f1']:.1f}%")
    print(f"\n  Per-class breakdown:")
    for name, m in metrics["per_class"].items():
        print(f"    {name}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

    # ── Sample predictions ────────────────────────────────────
    print(f"\nSample Predictions:")
    print("-" * 40)
    test_files = [
        "samples/syllable_repetition/HARF_005.wav",
        "samples/word_repetition/LAFZ_020.wav",
        "samples/block/BLOCK_010.wav",
    ]
    sample_results = predict_on_samples(model, device, test_files)

    # ── Save results ──────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_log.txt", "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"PHASE 4 — Custom (1D CNN + BiGRU + Attention)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Architecture: Conv1D→Conv1D→BiGRU→Attention→Dense\n")
        f.write(f"Features: 40 MFCC + 40 delta + 40 delta² = 120 per frame\n")
        f.write(f"Params: {train_p:,} (vs 722K FluentNet, 434K StutterNet)\n")
        f.write(f"Training: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}\n")
        f.write(f"  mixup={MIXUP_ALPHA}, label_smoothing={LABEL_SMOOTHING}\n")
        f.write(f"\nAccuracy:  {metrics['accuracy']:.1f}%\n")
        f.write(f"Precision: {metrics['precision']:.1f}%\n")
        f.write(f"Recall:    {metrics['recall']:.1f}%\n")
        f.write(f"F1 Score:  {metrics['f1']:.1f}%\n")
        f.write(f"\nPer-class:\n")
        for name, m in metrics["per_class"].items():
            f.write(f"  {name}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}\n")
        f.write(f"\nSample Predictions:\n")
        for r in sample_results:
            f.write(f"  File: {r['file']}\n")
            f.write(f"  Prediction: {r['prediction']} (Confidence: {r['confidence']})\n")
            f.write(f"  True label: {r['true_label']} — {'CORRECT' if r['correct'] else 'WRONG'}\n\n")
        f.write(f"What changed: New custom architecture — 1D CNN + BiGRU + Attention with MFCC sequences\n")
        f.write(f"{'=' * 60}\n\n")

    # ── Final comparison ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Phase 1 (FluentNet):   Accuracy: 71.3%  F1: 71.0%")
    print(f"  Phase 2 (Clean Data):  Accuracy: 55.9%  F1: 54.8%")
    print(f"  Phase 3 (StutterNet):  Accuracy: 40.4%  F1: 35.4%")
    print(f"  Phase 4 (Custom):      Accuracy: {metrics['accuracy']:.1f}%  F1: {metrics['f1']:.1f}%")
    print("=" * 60)

    best_name = "Phase 1 (FluentNet)" if 71.0 > metrics["f1"] else "Phase 4 (Custom)"
    if metrics["f1"] > 71.0:
        best_name = "Phase 4 (Custom)"
    print(f"  Best performing approach: {best_name}")
    print("=" * 60)

    # Final log entry
    with open("results_log.txt", "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"FINAL RESULTS SUMMARY\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Phase 1 (FluentNet):   Accuracy: 71.3%  F1: 71.0%\n")
        f.write(f"Phase 2 (Clean Data):  Accuracy: 55.9%  F1: 54.8%\n")
        f.write(f"Phase 3 (StutterNet):  Accuracy: 40.4%  F1: 35.4%\n")
        f.write(f"Phase 4 (Custom):      Accuracy: {metrics['accuracy']:.1f}%  F1: {metrics['f1']:.1f}%\n")
        f.write(f"Best: {best_name}\n")
        f.write(f"{'=' * 60}\n\n")

    print(f"\nAll results saved to results_log.txt")
    return metrics


if __name__ == "__main__":
    main()
