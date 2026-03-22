"""
Phase 2 — Dataset cleaning and retraining.
Creates a filtered dataset that skips problematic samples,
then retrains the model on clean data only.

Does NOT delete any original files.

Run: python3 phase2_clean.py
"""
import os
import sys
import json
import time
import datetime
from collections import Counter

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import StutterNetPlus, count_parameters
from dataset import StutterNetDataset, SpecAugment
from torch.utils.data import DataLoader


# ─── LABEL NAMES ───────────────────────────────────────────────────
CLASS_NAMES = ["clean", "syllable_repetition", "word_repetition", "block"]

# ─── CLEANING THRESHOLDS ──────────────────────────────────────────
MIN_DURATION = 3.0    # Skip audio shorter than 3 seconds
MAX_DURATION = 15.0   # Skip audio longer than 15 seconds
MIN_SPEC_VAR = 0.01   # Skip spectrograms with very low variance (near-silence)


# ─── FILTER SAMPLES ──────────────────────────────────────────────
def filter_samples(samples):
    """
    Filter out problematic samples based on duration and spectrogram quality.
    Returns (kept_samples, skipped_samples_with_reasons).
    """
    kept = []
    skipped = []

    for s in samples:
        wav_path = s["wav_file"]
        spec_path = s["spectrogram_file"]
        sid = s["id"]
        reasons = []

        # Check 1: WAV file must exist
        if not os.path.exists(wav_path):
            reasons.append("missing WAV file")
            skipped.append((sid, reasons))
            continue

        # Check 2: Spectrogram file must exist
        if not os.path.exists(spec_path):
            reasons.append("missing spectrogram file")
            skipped.append((sid, reasons))
            continue

        # Check 3: Audio duration within acceptable range
        try:
            info = sf.info(wav_path)
            duration = info.duration
            if duration < MIN_DURATION:
                reasons.append(f"too short ({duration:.1f}s < {MIN_DURATION}s)")
            if duration > MAX_DURATION:
                reasons.append(f"too long ({duration:.1f}s > {MAX_DURATION}s)")
        except Exception as e:
            reasons.append(f"corrupt audio: {e}")

        # Check 4: Spectrogram quality
        try:
            spec = np.load(spec_path)
            if np.isnan(spec).any() or np.isinf(spec).any():
                reasons.append("NaN/Inf in spectrogram")
            variance = np.var(spec)
            if variance < MIN_SPEC_VAR:
                reasons.append(f"low spectrogram variance ({variance:.4f})")
        except Exception as e:
            reasons.append(f"corrupt spectrogram: {e}")

        # Keep or skip
        if reasons:
            skipped.append((sid, reasons))
        else:
            kept.append(s)

    return kept, skipped


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
            focal_weight = self.alpha[targets] * focal_weight
        loss = focal_weight * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ─── EARLY STOPPING ───────────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=25, min_delta=1e-4):
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


# ─── DEVICE ───────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── CLASS WEIGHTS ────────────────────────────────────────────────
def compute_class_weights(samples, num_classes=4):
    """Inverse-frequency weights for focal loss."""
    counts = Counter(s["label"] for s in samples)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        if counts.get(c, 0) > 0:
            weights.append(total / (num_classes * counts[c]))
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float32)


# ─── STRATIFIED SPLIT ────────────────────────────────────────────
def stratified_split(samples, val_split=0.2, seed=42):
    """Split samples into train/val maintaining class proportions."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    label_to_samples = {}
    for s in samples:
        label_to_samples.setdefault(s["label"], []).append(s)

    train_samples, val_samples = [], []
    for lab, group in label_to_samples.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * val_split))
        val_samples.extend(group[:n_val])
        train_samples.extend(group[n_val:])

    return train_samples, val_samples


# ─── TRAINING FUNCTIONS ──────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """One training epoch."""
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


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate on validation set."""
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
def predict_samples(model, device, sample_ids, all_samples):
    """Run prediction on specific sample IDs and print results."""
    id_to_sample = {s["id"]: s for s in all_samples}
    model.eval()
    results = []
    for sid in sample_ids:
        s = id_to_sample.get(sid)
        if s is None:
            continue
        spec = np.load(s["spectrogram_file"])
        spec_tensor = torch.from_numpy(spec).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(spec_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[pred_class].item() * 100
        true_label = s["label"]
        pred_name = CLASS_NAMES[pred_class]
        true_name = CLASS_NAMES[true_label]
        print(f"  File: {os.path.basename(s['wav_file'])}")
        print(f"  Prediction: {pred_name} ({'STUTTERED' if pred_class != 0 else 'CLEAN'})")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  True label: {true_name}")
        print(f"  {'CORRECT' if pred_class == true_label else 'WRONG'}")
        print()
        results.append({
            "file": os.path.basename(s["wav_file"]),
            "prediction": pred_name,
            "confidence": f"{confidence:.1f}%",
            "true_label": true_name,
            "correct": pred_class == true_label,
        })
    return results


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    device = get_device()

    # Load all samples
    with open("annotations/annotations.json") as f:
        data = json.load(f)
    all_samples = data["samples"]

    # ─── STEP 1: Filter out bad samples ────────────────────────
    print("=" * 60)
    print("PHASE 2 — DATASET CLEANING")
    print("=" * 60)

    kept, skipped = filter_samples(all_samples)

    print(f"\nFiltering results:")
    print(f"  Original samples: {len(all_samples)}")
    print(f"  Kept: {len(kept)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"\nSkipped samples:")
    for sid, reasons in skipped:
        print(f"  {sid}: {', '.join(reasons)}")

    # Class distribution after cleaning
    clean_counts = Counter(s["label"] for s in kept)
    print(f"\nClass distribution after cleaning:")
    for label in sorted(clean_counts.keys()):
        name = CLASS_NAMES[label]
        orig = sum(1 for s in all_samples if s["label"] == label)
        print(f"  {name}: {clean_counts[label]} (was {orig})")

    # ─── STEP 2: Train on clean data ──────────────────────────
    print(f"\n{'=' * 60}")
    print("TRAINING ON CLEAN DATA")
    print("=" * 60)

    # Use best hyperparams from Phase 1 Attempt 2
    EPOCHS = 120
    BATCH_SIZE = 8
    LR = 3e-4
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.3
    FOCAL_GAMMA = 1.5
    PATIENCE = 25

    print(f"Using Phase 1 best hyperparameters:")
    print(f"  lr={LR}, batch_size={BATCH_SIZE}, dropout={DROPOUT}")
    print(f"  focal_gamma={FOCAL_GAMMA}, patience={PATIENCE}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Stratified split on clean data
    train_samples, val_samples = stratified_split(kept, val_split=0.2, seed=42)
    print(f"\nDataset split: {len(train_samples)} train, {len(val_samples)} val")
    train_labels = Counter(s["label"] for s in train_samples)
    val_labels = Counter(s["label"] for s in val_samples)
    print(f"  Train labels: {dict(train_labels)}")
    print(f"  Val labels:   {dict(val_labels)}")

    # Create data loaders with augmentation and oversampling
    augment = SpecAugment()
    train_ds = StutterNetDataset(train_samples, ".", transform=augment, oversample=True)
    val_ds = StutterNetDataset(val_samples, ".", transform=None, oversample=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model
    model = StutterNetPlus(num_classes=4, dropout_rate=DROPOUT).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model: {train_p:,} trainable parameters")

    # Loss with class weights
    class_weights = compute_class_weights(train_samples).to(device)
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=PATIENCE)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/phase2_clean.pt"

    print(f"\nTraining for up to {EPOCHS} epochs...")
    print("-" * 70)

    best_val_loss = float("inf")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
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
                "config": {"num_classes": 4, "dropout_rate": DROPOUT},
            }, ckpt_path)

        marker = "*" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"[{epoch:>3}/{EPOCHS}] train_loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.1f}% | lr={lr_now:.2e} {marker}")

        if early_stopping.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s. Best val loss: {best_val_loss:.4f}")

    # ─── STEP 3: Evaluate on VALIDATION SET ONLY (honest evaluation) ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eval_ds = StutterNetDataset(val_samples, ".", transform=None, oversample=False)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, labels in eval_loader:
            specs = specs.to(device)
            preds = model(specs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    print(f"\n  (Evaluated on {len(val_samples)} validation samples only)")

    metrics = compute_metrics(all_labels, all_preds)

    print(f"\n{'=' * 60}")
    print("PHASE 2 RESULTS (Clean Data)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  Precision: {metrics['precision']:.1f}%")
    print(f"  Recall:    {metrics['recall']:.1f}%")
    print(f"  F1 Score:  {metrics['f1']:.1f}%")

    # Compare to Phase 1 best
    phase1_f1 = 71.0
    improved = metrics["f1"] > phase1_f1
    print(f"\n  Phase 1 best F1: {phase1_f1:.1f}%")
    print(f"  Phase 2 F1:      {metrics['f1']:.1f}%")
    print(f"  Improved: {'YES' if improved else 'NO'}")

    # ─── STEP 4: Sample predictions ──────────────────────────
    print(f"\nSample Predictions:")
    print("-" * 40)
    test_ids = ["HARF_005", "LAFZ_020", "BLOCK_010"]
    sample_results = predict_samples(model, device, test_ids, kept)

    # ─── STEP 5: Save results ────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_log.txt", "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"PHASE 2 — Clean Dataset Training\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Samples: {len(kept)} kept, {len(skipped)} skipped\n")
        f.write(f"Skipped samples:\n")
        for sid, reasons in skipped:
            f.write(f"  {sid}: {', '.join(reasons)}\n")
        f.write(f"\nAccuracy:  {metrics['accuracy']:.1f}%\n")
        f.write(f"Precision: {metrics['precision']:.1f}%\n")
        f.write(f"Recall:    {metrics['recall']:.1f}%\n")
        f.write(f"F1 Score:  {metrics['f1']:.1f}%\n")
        f.write(f"\nSample Predictions:\n")
        for r in sample_results:
            f.write(f"  File: {r['file']}\n")
            f.write(f"  Prediction: {r['prediction']} (Confidence: {r['confidence']})\n")
            f.write(f"  True label: {r['true_label']} — {'CORRECT' if r['correct'] else 'WRONG'}\n\n")
        f.write(f"What changed: Removed {len(skipped)} bad samples (too short <3s, too long >15s)\n")
        f.write(f"Improved: {'YES' if improved else 'NO'} (Phase 1 F1={phase1_f1:.1f}% vs Phase 2 F1={metrics['f1']:.1f}%)\n")
        f.write(f"{'=' * 60}\n\n")

    print(f"\nResults saved to results_log.txt")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
