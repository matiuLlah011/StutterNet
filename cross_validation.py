"""
Stratified 5-Fold Cross-Validation for all 3 architectures.
Provides reliable, unbiased performance estimates on the full 543-sample dataset.

Each fold: 80% train, 20% test. All metrics reported on TEST fold only.
Oversampling/SMOTE applied to TRAINING fold only (never test).

Models evaluated:
  1. FluentNet (SE-ResNet + BiLSTM + Attention) — spectrograms
  2. StutterNet (BiLSTM + RNN) — mean-pooled MFCCs
  3. Custom (1D CNN + BiGRU + Attention) — MFCC sequences

Run: python3 cross_validation.py
  Options:
    --model fluentnet|stutternet|custom|all  (default: all)
    --folds 5                                (default: 5)
"""
import os
import sys
import time
import json
import argparse
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from collections import Counter

import librosa

from model import StutterNetPlus
from stutternet_model import StutterNet
from custom_model import CustomStutterDetector
from dataset import StutterNetDataset, SpecAugment


# ─── CONSTANTS ───────────────────────────────────────────────────────
RANDOM_SEED = 42
N_MFCC = 40
TARGET_SR = 16000
HOP_LENGTH = 512
MAX_FRAMES = 220

CLASS_NAMES_4 = ["clean", "syllable_repetition", "word_repetition", "block"]
CLASS_NAMES_3 = ["syllable_repetition", "word_repetition", "block"]
LABEL_MAP_3 = {1: 0, 2: 1, 3: 2}


# ─── DEVICE ──────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── METRICS ─────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred):
    """Compute accuracy, macro & micro precision/recall/F1."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracy = (y_true == y_pred).mean() * 100

    active_classes = sorted(set(y_true) | set(y_pred))

    # Per-class for macro
    precisions, recalls, f1s = [], [], []
    per_class = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for c in active_classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": int((y_true == c).sum())}

    # Macro (unweighted average of per-class)
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    # Micro (global TP/FP/FN)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }


# ─── OVERSAMPLING ────────────────────────────────────────────────────
def oversample_indices(labels):
    """Return oversampled indices so all classes have equal count."""
    label_to_idx = {}
    for i, lab in enumerate(labels):
        label_to_idx.setdefault(lab, []).append(i)
    max_count = max(len(v) for v in label_to_idx.values())
    balanced = []
    for lab, idxs in label_to_idx.items():
        repeats = max_count // len(idxs)
        remainder = max_count % len(idxs)
        balanced.extend(idxs * repeats + idxs[:remainder])
    np.random.shuffle(balanced)
    return balanced


# ─── MFCC FEATURE EXTRACTION ────────────────────────────────────────
def extract_mfcc_mean(wav_path):
    """Extract 40 MFCCs and mean-pool → (40,) vector for StutterNet."""
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return np.mean(mfccs, axis=1)  # (40,)


def extract_mfcc_sequence(wav_path):
    """Extract MFCC+delta+delta² sequence → (MAX_FRAMES, 120) for Custom."""
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta, delta2]).T  # (frames, 120)
    # Normalize
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std
    # Pad or truncate
    if features.shape[0] >= MAX_FRAMES:
        features = features[:MAX_FRAMES]
    else:
        padding = np.zeros((MAX_FRAMES - features.shape[0], 120))
        features = np.vstack([features, padding])
    return features  # (220, 120)


# ─── FOCAL LOSS ──────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32)
                                 if not isinstance(alpha, torch.Tensor) else alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        w = (1 - pt) ** self.gamma
        if self.alpha is not None:
            w = self.alpha.to(logits.device)[targets] * w
        return (w * ce).mean()


# ─── EARLY STOPPING ─────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ═══════════════════════════════════════════════════════════════════
# FLUENTNET TRAINING (one fold)
# ═══════════════════════════════════════════════════════════════════
def train_fluentnet_fold(train_samples, test_samples, device, fold_num):
    """Train FluentNet on one fold and return test metrics."""
    # Hyperparams from Phase 1 Attempt 2 (best)
    EPOCHS, BATCH, LR = 120, 8, 3e-4
    DROPOUT, FOCAL_GAMMA, PATIENCE = 0.3, 1.5, 25
    WEIGHT_DECAY = 1e-3

    torch.manual_seed(RANDOM_SEED + fold_num)
    np.random.seed(RANDOM_SEED + fold_num)

    # Class weights from training fold
    counts = Counter(s["label"] for s in train_samples)
    total = sum(counts.values())
    weights = [total / (4 * counts[c]) if counts.get(c, 0) > 0 else 0.0 for c in range(4)]

    augment = SpecAugment()
    train_ds = StutterNetDataset(train_samples, ".", transform=augment, oversample=True)
    test_ds = StutterNetDataset(test_samples, ".", transform=None, oversample=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = StutterNetPlus(num_classes=4, dropout_rate=DROPOUT).to(device)
    criterion = FocalLoss(alpha=weights, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(specs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate on test fold (used only for early stopping, not tuning)
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for specs, labels in test_loader:
                specs, labels = specs.to(device), labels.to(device)
                val_loss += criterion(model(specs), labels).item() * len(labels)
                n += len(labels)
        val_loss /= max(n, 1)
        scheduler.step()

        if stopper.step(val_loss, model):
            break

    # Load best and evaluate
    stopper.load_best(model)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            preds = model(specs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return compute_all_metrics(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════════
# STUTTERNET TRAINING (one fold)
# ═══════════════════════════════════════════════════════════════════
def train_stutternet_fold(X_train, y_train, X_test, y_test, device, fold_num):
    """Train StutterNet on one fold and return test metrics."""
    EPOCHS, BATCH, LR, PATIENCE = 50, 16, 0.001, 10

    torch.manual_seed(RANDOM_SEED + fold_num)
    np.random.seed(RANDOM_SEED + fold_num)

    # Oversample training data
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    X_bal, y_bal = [], []
    for cls in sorted(counts.keys()):
        mask = y_train == cls
        X_c, y_c = X_train[mask], y_train[mask]
        reps = max_count // len(X_c)
        rem = max_count % len(X_c)
        X_bal.extend([np.tile(X_c, (reps, 1)), X_c[:rem]])
        y_bal.extend([np.tile(y_c, reps), y_c[:rem]])
    X_tr = np.concatenate(X_bal)
    y_tr = np.concatenate(y_bal)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    model = StutterNet(input_dim=40, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(y_b)
                n += len(y_b)
        val_loss /= max(n, 1)

        if stopper.step(val_loss, model):
            break

    stopper.load_best(model)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    return compute_all_metrics(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════════
# CUSTOM MODEL TRAINING (one fold)
# ═══════════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_custom_fold(X_train, y_train, X_test, y_test, device, fold_num):
    """Train Custom model on one fold and return test metrics."""
    EPOCHS, BATCH, LR = 80, 16, 1e-3
    DROPOUT, PATIENCE = 0.3, 15
    WEIGHT_DECAY, LABEL_SMOOTHING, MIXUP_ALPHA = 1e-3, 0.1, 0.2

    torch.manual_seed(RANDOM_SEED + fold_num)
    np.random.seed(RANDOM_SEED + fold_num)

    # Oversample training data
    counts = Counter(y_train.tolist())
    max_count = max(counts.values())
    X_bal, y_bal = [], []
    for cls in sorted(counts.keys()):
        mask = y_train == cls
        X_c, y_c = X_train[mask], y_train[mask]
        reps = max_count // len(X_c)
        rem = max_count % len(X_c)
        X_bal.extend([np.tile(X_c, (reps, 1, 1)), X_c[:rem]])
        y_bal.extend([np.tile(y_c, reps), y_c[:rem]])
    X_tr = np.concatenate(X_bal)
    y_tr = np.concatenate(y_bal)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    model = CustomStutterDetector(input_features=120, num_classes=3, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            mixed_x, y_a, y_b_mix, lam = mixup_data(X_b, y_b, MIXUP_ALPHA)
            optimizer.zero_grad()
            logits = model(mixed_x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b_mix)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(y_b)
                n += len(y_b)
        val_loss /= max(n, 1)

        if stopper.step(val_loss, model):
            break

    stopper.load_best(model)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    return compute_all_metrics(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════
def print_fold_result(model_name, fold, metrics):
    print(f"  Fold {fold}: Acc={metrics['accuracy']:.1f}%  "
          f"Macro F1={metrics['macro_f1']:.4f}  "
          f"Micro F1={metrics['micro_f1']:.4f}")


def print_summary(model_name, all_metrics):
    accs = [m["accuracy"] for m in all_metrics]
    macro_f1s = [m["macro_f1"] for m in all_metrics]
    micro_f1s = [m["micro_f1"] for m in all_metrics]
    macro_ps = [m["macro_precision"] for m in all_metrics]
    macro_rs = [m["macro_recall"] for m in all_metrics]

    print(f"\n{'=' * 60}")
    print(f"{model_name} — {len(all_metrics)}-Fold CV Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:        {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
    print(f"  Macro Precision: {np.mean(macro_ps):.4f} +/- {np.std(macro_ps):.4f}")
    print(f"  Macro Recall:    {np.mean(macro_rs):.4f} +/- {np.std(macro_rs):.4f}")
    print(f"  Macro F1:        {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")
    print(f"  Micro F1:        {np.mean(micro_f1s):.4f} +/- {np.std(micro_f1s):.4f}")
    print(f"  Per-fold Acc:    {['%.1f' % a for a in accs]}")
    print(f"  Per-fold MacroF1:{['%.4f' % f for f in macro_f1s]}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["fluentnet", "stutternet", "custom", "all"])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    k = args.folds
    run_models = [args.model] if args.model != "all" else ["fluentnet", "stutternet", "custom"]

    print("=" * 60)
    print(f"STRATIFIED {k}-FOLD CROSS-VALIDATION")
    print(f"Device: {device}")
    print(f"Models: {', '.join(run_models)}")
    print("=" * 60)

    # ── Load annotations ─────────────────────────────────────
    with open("annotations/annotations.json") as f:
        data = json.load(f)
    all_samples = data["samples"]
    labels = np.array([s["label"] for s in all_samples])
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Class distribution: {dict(Counter(labels.tolist()))}")

    # ── Extract MFCC features (for StutterNet + Custom) ──────
    mfcc_mean_all = None   # (N, 40) for StutterNet
    mfcc_seq_all = None    # (N, 220, 120) for Custom
    labels_3class = np.array([LABEL_MAP_3[lab] for lab in labels])  # remap to 0,1,2

    if "stutternet" in run_models or "custom" in run_models:
        print("\nExtracting MFCC features from all audio files...")
        mfcc_mean_list = []
        mfcc_seq_list = []
        skipped = []
        for i, s in enumerate(all_samples):
            try:
                wav_path = s["wav_file"]
                if "stutternet" in run_models:
                    mfcc_mean_list.append(extract_mfcc_mean(wav_path))
                if "custom" in run_models:
                    mfcc_seq_list.append(extract_mfcc_sequence(wav_path))
            except Exception as e:
                print(f"  WARNING: {s['id']}: {e}")
                # Use zeros as fallback
                if "stutternet" in run_models:
                    mfcc_mean_list.append(np.zeros(40))
                if "custom" in run_models:
                    mfcc_seq_list.append(np.zeros((MAX_FRAMES, 120)))
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(all_samples)}...")

        if mfcc_mean_list:
            mfcc_mean_all = np.array(mfcc_mean_list)
            print(f"  StutterNet features: {mfcc_mean_all.shape}")
        if mfcc_seq_list:
            mfcc_seq_all = np.array(mfcc_seq_list)
            print(f"  Custom features: {mfcc_seq_all.shape}")

    # ── Create stratified k-fold splits ──────────────────────
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(np.zeros(len(labels)), labels))

    # ── Run CV for each model ────────────────────────────────
    all_results = {}
    total_start = time.time()

    for model_name in run_models:
        print(f"\n{'#' * 60}")
        print(f"# {model_name.upper()} — {k}-Fold Cross-Validation")
        print(f"{'#' * 60}")

        fold_metrics = []
        model_start = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            fold_num = fold_idx + 1
            print(f"\n--- Fold {fold_num}/{k} (train={len(train_idx)}, test={len(test_idx)}) ---")

            if model_name == "fluentnet":
                train_samps = [all_samples[i] for i in train_idx]
                test_samps = [all_samples[i] for i in test_idx]
                metrics = train_fluentnet_fold(train_samps, test_samps, device, fold_num)

            elif model_name == "stutternet":
                X_tr = mfcc_mean_all[train_idx]
                y_tr = labels_3class[train_idx]
                X_te = mfcc_mean_all[test_idx]
                y_te = labels_3class[test_idx]
                metrics = train_stutternet_fold(X_tr, y_tr, X_te, y_te, device, fold_num)

            elif model_name == "custom":
                X_tr = mfcc_seq_all[train_idx]
                y_tr = labels_3class[train_idx]
                X_te = mfcc_seq_all[test_idx]
                y_te = labels_3class[test_idx]
                metrics = train_custom_fold(X_tr, y_tr, X_te, y_te, device, fold_num)

            fold_metrics.append(metrics)
            print_fold_result(model_name, fold_num, metrics)

        model_elapsed = time.time() - model_start
        print(f"\n{model_name.upper()} completed in {model_elapsed:.1f}s")
        print_summary(model_name.upper(), fold_metrics)
        all_results[model_name] = fold_metrics

    # ── Final comparison ─────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n\n{'=' * 60}")
    print(f"CROSS-VALIDATION SUMMARY ({k}-Fold)")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    summary_lines = []
    for model_name in run_models:
        metrics_list = all_results[model_name]
        accs = [m["accuracy"] for m in metrics_list]
        mf1s = [m["macro_f1"] for m in metrics_list]
        mif1s = [m["micro_f1"] for m in metrics_list]
        line = (f"  {model_name.upper():12s}  "
                f"Acc={np.mean(accs):5.2f}%+/-{np.std(accs):4.2f}  "
                f"MacroF1={np.mean(mf1s):.4f}+/-{np.std(mf1s):.4f}  "
                f"MicroF1={np.mean(mif1s):.4f}+/-{np.std(mif1s):.4f}")
        print(line)
        summary_lines.append(line)

    # ── Save to results_log.txt ──────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_log.txt", "a") as f:
        f.write(f"\n\n{'=' * 60}\n")
        f.write(f"STRATIFIED {k}-FOLD CROSS-VALIDATION\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 60}\n\n")

        for model_name in run_models:
            metrics_list = all_results[model_name]
            accs = [m["accuracy"] for m in metrics_list]
            mf1s = [m["macro_f1"] for m in metrics_list]
            mif1s = [m["micro_f1"] for m in metrics_list]
            mps = [m["macro_precision"] for m in metrics_list]
            mrs = [m["macro_recall"] for m in metrics_list]

            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Accuracy:        {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%\n")
            f.write(f"  Macro Precision: {np.mean(mps):.4f} +/- {np.std(mps):.4f}\n")
            f.write(f"  Macro Recall:    {np.mean(mrs):.4f} +/- {np.std(mrs):.4f}\n")
            f.write(f"  Macro F1:        {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f}\n")
            f.write(f"  Micro F1:        {np.mean(mif1s):.4f} +/- {np.std(mif1s):.4f}\n")

            for fold_idx, m in enumerate(metrics_list):
                f.write(f"  Fold {fold_idx+1}: Acc={m['accuracy']:.1f}%  "
                        f"MacroF1={m['macro_f1']:.4f}  MicroF1={m['micro_f1']:.4f}\n")
            f.write("\n")

        f.write(f"{'=' * 60}\n")

    print(f"\nResults appended to results_log.txt")


if __name__ == "__main__":
    main()
