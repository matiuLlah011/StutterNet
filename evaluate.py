"""
StutterNet+ — Evaluation with confusion matrix and attention visualization.
Run: python3 evaluate.py
"""
import os
import argparse
import json
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import StutterNetPlus
from dataset import StutterNetDataset


CLASS_NAMES = ["clean", "syllable_repetition", "word_repetition", "block"]
CLASS_NAMES_URDU = ["صاف", "حرف", "لفظ", "بلاک"]


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = StutterNetPlus(
        num_classes=config.get("num_classes", 4),
        dropout_rate=config.get("dropout_rate", 0.5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', 0):.4f}, val_acc={ckpt.get('val_accuracy', 0):.1f}%)")
    return model


def compute_metrics(y_true, y_pred, class_names):
    """Compute per-class precision, recall, F1."""
    num_classes = len(class_names)
    metrics = {}
    for i, name in enumerate(class_names):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != i and p == i)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == i and p != i)
        support = sum(1 for t in y_true if t == i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return metrics


def print_classification_report(y_true, y_pred, class_names):
    """Print formatted classification report."""
    metrics = compute_metrics(y_true, y_pred, class_names)
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    print(f"\n{'':>22} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print("-" * 65)
    for name in class_names:
        m = metrics[name]
        if m["support"] > 0:
            print(f"{name:>22} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f} {m['support']:>10}")
        else:
            print(f"{name:>22} {'--':>10} {'--':>10} {'--':>10} {0:>10}")
    print("-" * 65)

    # Macro average (only classes with support > 0)
    active = [m for m in metrics.values() if m["support"] > 0]
    if active:
        macro_p = np.mean([m["precision"] for m in active])
        macro_r = np.mean([m["recall"] for m in active])
        macro_f1 = np.mean([m["f1"] for m in active])
        print(f"{'macro avg':>22} {macro_p:>10.2f} {macro_r:>10.2f} {macro_f1:>10.2f} {total:>10}")

    accuracy = correct / total if total > 0 else 0
    print(f"{'accuracy':>22} {'':>10} {'':>10} {accuracy:>10.2f} {total:>10}")


def confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(y_true, y_pred, class_names, class_names_urdu, save_path):
    """Plot and save confusion matrix."""
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("StutterNet+ Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=ax)

    labels = [f"{en}\n({ur})" for en, ur in zip(class_names, class_names_urdu)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    # Text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_attention(model, sample_spec, sample_id, device, save_path):
    """Visualize attention weights for a single sample."""
    model.eval()
    with torch.no_grad():
        spec_tensor = torch.from_numpy(sample_spec).permute(2, 0, 1).float().unsqueeze(0).to(device)
        _ = model(spec_tensor)
        attn_weights = model.attention.last_attention_weights.cpu().numpy()[0]  # (T,)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]})

    # Spectrogram
    spec_2d = sample_spec[:, :, 0]  # (257, 701)
    axes[0].imshow(spec_2d, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"Spectrogram — {sample_id}", fontsize=12)
    axes[0].set_ylabel("Frequency bin")

    # Attention weights
    time_steps = np.arange(len(attn_weights))
    axes[1].bar(time_steps, attn_weights, color="crimson", alpha=0.8)
    axes[1].set_xlim(0, len(attn_weights) - 1)
    axes[1].set_xlabel("Encoded time step")
    axes[1].set_ylabel("Attention weight")
    axes[1].set_title("Attention Focus (higher = more important for classification)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate StutterNet+")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--annotations", default="annotations/annotations.json")
    parser.add_argument("--output_dir", default="evaluation_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Run train.py first.")
        return
    model = load_model(args.checkpoint, device)

    # Load all data (no augmentation, no oversampling)
    with open(args.annotations, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    # Filter to ElevenLabs voices only
    from dataset import ELEVENLABS_VOICES
    samples = [s for s in samples if s.get("voice_used") in ELEVENLABS_VOICES]
    print(f"Filtered to ElevenLabs voices only: {len(samples)} samples")

    dataset = StutterNetDataset(samples, base_dir=".", transform=None, oversample=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Evaluate
    print("\n" + "=" * 60)
    print("StutterNet+ Evaluation Report")
    print("=" * 60)

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device)
            logits = model(specs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Classification report
    print_classification_report(y_true, y_pred, CLASS_NAMES)

    # Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, CLASS_NAMES_URDU, cm_path)

    # Attention visualizations for a few samples
    print("\nGenerating attention visualizations...")
    for i, sample in enumerate(samples[:3]):  # first 3 samples
        spec = np.load(os.path.join(".", sample["spectrogram_file"]))
        attn_path = os.path.join(args.output_dir, f"attention_{sample['id']}.png")
        plot_attention(model, spec, sample["id"], device, attn_path)

    # Summary
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    print(f"\n{'=' * 60}")
    print(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
