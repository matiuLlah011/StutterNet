"""
Custom — Feature Extraction
Extracts MFCC sequences with delta and delta-delta features.
Unlike StutterNet (Phase 3), we keep the FULL temporal sequence
rather than averaging across frames. This preserves the patterns
that distinguish stutter types.

Features per file: (max_frames, 120)
  - 40 MFCCs (spectral shape)
  - 40 delta MFCCs (how spectrum changes — detects repetitions)
  - 40 delta-delta MFCCs (acceleration — detects sudden pauses)

Run: python3 custom_features.py
"""
import os
import json
import numpy as np
import librosa
from collections import Counter
from sklearn.model_selection import train_test_split

# ─── CONSTANTS ─────────────────────────────────────────────────────
N_MFCC = 40              # MFCC coefficients per frame
TARGET_SR = 16000         # Sample rate
HOP_LENGTH = 512          # ~32ms per frame at 16kHz
MAX_FRAMES = 220          # Max frames to keep (~7 seconds)
TEST_SPLIT = 0.2
RANDOM_SEED = 42
OUTPUT_FILE = "custom_features.npz"

# Remap labels: original (1,2,3) → (0,1,2)
LABEL_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["syllable_repetition", "word_repetition", "block"]


def extract_features(wav_path):
    """
    Extract MFCC + delta + delta-delta features from audio.
    Returns: (num_frames, 120) array — full temporal sequence.
    """
    # Load audio
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # Extract 40 MFCCs (shape: 40 × num_frames)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    # Compute delta (first derivative — rate of spectral change)
    delta = librosa.feature.delta(mfccs)

    # Compute delta-delta (second derivative — acceleration)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Stack all features: (120, num_frames)
    features = np.vstack([mfccs, delta, delta2])

    # Transpose to (num_frames, 120) for sequence processing
    features = features.T

    # Normalize each feature to zero mean, unit variance
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features


def pad_or_truncate(features, max_frames=MAX_FRAMES):
    """Pad short sequences or truncate long ones to fixed length."""
    n_frames, n_features = features.shape
    if n_frames >= max_frames:
        return features[:max_frames]
    else:
        # Pad with zeros at the end
        padding = np.zeros((max_frames - n_frames, n_features))
        return np.vstack([features, padding])


def main():
    print("=" * 60)
    print("CUSTOM — FEATURE EXTRACTION")
    print("=" * 60)

    # Load annotations
    with open("annotations/annotations.json") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"Total samples: {len(samples)}")

    # Extract features from each audio file
    all_features = []
    all_labels = []
    skipped = 0

    for i, s in enumerate(samples):
        if s["label"] not in LABEL_MAP:
            skipped += 1
            continue
        if not os.path.exists(s["wav_file"]):
            skipped += 1
            continue

        try:
            feats = extract_features(s["wav_file"])
            feats = pad_or_truncate(feats, MAX_FRAMES)
            all_features.append(feats)
            all_labels.append(LABEL_MAP[s["label"]])
        except Exception as e:
            print(f"  WARNING: {s['id']}: {e}")
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)}...")

    X = np.array(all_features)  # (num_samples, MAX_FRAMES, 120)
    y = np.array(all_labels)

    print(f"\nExtraction complete:")
    print(f"  Samples: {len(X)}, Skipped: {skipped}")
    print(f"  Feature shape: {X.shape}")
    print(f"  = {MAX_FRAMES} frames × 120 features (40 MFCC + 40 delta + 40 delta²)")

    # Class distribution
    counts = Counter(y)
    print(f"\nClass distribution:")
    for cls in sorted(counts.keys()):
        print(f"  {CLASS_NAMES[cls]}: {counts[cls]}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")

    # Save
    np.savez(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        class_names=CLASS_NAMES,
    )
    print(f"Saved to: {OUTPUT_FILE}")
    print("Next: python3 custom_train.py")


if __name__ == "__main__":
    main()
