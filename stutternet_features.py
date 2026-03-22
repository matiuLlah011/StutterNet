"""
StutterNet — Feature Extraction
Extracts 40 MFCC features from each audio file, takes the mean across
frames to get one 40-value vector per file, and applies SMOTE to
balance classes.

Saves: stutternet_features.npz (X_train, X_test, y_train, y_test)

Run: python3 stutternet_features.py
"""
import os
import json
import numpy as np
import librosa
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ─── CONSTANTS ─────────────────────────────────────────────────────
N_MFCC = 40             # Number of MFCC coefficients to extract
TARGET_SR = 16000        # Sample rate for loading audio
TEST_SPLIT = 0.2         # 20% for testing
RANDOM_SEED = 42
OUTPUT_FILE = "stutternet_features.npz"

# Map original labels (1,2,3) to StutterNet labels (0,1,2)
# Since we have no clean samples, we remap to 0-indexed classes
LABEL_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["syllable_repetition", "word_repetition", "block"]


def extract_mfcc(wav_path, n_mfcc=N_MFCC, sr=TARGET_SR):
    """
    Extract MFCC features from an audio file.
    Returns the mean of 40 MFCCs across all frames → one 40-dim vector.
    """
    # Load audio at 16kHz mono
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # Extract 40 MFCC coefficients (shape: n_mfcc × num_frames)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Take mean across frames → single 40-dimensional vector
    mfcc_mean = np.mean(mfccs, axis=1)

    return mfcc_mean


def main():
    print("=" * 60)
    print("STUTTERNET — FEATURE EXTRACTION")
    print("=" * 60)

    # Load sample annotations
    with open("annotations/annotations.json") as f:
        data = json.load(f)
    samples = data["samples"]

    print(f"Total samples in annotations: {len(samples)}")

    # Extract MFCC features from each audio file
    features = []  # list of 40-dim vectors
    labels = []    # corresponding class labels
    skipped = 0

    for i, s in enumerate(samples):
        wav_path = s["wav_file"]
        original_label = s["label"]

        # Skip if label not in our mapping (e.g., clean=0 which has no samples)
        if original_label not in LABEL_MAP:
            skipped += 1
            continue

        # Skip if WAV file missing
        if not os.path.exists(wav_path):
            skipped += 1
            continue

        try:
            mfcc_vec = extract_mfcc(wav_path)
            features.append(mfcc_vec)
            labels.append(LABEL_MAP[original_label])
        except Exception as e:
            print(f"  WARNING: Failed on {s['id']}: {e}")
            skipped += 1
            continue

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} files...")

    X = np.array(features)  # shape: (num_samples, 40)
    y = np.array(labels)    # shape: (num_samples,)

    print(f"\nFeature extraction complete:")
    print(f"  Samples processed: {len(X)}")
    print(f"  Skipped: {skipped}")
    print(f"  Feature shape: {X.shape}")

    # Show class distribution before SMOTE
    counts_before = Counter(y)
    print(f"\nClass distribution BEFORE SMOTE:")
    for cls in sorted(counts_before.keys()):
        print(f"  {CLASS_NAMES[cls]}: {counts_before[cls]}")

    # Split into train/test BEFORE applying SMOTE
    # (SMOTE only on training data to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Apply SMOTE to balance training classes
    print(f"\nApplying SMOTE to balance training data...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    counts_after = Counter(y_train_balanced)
    print(f"Class distribution AFTER SMOTE:")
    for cls in sorted(counts_after.keys()):
        print(f"  {CLASS_NAMES[cls]}: {counts_after[cls]}")
    print(f"  Total training samples: {len(X_train_balanced)} (was {len(X_train)})")

    # Save features and labels to file
    np.savez(
        OUTPUT_FILE,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_test=X_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
    )
    print(f"\nFeatures saved to: {OUTPUT_FILE}")
    print("Next: python3 stutternet_train.py")


if __name__ == "__main__":
    main()
