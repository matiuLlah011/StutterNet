"""
Custom Demo — Predict stutter type from any audio file.
Uses the custom 1D CNN + BiGRU + Attention model.

Usage: python3 custom_demo.py path/to/audio.wav
"""
import os
import sys

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
import librosa

from custom_model import CustomStutterDetector


# ─── CONSTANTS ─────────────────────────────────────────────────────
N_MFCC = 40
TARGET_SR = 16000
HOP_LENGTH = 512
MAX_FRAMES = 220
CHECKPOINT = "checkpoints/custom_best.pt"

CLASS_NAMES = ["syllable_repetition", "word_repetition", "block"]
CLASS_DISPLAY = {
    "syllable_repetition": "STUTTERED — Syllable Repetition (حرف)",
    "word_repetition":     "STUTTERED — Word Repetition (لفظ)",
    "block":               "STUTTERED — Block/Pause (بلاک)",
}


def extract_features(wav_path):
    """Extract MFCC + delta + delta-delta features from audio file."""
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

    return features


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 custom_demo.py <audio_file.wav>")
        print("Example: python3 custom_demo.py samples/block/BLOCK_001.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Model not found at {CHECKPOINT}. Run custom_train.py first.")
        sys.exit(1)

    model = CustomStutterDetector(input_features=120, num_classes=3)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Extract features and predict
    features = extract_features(audio_path)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[pred_class].item() * 100

    pred_name = CLASS_NAMES[pred_class]
    display = CLASS_DISPLAY[pred_name]

    print()
    print("=" * 50)
    print(f"  File:       {audio_path}")
    print(f"  Result:     {display}")
    print(f"  Confidence: {confidence:.1f}%")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
