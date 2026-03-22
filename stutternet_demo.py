"""
StutterNet Demo — Predict stutter type from any audio file.
Uses the trained StutterNet model (MFCC + BiLSTM + RNN).

Usage: python3 stutternet_demo.py path/to/audio.wav
"""
import os
import sys

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
import librosa

from stutternet_model import StutterNet


# ─── CONSTANTS ─────────────────────────────────────────────────────
N_MFCC = 40
TARGET_SR = 16000
CHECKPOINT = "checkpoints/stutternet_best.pt"

# Class names and display strings
CLASS_NAMES = ["syllable_repetition", "word_repetition", "block"]
CLASS_DISPLAY = {
    "syllable_repetition": "STUTTERED — Syllable Repetition (حرف)",
    "word_repetition":     "STUTTERED — Word Repetition (لفظ)",
    "block":               "STUTTERED — Block/Pause (بلاک)",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 stutternet_demo.py <audio_file.wav>")
        print("Example: python3 stutternet_demo.py samples/block/BLOCK_001.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    # Pick device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Model not found at {CHECKPOINT}. Run stutternet_train.py first.")
        sys.exit(1)

    model = StutterNet(input_dim=40, num_classes=3)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Extract 40 MFCC features and take mean across frames
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfccs, axis=1)

    # Predict
    x = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[pred_class].item() * 100

    pred_name = CLASS_NAMES[pred_class]
    display = CLASS_DISPLAY[pred_name]

    # Print result
    print()
    print("=" * 50)
    print(f"  File:       {audio_path}")
    print(f"  Result:     {display}")
    print(f"  Confidence: {confidence:.1f}%")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
