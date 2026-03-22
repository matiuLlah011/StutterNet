"""
StutterNet+ Demo — Predict stuttering from any audio file.
Loads the best trained model and classifies the input audio.

Usage: python3 demo.py path/to/audio.wav
"""
import os
import sys

# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
import librosa

from model import StutterNetPlus


# ─── CONSTANTS ─────────────────────────────────────────────────────
TARGET_SR = 16000           # Sample rate expected by model
TARGET_DURATION = 7.0       # Duration in seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 112000 samples

# STFT parameters (must match preprocess.py)
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

# Class names for output
CLASS_NAMES = ["clean", "syllable_repetition", "word_repetition", "block"]
CLASS_DISPLAY = {
    "clean": "CLEAN SPEECH (No Stutter)",
    "syllable_repetition": "STUTTERED — Syllable Repetition (حرف)",
    "word_repetition": "STUTTERED — Word Repetition (لفظ)",
    "block": "STUTTERED — Block/Pause (بلاک)",
}

# Default checkpoint path
DEFAULT_CHECKPOINT = "checkpoints/phase1_attempt2.pt"


# ─── PREPROCESS AUDIO ─────────────────────────────────────────────
def preprocess_audio(wav_path):
    """Load audio file and convert to spectrogram tensor."""
    # Load and resample to 16kHz mono
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # Pad or truncate to exactly 7 seconds
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
    elif len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]

    # Pre-emphasis filter
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # STFT → magnitude → dB → normalize to [0, 1]
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann")
    magnitude = np.abs(stft)
    spec_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    spec_min, spec_max = spec_db.min(), spec_db.max()
    if spec_max - spec_min > 0:
        spec_norm = (spec_db - spec_min) / (spec_max - spec_min)
    else:
        spec_norm = np.zeros_like(spec_db)

    # Shape: (1, 1, 257, 701) — batch, channel, freq, time
    spec_tensor = torch.from_numpy(spec_norm).float().unsqueeze(0).unsqueeze(0)
    return spec_tensor


# ─── LOAD MODEL ───────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    """Load the trained StutterNet+ model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = StutterNetPlus(
        num_classes=config.get("num_classes", 4),
        dropout_rate=config.get("dropout_rate", 0.5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ─── PREDICT ──────────────────────────────────────────────────────
def predict(model, spec_tensor, device):
    """Run model inference and return prediction + confidence."""
    spec_tensor = spec_tensor.to(device)
    with torch.no_grad():
        logits = model(spec_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[pred_class].item() * 100
    return pred_class, confidence


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo.py <audio_file.wav>")
        print("Example: python3 demo.py samples/block/BLOCK_001.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    # Pick best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Find the best checkpoint
    checkpoint = DEFAULT_CHECKPOINT
    if not os.path.exists(checkpoint):
        checkpoint = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint):
        print("ERROR: No model checkpoint found. Run training first.")
        sys.exit(1)

    # Load model
    model = load_model(checkpoint, device)

    # Preprocess audio
    spec = preprocess_audio(audio_path)

    # Predict
    pred_class, confidence = predict(model, spec, device)
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
