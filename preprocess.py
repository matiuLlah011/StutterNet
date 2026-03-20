"""
StutterNet+ — Step 4: Preprocess WAV files into spectrograms.
Run: python preprocess.py
"""
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_SR = 16000
TARGET_DURATION = 7.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 112000

# STFT parameters
N_FFT = 512
HOP_LENGTH = 160      # 10ms at 16kHz
WIN_LENGTH = 400      # 25ms at 16kHz
WINDOW = "hann"

SAMPLE_DIRS = {
    "syllable_repetition": "samples/syllable_repetition",
    "word_repetition": "samples/word_repetition",
    "block": "samples/block",
    "clean": "samples/clean",
}


def preprocess_file(wav_path, output_dir, sample_id):
    """Load WAV, compute spectrogram, save .npy and .png."""
    # Step A — Load and standardize
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
    elif len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]

    # Step B — Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Step C — STFT
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window=WINDOW)

    # Step D — Magnitude → dB → normalize
    magnitude = np.abs(stft)
    spec_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    spec_min, spec_max = spec_db.min(), spec_db.max()
    if spec_max - spec_min > 0:
        spec_norm = (spec_db - spec_min) / (spec_max - spec_min)
    else:
        spec_norm = np.zeros_like(spec_db)

    # Add channel dimension: (257, 701, 1)
    spec_final = spec_norm[:, :, np.newaxis]

    # Step E — Save
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"{sample_id}_spectrogram.npy")
    png_path = os.path.join(output_dir, f"{sample_id}_spectrogram.png")

    np.save(npy_path, spec_final)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.imshow(spec_norm, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(png_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Step F — Report
    print(f"Processed {sample_id}: ({len(y)},) → spectrogram {spec_final.shape}")
    return spec_final.shape


def main():
    print("=" * 60)
    print("StutterNet+ Preprocessing Pipeline")
    print("=" * 60)

    total = 0
    for subfolder, sample_dir in SAMPLE_DIRS.items():
        output_dir = f"spectrograms/{subfolder}"
        if not os.path.isdir(sample_dir):
            print(f"WARNING: {sample_dir} not found — skipping.")
            continue

        wav_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".wav")])
        if not wav_files:
            print(f"No WAV files in {sample_dir} — skipping.")
            continue

        print(f"\nProcessing {subfolder} ({len(wav_files)} files):")
        for wav_file in wav_files:
            sample_id = wav_file.replace(".wav", "")
            wav_path = os.path.join(sample_dir, wav_file)
            preprocess_file(wav_path, output_dir, sample_id)
            total += 1

    print(f"\nDone. Processed {total} files total.")
    print("Next step: run  python verify.py")


if __name__ == "__main__":
    main()
