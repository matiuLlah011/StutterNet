"""
StutterNet+ — Step 6: Verify all generated files.
Run: python verify.py
"""
import os
import numpy as np
import soundfile as sf

EXPECTED_IDS = {
    "syllable_repetition": ["HARF_001", "HARF_002", "HARF_003", "HARF_004"],
    "word_repetition": ["LAFZ_001", "LAFZ_002", "LAFZ_003"],
    "block": ["BLOCK_001", "BLOCK_002", "BLOCK_003"],
}

EXPECTED_SR = 16000
EXPECTED_DURATION = 7.0
EXPECTED_SHAPE = (257, 701, 1)


def check(condition, label):
    symbol = "\u2713" if condition else "\u2717"
    print(f"  {symbol}  {label}")
    return condition


def main():
    wav_ok = 0
    wav_total = 0
    spec_ok = 0
    spec_total = 0
    all_sr_ok = True
    all_dur_ok = True
    all_shape_ok = True
    no_corrupt = True

    print("=" * 60)
    print("StutterNet+ VERIFICATION REPORT")
    print("=" * 60)

    for subfolder, ids in EXPECTED_IDS.items():
        print(f"\n--- {subfolder.upper()} ---")
        for sid in ids:
            wav_path = f"samples/{subfolder}/{sid}.wav"
            npy_path = f"spectrograms/{subfolder}/{sid}_spectrogram.npy"

            wav_total += 1
            spec_total += 1

            # WAV checks
            print(f"\n[{sid}] WAV: {wav_path}")
            exists = os.path.exists(wav_path)
            check(exists, f"File exists")
            if exists:
                try:
                    data, sr = sf.read(wav_path)
                    channels = 1 if data.ndim == 1 else data.shape[1]
                    duration = len(data) / sr
                    size_kb = os.path.getsize(wav_path) / 1024

                    sr_ok = check(sr == EXPECTED_SR, f"Sample rate = {sr} Hz")
                    ch_ok = check(channels == 1, f"Channels = {channels} (mono)")
                    dur_ok = check(abs(duration - EXPECTED_DURATION) < 0.1,
                                   f"Duration = {duration:.1f} seconds")
                    print(f"        File size: {size_kb:.0f} KB")

                    if not sr_ok: all_sr_ok = False
                    if not dur_ok: all_dur_ok = False
                    if sr_ok and ch_ok and dur_ok:
                        wav_ok += 1
                except Exception as e:
                    print(f"  \u2717  Error reading: {e}")
                    no_corrupt = False
            else:
                all_sr_ok = False
                all_dur_ok = False

            # Spectrogram checks
            print(f"[{sid}] Spectrogram: {npy_path}")
            exists = os.path.exists(npy_path)
            check(exists, f"File exists")
            if exists:
                try:
                    spec = np.load(npy_path)
                    shape_ok = check(spec.shape == EXPECTED_SHAPE,
                                     f"Shape = {spec.shape}")
                    range_ok = check(spec.min() >= 0 and spec.max() <= 1,
                                     f"Values between 0 and 1 (min={spec.min():.4f}, max={spec.max():.4f})")
                    nan_ok = check(not (np.isnan(spec).any() or np.isinf(spec).any()),
                                   f"No NaN or Inf values")

                    if not shape_ok: all_shape_ok = False
                    if not nan_ok: no_corrupt = False
                    if shape_ok and range_ok and nan_ok:
                        spec_ok += 1
                except Exception as e:
                    print(f"  \u2717  Error reading: {e}")
                    no_corrupt = False
                    all_shape_ok = False
            else:
                all_shape_ok = False

    # Summary
    expected_total = sum(len(ids) for ids in EXPECTED_IDS.values())
    ready = (wav_ok == wav_total == expected_total and spec_ok == spec_total == expected_total
             and all_sr_ok and all_dur_ok and all_shape_ok and no_corrupt)

    yn = lambda b: "YES" if b else "NO"
    print("\n" + "=" * 42)
    print("VERIFICATION REPORT")
    print("=" * 42)
    print(f"WAV files generated     : {wav_ok} / {wav_total}")
    print(f"Spectrograms generated  : {spec_ok} / {spec_total}")
    print(f"Sample rates correct    : {yn(all_sr_ok)}")
    print(f"Durations correct       : {yn(all_dur_ok)}")
    print(f"Spectrogram shapes OK   : {yn(all_shape_ok)}")
    print(f"No corrupted files      : {yn(no_corrupt)}")
    print("-" * 42)
    print(f"READY FOR IBRAHIM REVIEW : {yn(ready)}")
    print("=" * 42)


if __name__ == "__main__":
    main()
