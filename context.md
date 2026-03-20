# StutterNet+ — Project Context

## What is this?
A deep learning pipeline for **Urdu stuttering detection**. Given a short audio clip (~7s), the model classifies it into one of three stutter types:
- **حرف (Syllable Repetition)** — repeating syllables: "ک... ک... کام"
- **لفظ (Word Repetition)** — repeating whole words: "مجھے... مجھے... پروموشن"
- **بلاک (Block/Pause)** — long pauses/blocks mid-sentence

## Dataset: 543 samples (~65 min audio)

### Data Sources
| Source | Samples | Type | Notes |
|---|---|---|---|
| Abdullah (TTS) | 145 | ElevenLabs `eleven_v3` | Original voice, 45 scenarios × 3 types |
| Ibrahim (TTS) | 135 | ElevenLabs `eleven_v3` | Same scenarios, different voice |
| Mati (TTS) | 135 | ElevenLabs `eleven_v3` | Same scenarios, different voice |
| Podcast (Real) | 48 | Real human speech | Google Drive podcast clips |
| Zip Dataset (Real) | 80 | Real human speech | Mixed speakers from labeled dataset |

### Class Distribution
| Label | Name | Count |
|---|---|---|
| 1 | syllable_repetition | 203 |
| 2 | word_repetition | 173 |
| 3 | block | 167 |
| 0 | clean | 0 (unused) |

## Model: StutterNet+ (722K params)
```
Input: (B, 1, 257, 701) STFT spectrograms
  → SE-ResNet Encoder (3 SE-ResBlocks: 32→64→128 channels)
  → BiLSTM (128 → 128 bidirectional)
  → Bahdanau Attention Pooling
  → Classifier (128→64→4)
Output: (B, 4) logits
```

## Training Results History

| Phase | Samples | Accuracy | Syl F1 | Word F1 | Block F1 | Macro F1 | Val Loss |
|---|---|---|---|---|---|---|---|
| 2: Abdullah only | 145 | 53.8% | 0.67 | 0.59 | 0.20 | 0.49 | 0.2946 |
| 3: +Ibrahim+Mati | 415 | 56.1% | **0.78** | 0.58 | 0.06 | 0.47 | 0.3153 |
| 4: +Podcast data | 463 | 52.9% | 0.57 | **0.64** | **0.35** | **0.52** | 0.3055 |
| 5: +Zip dataset | 543 | 49.2% | 0.53 | 0.57 | 0.32 | 0.47 | 0.3186 |

### Key Findings
1. **Real podcast data gave the biggest improvement** — block F1: 0.06 → 0.35
2. **Multi-voice TTS improved syllable detection** — F1 peaked at 0.78
3. **Phase 4 (463 samples) is the best overall** — macro F1 = 0.52, most balanced
4. **Zip dataset caused regression** — different acoustic properties led to domain mismatch
5. **Block detection remains the hardest class** across all phases

## Tech Stack
- Python 3, PyTorch (MPS/CUDA/CPU)
- ElevenLabs TTS API for synthetic data generation
- pydub + ffmpeg for audio processing
- STFT spectrograms (257 freq bins × 701 time frames)
- Focal Loss + AdamW + CosineAnnealingLR + Early Stopping
- SpecAugment (freq/time masking, noise, gain, time reversal)

## Potential Next Steps
- Add more real speech data (biggest impact on model quality)
- Add clean (non-stuttered) samples for label 0
- Address domain mismatch between TTS and real speech data
- Try domain adaptation or data augmentation techniques
- Move API keys to environment variables
- Integrate remaining ~26 MP3s from zip "new data" folder
- Consider splitting prolongation into its own class
