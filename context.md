# StutterNet+ — Project Context

## What is this?
A deep learning pipeline for **Urdu stuttering detection**. Given a short audio clip (~7s), the model classifies it into one of three stutter types:
- **حرف (Syllable Repetition)** — repeating syllables: "ک... ک... کام"
- **لفظ (Word Repetition)** — repeating whole words: "مجھے... مجھے... پروموشن"
- **بلاک (Block/Pause)** — long pauses/blocks mid-sentence

## Dataset: 415 ElevenLabs samples (currently used for training)

### Data Sources
| Source | Samples | Type | Notes | Used |
|---|---|---|---|---|
| Abdullah (TTS) | 145 | ElevenLabs `eleven_v3` | Original voice, 45 scenarios × 3 types | Yes |
| Ibrahim (TTS) | 135 | ElevenLabs `eleven_v3` | Same scenarios, different voice | Yes |
| Mati (TTS) | 135 | ElevenLabs `eleven_v3` | Same scenarios, different voice | Yes |
| Podcast (Real) | 48 | Real human speech | Google Drive podcast clips | No (domain mismatch) |
| Zip Dataset (Real) | 80 | Real human speech | Mixed speakers from labeled dataset | No (domain mismatch) |

### Class Distribution (ElevenLabs only — 415 samples)
| Label | Name | Count |
|---|---|---|
| 1 | syllable_repetition | 139 |
| 2 | word_repetition | 138 |
| 3 | block | 138 |
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

| Phase | Samples | Data | Accuracy | Syl F1 | Word F1 | Block F1 | Macro F1 | Val Loss |
|---|---|---|---|---|---|---|---|---|
| 2: Abdullah only | 145 | TTS | 53.8% | 0.67 | 0.59 | 0.20 | 0.49 | 0.2946 |
| 3: +Ibrahim+Mati | 415 | TTS | 56.1% | 0.78 | 0.58 | 0.06 | 0.47 | 0.3153 |
| 4: +Podcast data | 463 | TTS+Real | 52.9% | 0.57 | 0.64 | 0.35 | 0.52 | 0.3055 |
| 5: +Zip dataset | 543 | TTS+Real | 49.2% | 0.53 | 0.57 | 0.32 | 0.47 | 0.3186 |
| **6: ElevenLabs 90/10** | **415** | **TTS only** | **61.9%** | **0.78** | **0.61** | **0.40** | **0.60** | **0.2506** |

### Key Findings
1. **ElevenLabs-only with 90/10 split is the best** — 61.9% accuracy, macro F1 = 0.60
2. **Domain consistency > data volume** — 415 clean TTS samples beat 543 mixed samples
3. **Near-perfect class balance** (139/138/138) in ElevenLabs data helps training
4. **Block detection improved** (F1: 0.32 → 0.40) but remains the weakest class
5. **Syllable repetition** is the most reliable class (F1=0.78)

## Tech Stack
- Python 3, PyTorch (MPS/CUDA/CPU)
- ElevenLabs TTS API for synthetic data generation
- pydub + ffmpeg for audio processing
- STFT spectrograms (257 freq bins × 701 time frames)
- Focal Loss + AdamW + CosineAnnealingLR + Early Stopping
- SpecAugment (freq/time masking, noise, gain, time reversal)

## Current Training Config
- **Data**: ElevenLabs TTS only (415 samples), `elevenlabs_only=True` in train.py
- **Split**: 90/10 stratified (376 train / 39 test), `val_split=0.1`
- **Training**: Focal Loss, AdamW, lr=1e-3, CosineAnnealingLR, patience=20

## Potential Next Steps
- Add more real speech data (biggest impact on model quality)
- Add clean (non-stuttered) samples for label 0
- Address domain mismatch between TTS and real speech data
- Try domain adaptation or data augmentation techniques
- Move API keys to environment variables
- Integrate remaining ~26 MP3s from zip "new data" folder
- Consider splitting prolongation into its own class
