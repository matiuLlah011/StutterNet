# StutterNet+ — Project Progress

## Overview
**StutterNet+** is a deep learning system for **Urdu stuttering detection** using synthesized speech data. The model classifies speech into three stutter types: syllable repetition, word repetition, and block/pause stuttering.

---

## Phase 1: Data Generation (Abdullah Voice)

### Audio Synthesis
- Generated **135 samples** using ElevenLabs TTS (`eleven_v3` model)
- Voice: **Abdullah** (cloned voice)
- 3 stutter types × 45 samples each:
  - **حرف (Syllable Repetition):** e.g., `ک... ک... کام` — syllable-level repeats
  - **لفظ (Word Repetition):** e.g., `مجھے... مجھے... پروموشن` — full word repeats
  - **بلاک (Block/Pause):** Long pauses mid-sentence simulating speech blocks
- All audio: 16kHz mono WAV, ~7s per sample
- 45 unique real-life scenarios (doctor visit, job interview, shopping, etc.)

### Preprocessing
- Converted WAV → spectrograms (257 × 701 × 1) using STFT
- Normalized to [0, 1] range
- Saved as `.npy` files with PNG visualizations

---

## Phase 2: Initial Model Training

### Architecture — StutterNet+ (FluentNet)
```
SE-ResNet Encoder → BiLSTM → Attention Pooling → Classifier
```
- **SE-ResNet Encoder:** 3 SE-ResBlocks with squeeze-and-excitation channel attention
- **BiLSTM:** Bidirectional LSTM for temporal modeling (hidden=64)
- **Attention Pooling:** Bahdanau-style attention over temporal sequence
- **Classifier:** 2-layer MLP with dropout
- **Parameters:** 722,680 trainable

### Training Configuration
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-3)
- Scheduler: CosineAnnealingLR
- Loss: Focal Loss with inverse-frequency class weights
- Augmentation: SpecAugment (freq/time masking, Gaussian noise, random gain, time reversal)
- Oversampling: Minority class upsampling for balanced training
- Early stopping: patience=20

### Results (145 samples — Abdullah only)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Syllable Repetition | 0.72 | 0.63 | 0.67 |
| Word Repetition | 0.46 | 0.85 | 0.59 |
| Block | 0.50 | 0.12 | 0.20 |
| **Overall Accuracy** | | | **53.8%** |

- Trained for 32 epochs (early stopped)
- Best val loss: 0.2946

---

## Phase 3: Multi-Voice Data Expansion

### New Data Generation
- Added **270 new samples** using two additional cloned voices:
  - **Ibrahim** — 135 samples (45 syllable + 45 word + 45 block)
  - **Mati** — 135 samples (45 syllable + 45 word + 45 block)
- Abdullah voice was **excluded** from new generation
- Block stutters enhanced with longer pauses (`[Silence]` markers) for more realistic blocking
- Same 45 scenarios reused across voices for speaker diversity

### Updated Dataset Totals
| Voice | Samples |
|---|---|
| Abdullah (original) | 145 |
| Ibrahim (new) | 135 |
| Mati (new) | 135 |
| **Total** | **415** |

| Stutter Type | Count |
|---|---|
| Syllable Repetition | 139 |
| Word Repetition | 138 |
| Block | 138 |

---

## Phase 4: Retrained Model

### Results (415 samples — 3 voices)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Syllable Repetition | 0.71 | 0.86 | **0.78** |
| Word Repetition | 0.46 | 0.79 | 0.58 |
| Block | 0.57 | 0.03 | 0.06 |
| **Overall Accuracy** | | | **56.1%** |

- Trained for 75 epochs (early stopped)
- Best val loss: 0.3153

### Comparison: Before vs After
| Metric | 145 samples | 415 samples | Change |
|---|---|---|---|
| Overall Accuracy | 53.8% | 56.1% | +2.3% |
| Syllable Rep F1 | 0.67 | 0.78 | **+0.11** |
| Word Rep F1 | 0.59 | 0.58 | -0.01 |
| Block F1 | 0.20 | 0.06 | -0.14 |

### Key Observations
- **Syllable repetition** improved significantly with multi-voice data
- **Word repetition** remained stable
- **Block detection** remains the hardest class — pauses are subtle in spectrograms and easily confused with other stutter types

---

## Project Structure
```
StutterNet_Data/
├── annotations/          # annotations.json with all sample metadata
├── samples/              # WAV audio files by stutter type
│   ├── syllable_repetition/
│   ├── word_repetition/
│   ├── block/
│   └── clean/
├── spectrograms/         # Preprocessed spectrograms (.npy + .png)
├── checkpoints/          # Model checkpoints (best_model.pt, last_model.pt)
├── evaluation_results/   # Confusion matrix, attention visualizations
├── voice_samples/        # Original voice clone recordings
├── model.py              # StutterNet+ architecture (SE-ResNet + BiLSTM + Attention)
├── dataset.py            # Dataset, augmentation, dataloaders
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation and metrics
├── preprocess.py         # WAV → spectrogram conversion
├── generate_bulk.py      # Bulk TTS generation (Abdullah)
├── generate_multivoice.py # Multi-voice TTS generation (Ibrahim + Mati)
└── progress.md           # This file
```

---

## Next Steps
- Improve block detection (more distinctive pause patterns, higher focal gamma)
- Add clean (non-stuttered) speech samples as a 4th class
- Experiment with more training epochs and hyperparameter tuning
- Add more speakers for better generalization
