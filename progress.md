# StutterNet+ — Project Progress

## Overview
**StutterNet+** is a deep learning system for **Urdu stuttering detection** using synthesized + real speech data. The model classifies speech into three stutter types: syllable repetition (حرف), word repetition (لفظ), and block/pause stuttering (بلاک).

**Current state:** 543 samples (~65 min audio), 3 TTS voices + real podcast + zip dataset, best macro F1 = 0.52 (Phase 4, 463 samples).

---

## Quick Context for New Sessions

### How to run the full pipeline
```bash
# 1. Generate new TTS samples (if needed)
PATH="$HOME/bin:$PATH" python3 generate_bulk.py          # Abdullah voice (135 samples)
PATH="$HOME/bin:$PATH" python3 generate_multivoice.py    # Ibrahim + Mati voices (270 samples)

# 2. Preprocess: WAV → spectrograms
python3 preprocess.py

# 3. Train the model
python3 train.py

# 4. Evaluate
python3 evaluate.py
```

### Key files
| File | Purpose |
|---|---|
| `model.py` | StutterNet+ architecture (SE-ResNet + BiLSTM + Attention, 722K params) |
| `dataset.py` | Dataset class, SpecAugment, dataloaders with stratified split + oversampling |
| `train.py` | Training loop with Focal Loss, AdamW, CosineAnnealingLR, early stopping |
| `evaluate.py` | Full evaluation: per-class metrics, confusion matrix, attention plots |
| `preprocess.py` | WAV → STFT spectrogram (257×701×1) normalized to [0,1] |
| `generate_bulk.py` | ElevenLabs TTS generation for Abdullah voice (135 samples, contains API key) |
| `generate_multivoice.py` | ElevenLabs TTS generation for Ibrahim + Mati voices (270 samples) |
| `annotations/annotations.json` | Master dataset file — all sample metadata, labels, file paths |
| `cloned_voices.json` | Voice IDs: Mati=LRG2Nfqsg6bBGBbp8lMW, Ibrahim=JjJY6XmAqwiYILFnZ2Tv, Abdullah=A4ASMsAwv9E88KBNEJJy |

### Label convention
| Label | Name | Urdu | ID Prefix |
|---|---|---|---|
| 0 | clean | صاف | (no samples yet) |
| 1 | syllable_repetition | حرف | HARF_*, PODCAST_Phenome*, ZIPAUDIO_*_Sy, ZIPNEW_حرف_* |
| 2 | word_repetition | لفظ | LAFZ_*, ZIPAUDIO_*_w, ZIPNEW_لفظ_* |
| 3 | block | بلاک | BLOCK_*, PODCAST_Pause*, PODCAST_Pronglation*, ZIPAUDIO_*_p, ZIPNEW_بلاک_* |

### Voice ID suffixes in sample names
- No suffix (e.g., `HARF_005`) = Abdullah voice
- `_I` suffix (e.g., `HARF_I005`) = Ibrahim voice
- `_M` suffix (e.g., `HARF_M005`) = Mati voice
- `PODCAST_` prefix = Real speech from podcast recordings
- `ZIPAUDIO_` prefix = Real speech from zip dataset (Audios folder)
- `ZIPNEW_` prefix = Real speech from zip dataset (new data folder)

### ElevenLabs TTS config
- API key is in `generate_bulk.py` and `generate_multivoice.py` (should be moved to env var)
- Model: `eleven_v3`
- Voice settings: stability=0.4, similarity_boost=0.85, style=0.3, speaker_boost=True
- Output: MP3 → converted to 16kHz mono 16-bit WAV

### Model architecture
```
Input: (B, 1, 257, 701) spectrograms
  → SE-ResNet Encoder (3 SE-ResBlocks: 32→64→128→128 channels)
  → BiLSTM (input=128, hidden=64, bidirectional → 128)
  → Bahdanau Attention Pooling (128→64→1)
  → Classifier (Dropout→Linear 128→64→ReLU→Dropout→Linear 64→4)
Output: (B, 4) logits
```

### Training config (TrainConfig in train.py)
- epochs=100, batch_size=4, lr=1e-3, weight_decay=1e-3
- val_split=0.2, patience=20, focal_gamma=2.0, dropout=0.5
- Device auto-detected: MPS (Apple Silicon) > CUDA > CPU

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

### Results (145 samples — Abdullah only)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Syllable Repetition | 0.72 | 0.63 | 0.67 |
| Word Repetition | 0.46 | 0.85 | 0.59 |
| Block | 0.50 | 0.12 | 0.20 |
| **Overall Accuracy** | | | **53.8%** |

- Trained for 32 epochs (early stopped), best val loss: 0.2946
- Block detection was very weak (F1=0.20)

---

## Phase 3: Multi-Voice Data Expansion

### New Data Generation
- Added **270 new samples** using two additional cloned voices:
  - **Ibrahim** — 135 samples (45 syllable + 45 word + 45 block)
  - **Mati** — 135 samples (45 syllable + 45 word + 45 block)
- Abdullah voice was **excluded** from new generation
- Block stutters enhanced with longer pauses (`[Silence]` markers) for more realistic blocking
- Same 45 scenarios reused across voices for speaker diversity

### Results (415 samples — 3 TTS voices)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Syllable Repetition | 0.71 | 0.86 | **0.78** |
| Word Repetition | 0.46 | 0.79 | 0.58 |
| Block | 0.57 | 0.03 | 0.06 |
| **Overall Accuracy** | | | **56.1%** |

- Trained for 75 epochs (early stopped), best val loss: 0.3153
- Syllable rep improved significantly, but block detection collapsed

---

## Phase 4: Real Podcast Data Integration

### Data Source
- Downloaded **48 real stuttering recordings** from Google Drive podcast collection
- Source: `https://drive.google.com/drive/folders/1jpxY20oReVYuJD1YKTG5IujBBMzQvXyp`
- These are **real human speech** clips (not TTS) — much more valuable for training
- Additional zip file at `https://drive.google.com/file/d/1UWVAV8po7-j4PIaNyFmQHz8BrQNvxfxO/view` could not be downloaded (permission issue — needs "Anyone with the link" sharing)

### Downloaded Data Breakdown
| Category | Files | Duration | Mapped To |
|---|---|---|---|
| Pause/Pasue (block) | 15 | 1.8 min | block (label 3) |
| Phenome (syllable rep) | 28 | 3.3 min | syllable_repetition (label 1) |
| Prolongation | 5 | 0.5 min | block (label 3) |
| **Total** | **48** | **5.6 min** | |

- Files saved to `gdrive_data/Stutter Podcast Data/`
- Converted to WAV and added to `samples/` folders with `PODCAST_` prefix IDs

### Results (463 samples — 3 TTS voices + real podcast)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Syllable Repetition | 0.63 | 0.51 | 0.57 |
| Word Repetition | 0.54 | 0.79 | **0.64** |
| Block | 0.40 | 0.32 | **0.35** |
| **Overall Accuracy** | | | **52.9%** |
| **Macro F1** | | | **0.52** |

- Trained for 41 epochs (early stopped), best val loss: 0.3055

---

## Phase 5: Zip Dataset Integration

### Data Source
- Downloaded zip file from Google Drive: `https://drive.google.com/file/d/1UWVAV8po7-j4PIaNyFmQHz8BrQNvxfxO/view`
- Contains **80 usable samples** from multiple real speakers

### Zip Dataset Breakdown
| Source Folder | Files | Labeling Method | Mapped To |
|---|---|---|---|
| Audios/ (suffix -Sy) | 10 | Filename suffix | syllable_repetition (label 1) |
| Audios/ (suffix -w) | 27 | Filename suffix | word_repetition (label 2) |
| Audios/ (suffix -p) | 24 | Filename suffix | block (label 3) |
| new data/ (CSV) | 19 | annotations.csv | Mixed types |
| **Total** | **80** | | |

- Voice attribution: ZipData_V1 (7), ZipData_V2 (6), ZipData_V3 (6), ZipData_Real (61)
- Files converted to 16kHz mono WAV and added to `samples/` with `ZIPAUDIO_` and `ZIPNEW_` prefixes

### Results (543 samples — 3 TTS + podcast + zip)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Syllable Repetition | 0.63 | 0.46 | 0.53 |
| Word Repetition | 0.46 | 0.74 | 0.57 |
| Block | 0.39 | 0.28 | 0.32 |
| **Overall Accuracy** | | | **49.2%** |
| **Macro F1** | | | **0.47** |

- Trained for 58 epochs (early stopped), best val loss: 0.3186
- **Regression from Phase 4** — zip data has different acoustic properties (domain mismatch)
- ~26 additional MP3s in zip "new data" folder remain unintegrated

---

## Full Results Comparison

| Metric | Phase 2 (145) | Phase 3 (415) | Phase 4 (463) | Phase 5 (543) | Best |
|---|---|---|---|---|---|
| Overall Accuracy | 53.8% | **56.1%** | 52.9% | 49.2% | Phase 3 |
| Syllable Rep F1 | 0.67 | **0.78** | 0.57 | 0.53 | Phase 3 |
| Word Rep F1 | 0.59 | 0.58 | **0.64** | 0.57 | Phase 4 |
| Block F1 | 0.20 | 0.06 | **0.35** | 0.32 | Phase 4 |
| **Macro F1** | 0.49 | 0.47 | **0.52** | 0.47 | Phase 4 |
| Val Loss | **0.2946** | 0.3153 | 0.3055 | 0.3186 | Phase 2 |

### Key Takeaways
1. **Real podcast data was the biggest improvement** — block F1 went from 0.06 → 0.35
2. **Multi-voice TTS helped syllable detection** (F1=0.78 in Phase 3)
3. **Phase 4 has best macro F1** (0.52) — most balanced across all classes
4. **Zip data caused regression** — acoustic domain mismatch hurt overall performance
5. Domain adaptation or better preprocessing may be needed for heterogeneous data sources
6. More real speech data with consistent recording quality is the most impactful next step

---

## Current Dataset Summary

### Total: 543 samples (~65 minutes of audio)

| Voice | Samples | Type |
|---|---|---|
| Abdullah | 145 | TTS (ElevenLabs) |
| Ibrahim | 135 | TTS (ElevenLabs) |
| Mati | 135 | TTS (ElevenLabs) |
| Podcast (Real) | 48 | Real human speech |
| Zip Dataset (Real) | 80 | Real human speech (mixed speakers) |

| Stutter Type | Count |
|---|---|
| Syllable Repetition | 203 |
| Word Repetition | 173 |
| Block | 167 |
| Clean | 0 |

---

## Project Structure
```
StutterNet_Data/
├── annotations/              # annotations.json — master dataset metadata
├── samples/                  # WAV audio files (16kHz mono)
│   ├── syllable_repetition/  # HARF_*, HARF_I*, HARF_M*, PODCAST_Phenome*
│   ├── word_repetition/      # LAFZ_*, LAFZ_I*, LAFZ_M*
│   ├── block/                # BLOCK_*, BLOCK_I*, BLOCK_M*, PODCAST_Pause*, PODCAST_Pronglation*
│   └── clean/                # (empty — no clean samples yet)
├── spectrograms/             # Preprocessed spectrograms (.npy 257×701×1 + .png)
├── checkpoints/              # best_model.pt, last_model.pt
├── evaluation_results/       # confusion_matrix.png, attention plots
├── gdrive_data/              # Raw downloads from Google Drive
│   ├── Stutter Podcast Data/ # 48 real podcast MP3s
│   └── zipdata/Dataset/      # Zip dataset (Audios/, new data/, Transcriptions/)
├── voice_samples/            # Original voice clone recordings (Abdullah, Ibrahim, Mati)
├── model.py                  # StutterNet+ architecture
├── dataset.py                # Dataset, SpecAugment, dataloaders
├── train.py                  # Training pipeline (Focal Loss, AdamW, early stopping)
├── evaluate.py               # Evaluation metrics + visualizations
├── preprocess.py             # WAV → spectrogram pipeline
├── generate_bulk.py          # TTS generation — Abdullah (135 samples)
├── generate_multivoice.py    # TTS generation — Ibrahim + Mati (270 samples)
├── generate_audio.py         # Single sample TTS generation utility
├── generate_test.py          # Test generation script
├── list_voices.py            # List available ElevenLabs voices
├── test_voices.py            # Test voice clones
├── verify.py                 # Verify dataset integrity
├── cloned_voices.json        # Voice name → ElevenLabs voice ID mapping
├── .gitignore                # Excludes WAV, .npy, .pt files (too large for git)
├── progress.md               # Full project history and training results
├── context.md                # Quick-reference project context
└── CLAUDE.md                 # Claude Code project instructions
```

---

## Known Issues & Next Steps
- **ElevenLabs API key** is hardcoded in generate_bulk.py and generate_multivoice.py — should be moved to environment variable
- **No clean (non-stuttered) samples** — model only has 3 classes, adding class 0 would improve real-world usage
- **Block detection** is the weakest class — more real block/pause data would help most
- **Domain mismatch** — zip dataset's different acoustic properties caused regression; may need domain adaptation
- **~26 unintegrated MP3s** in zip "new data" folder (Urdu-named files beyond CSV coverage: بلاک_*, حرف_*, لفظ_*)
- **Prolongation** is currently merged into block — could be split into its own class if enough data
- Large files (WAV, spectrograms, checkpoints) are in `.gitignore` — not in git repo
- **GitHub repo:** https://github.com/matiuLlah011/StutterNet-
