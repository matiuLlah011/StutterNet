# StutterNet+ — Claude Code Instructions

## Project Overview
StutterNet+ is a deep learning system for **Urdu stuttering detection** using synthesized (ElevenLabs TTS) and real speech data. It classifies speech into three stutter types plus clean speech.

## Critical Rules
- **NEVER commit or push API keys.** The ElevenLabs API key in `generate_bulk.py` and `generate_multivoice.py` is sensitive.
- **Always use `PATH="$HOME/bin:$PATH"`** before running commands that need ffmpeg/ffprobe (e.g., TTS generation, audio conversion with pydub).
- **Large files are gitignored**: WAV, MP3, .npy spectrograms, .pt checkpoints. Never try to `git add` these.
- **Labels start at 1** (no clean/label-0 samples exist yet). The model has `num_classes=4` but class 0 is unused.

## Pipeline Commands
```bash
# Generate TTS data (only if adding new samples)
PATH="$HOME/bin:$PATH" python3 generate_bulk.py          # Abdullah voice
PATH="$HOME/bin:$PATH" python3 generate_multivoice.py    # Ibrahim + Mati voices

# Full training pipeline
python3 preprocess.py    # WAV → spectrograms (257×701×1)
python3 train.py         # Train model (~5-10 min on MPS)
python3 evaluate.py      # Metrics, confusion matrix, attention plots
```

## Key Architecture
- **Model**: SE-ResNet → BiLSTM → Bahdanau Attention → Classifier (722K params)
- **Input**: (B, 1, 257, 701) STFT spectrograms from 7s audio at 16kHz
- **Loss**: Focal Loss with inverse-frequency class weights
- **Optimizer**: AdamW + CosineAnnealingLR + early stopping (patience=20)

## Label Convention
| Label | Name | Urdu | Sample ID Patterns |
|---|---|---|---|
| 0 | clean | صاف | (no samples yet) |
| 1 | syllable_repetition | حرف | HARF_*, PODCAST_Phenome*, ZIPAUDIO_*_Sy, ZIPNEW_حرف_* |
| 2 | word_repetition | لفظ | LAFZ_*, ZIPAUDIO_*_w, ZIPNEW_لفظ_* |
| 3 | block | بلاک | BLOCK_*, PODCAST_Pause*, PODCAST_Pronglation*, ZIPAUDIO_*_p, ZIPNEW_بلاک_* |

## Voice Sources
| Voice | Prefix/Pattern | Type | Samples |
|---|---|---|---|
| Abdullah | HARF_###, LAFZ_###, BLOCK_### | TTS | 145 |
| Ibrahim | *_I### | TTS | 135 |
| Mati | *_M### | TTS | 135 |
| Podcast | PODCAST_* | Real speech | 48 |
| Zip Dataset | ZIPAUDIO_*, ZIPNEW_* | Real speech | 80 |

## File Map
| File | Purpose |
|---|---|
| `model.py` | StutterNet+ architecture definition |
| `dataset.py` | Dataset class, SpecAugment augmentation, stratified dataloaders with oversampling |
| `train.py` | Training loop (Focal Loss, AdamW, CosineAnnealingLR, early stopping) |
| `evaluate.py` | Per-class metrics, confusion matrix, attention visualizations |
| `preprocess.py` | WAV → STFT spectrogram conversion (pad/truncate to 7s) |
| `generate_bulk.py` | ElevenLabs TTS — Abdullah voice (135 samples). Contains API key. |
| `generate_multivoice.py` | ElevenLabs TTS — Ibrahim + Mati voices (270 samples) |
| `annotations/annotations.json` | Master dataset file — all sample metadata, labels, paths |
| `cloned_voices.json` | Voice name → ElevenLabs voice ID mapping |
| `progress.md` | Full project history with all training results |
| `context.md` | Quick-reference context for the project |

## Data Directories
- `samples/{syllable_repetition,word_repetition,block,clean}/` — WAV files (16kHz mono)
- `spectrograms/{syllable_repetition,word_repetition,block,clean}/` — .npy + .png
- `checkpoints/` — best_model.pt, last_model.pt
- `evaluation_results/` — confusion_matrix.png, attention plots
- `gdrive_data/` — Raw downloads from Google Drive (podcast + zip dataset)

## Current Best Results (Phase 4 — 463 samples)
- Accuracy: 52.9%, Macro F1: 0.52
- Best at balanced performance across all stutter types
- Phase 5 (543 samples with zip data) regressed slightly (49.2%, macro F1: 0.47) due to acoustic domain mismatch

## Known Issues
- API key hardcoded in generation scripts — move to env var
- No clean (non-stuttered) samples yet
- Block detection remains the weakest class
- Zip dataset has different acoustic properties causing domain mismatch
- ~26 additional MP3s in zip "new data" folder not yet integrated (Urdu-named files beyond CSV coverage)
