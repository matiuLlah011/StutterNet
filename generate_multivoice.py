"""
StutterNet+ — Multi-voice bulk generation: 135 × 2 voices = 270 samples.
Voices: Ibrahim + Mati (NO Abdullah).
Block stutters have enhanced pauses with [Silence] markers.

Run: PATH="$HOME/bin:$PATH" python3 generate_multivoice.py
"""
import requests
import json
import time
import os
import sys
import copy

from generate_bulk import SYLLABLE_SAMPLES, WORD_SAMPLES, BLOCK_SAMPLES

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY = "sk_8dc5f216880280427d534d296e25fcee1788403e46bf1144"
MODEL_ID = "eleven_v3"

VOICES = [
    {"name": "Ibrahim", "voice_id": "JjJY6XmAqwiYILFnZ2Tv", "prefix": "I"},
    {"name": "Mati",    "voice_id": "LRG2Nfqsg6bBGBbp8lMW", "prefix": "M"},
]
# ──────────────────────────────────────────────────────────────────────────────

FOLDER_MAP = {
    "syllable_repetition": "samples/syllable_repetition",
    "word_repetition":     "samples/word_repetition",
    "block":               "samples/block",
}

LABEL_MAP = {
    "syllable_repetition": 1,
    "word_repetition": 2,
    "block": 3,
}

CATEGORY_URDU = {
    "syllable_repetition": "حرف",
    "word_repetition": "لفظ",
    "block": "بلاک",
}


def enhance_block_pauses(block_samples):
    """Add [Silence] and longer pauses to block stutter texts."""
    enhanced = []
    for s in block_samples:
        new_s = copy.deepcopy(s)
        text = new_s["tts_text"]
        annotated = new_s["annotated"]

        # Replace the single "..." pause in block text with much longer pause
        # Pattern in blocks: "...flow text... recovery..."
        # Add extra pauses: "...... [Silence] ...... recovery..."
        text = text.replace("...", "............ ............", 1)

        # Update annotated text with [Silence] marker
        annotated = annotated.replace("[بلاک]", "[بلاک] [Silence]")

        new_s["tts_text"] = text
        new_s["annotated"] = annotated
        enhanced.append(new_s)
    return enhanced


def make_voice_id(original_id, voice_prefix):
    """Create new sample ID: HARF_005 → HARF_I005"""
    parts = original_id.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}_{voice_prefix}{parts[1]}"
    return f"{original_id}_{voice_prefix}"


def get_label_name(sid):
    if "HARF" in sid:
        return "syllable_repetition"
    elif "LAFZ" in sid:
        return "word_repetition"
    elif "BLOCK" in sid:
        return "block"
    return "unknown"


def call_tts_api(text, voice_id):
    """Call ElevenLabs TTS API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.85,
            "style": 0.3,
            "use_speaker_boost": True,
        },
    }
    response = requests.post(url, headers=headers, json=body, timeout=90)
    response.raise_for_status()
    return response.content


def main():
    # Prepare samples: syllable + word stay same, blocks get enhanced pauses
    enhanced_blocks = enhance_block_pauses(BLOCK_SAMPLES)
    base_samples = SYLLABLE_SAMPLES + WORD_SAMPLES + enhanced_blocks

    print("=" * 60)
    print("StutterNet+ — MULTI-VOICE GENERATION")
    print(f"Voices : {', '.join(v['name'] for v in VOICES)}")
    print(f"Model  : {MODEL_ID}")
    print(f"Samples: {len(base_samples)} texts × {len(VOICES)} voices = {len(base_samples) * len(VOICES)} total")
    print("=" * 60)

    for folder in FOLDER_MAP.values():
        os.makedirs(folder, exist_ok=True)

    # Load existing annotations
    ann_path = "annotations/annotations.json"
    existing_samples = []
    if os.path.exists(ann_path):
        with open(ann_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_samples = existing_data.get("samples", [])
    existing_ids = {s["id"] for s in existing_samples}

    # Build generation list: each base sample × each voice
    to_generate = []
    for voice in VOICES:
        for sample in base_samples:
            new_id = make_voice_id(sample["id"], voice["prefix"])
            if new_id not in existing_ids:
                to_generate.append({
                    "sample": sample,
                    "voice": voice,
                    "new_id": new_id,
                })

    skipped = len(base_samples) * len(VOICES) - len(to_generate)
    if skipped > 0:
        print(f"Skipping {skipped} already-generated samples.")
    print(f"Generating {len(to_generate)} new samples...\n")

    if len(to_generate) == 0:
        print("Nothing to generate. All samples already exist.")
        return

    annotation_samples = list(existing_samples)
    total = len(to_generate)
    success = 0
    fail = 0

    for i, item in enumerate(to_generate, 1):
        sample = item["sample"]
        voice = item["voice"]
        sid = item["new_id"]
        label_name = get_label_name(sid)
        folder = FOLDER_MAP[label_name]
        tts_text = sample["tts_text"]
        mp3_path = f"{folder}/{sid}.mp3"
        wav_path = f"{folder}/{sid}.wav"

        print(f"[{i}/{total}] {sid} — {voice['name']} — {sample.get('scenario', '')} ({CATEGORY_URDU.get(label_name, '')})")

        # Try up to 2 times
        audio_bytes = None
        for attempt in range(1, 3):
            try:
                audio_bytes = call_tts_api(tts_text, voice["voice_id"])
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                print(f"  Attempt {attempt} failed (HTTP {status})")
                if attempt < 2:
                    time.sleep(3)
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt < 2:
                    time.sleep(3)

        if audio_bytes is None:
            print(f"  ERROR: Could not generate {sid} — skipping.")
            fail += 1
            continue

        # Save MP3
        with open(mp3_path, "wb") as fout:
            fout.write(audio_bytes)

        # Convert to WAV 16kHz mono 16-bit PCM
        duration = 0.0
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(wav_path, format="wav")
            os.remove(mp3_path)
            duration = len(audio) / 1000.0
            size_kb = os.path.getsize(wav_path) / 1024
            print(f"  OK — {duration:.1f}s — {size_kb:.0f}KB")
            success += 1
        except Exception as e:
            print(f"  WARNING: WAV conversion failed: {e}")
            success += 1  # MP3 still exists

        # Annotation entry
        entry = {
            "id":               sid,
            "wav_file":         wav_path,
            "spectrogram_file": f"spectrograms/{label_name}/{sid}_spectrogram.npy",
            "spectrogram_image": f"spectrograms/{label_name}/{sid}_spectrogram.png",
            "tts_text":         tts_text,
            "annotated":        sample["annotated"],
            "scenario":         sample.get("scenario", ""),
            "stutter_unit":     sample.get("stutter_unit"),
            "category_urdu":    CATEGORY_URDU.get(label_name, ""),
            "label":            LABEL_MAP.get(label_name, -1),
            "label_name":       label_name,
            "language":         "Urdu",
            "duration_seconds": round(duration, 1),
            "voice_used":       voice["name"],
        }
        annotation_samples.append(entry)

        # Save annotations every 10 samples
        if i % 10 == 0:
            _save_annotations(annotation_samples, ann_path)
            print(f"  [checkpoint] Annotations saved ({len(annotation_samples)} total)")

        if i < total:
            time.sleep(1.5)

    # Final save
    _save_annotations(annotation_samples, ann_path)

    print(f"\n{'=' * 60}")
    print(f"MULTI-VOICE GENERATION COMPLETE")
    print(f"  New samples generated: {success}/{total}")
    print(f"  Failed: {fail}")
    print(f"  Total in annotations: {len(annotation_samples)}")
    print(f"  Annotations saved to: {ann_path}")
    print(f"\nNext: PATH=\"$HOME/bin:$PATH\" python3 preprocess.py")
    print(f"{'=' * 60}")


def _save_annotations(samples, path):
    """Save annotations.json with current samples."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from collections import Counter
    label_counts = Counter(s["label"] for s in samples)
    voice_counts = Counter(s.get("voice_used", "unknown") for s in samples)

    annotations = {
        "dataset_info": {
            "name":                   "StutterNet+ Urdu Stuttering Dataset",
            "version":                "2.0 — multi-voice",
            "total_samples":          len(samples),
            "samples_by_type":        dict(label_counts),
            "samples_by_voice":       dict(voice_counts),
            "stutter_types_included": ["syllable_repetition", "word_repetition", "block"],
            "stutter_types_urdu":     {"حرف": "syllable_repetition", "لفظ": "word_repetition", "بلاک": "block"},
            "sample_rate":            16000,
            "target_duration_seconds": 7,
            "spectrogram_shape":      [257, 701, 1],
            "language":               "Urdu",
            "tts_model":              MODEL_ID,
            "voices_used":            {v["name"]: v["voice_id"] for v in VOICES},
            "label_convention":       {"syllable_repetition": 1, "word_repetition": 2, "block": 3, "clean": 0},
        },
        "samples": samples,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
