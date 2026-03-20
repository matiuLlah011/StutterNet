"""
StutterNet+ — Generate Urdu stuttered speech dataset using ElevenLabs TTS.
Model: eleven_v3 | Voice: Abdullah (A4ASMsAwv9E88KBNEJJy)

3 categories: حرف (syllable), لفظ (word), بلاک (block)
Run: PATH="$HOME/bin:$PATH" python3 generate_audio.py
"""
import requests
import json
import time
import os
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY = "sk_8dc5f216880280427d534d296e25fcee1788403e46bf1144"
VOICE_ID = "A4ASMsAwv9E88KBNEJJy"   # Abdullah (cloned male voice)
VOICE_NAME = "Abdullah"
MODEL_ID = "eleven_v3"               # Latest, best quality — supports Urdu
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# حرف — Syllable-level stutter (4 samples)
# Pacing: natural flow ... syllable... syllable... full-word ... natural flow
# ──────────────────────────────────────────────────────────────────────────────
SYLLABLE_SAMPLES = [
    {
        "id": "HARF_001",
        "scenario": "Job Interview",
        "tts_text": "میں نے پچھلے پانچ سال مارکیٹنگ میں... ک... ک... کام کیا ہے اور مجھے اس فیلڈ میں کافی تجربہ ہے",
        "annotated": "میں نے پچھلے پانچ سال مارکیٹنگ میں [حرف] ک... ک... [/حرف] کام کیا ہے اور مجھے اس فیلڈ میں کافی تجربہ ہے",
        "stutter_unit": "ک",
        "label": 1,
        "label_name": "syllable_repetition",
    },
    {
        "id": "HARF_002",
        "scenario": "Office Presentation",
        "tts_text": "اس پروجیکٹ کی ڈیڈ لائن اگلے ہفتے ہے اور ہمیں... ت... ت... تیاری مکمل کرنی ہوگی جلد از جلد",
        "annotated": "اس پروجیکٹ کی ڈیڈ لائن اگلے ہفتے ہے اور ہمیں [حرف] ت... ت... [/حرف] تیاری مکمل کرنی ہوگی جلد از جلد",
        "stutter_unit": "ت",
        "label": 1,
        "label_name": "syllable_repetition",
    },
    {
        "id": "HARF_003",
        "scenario": "Phone Call to Family",
        "tts_text": "امی مجھے آج رات کو تھوڑی دیر ہو جائے گی کیونکہ... م... م... میٹنگ ابھی ختم نہیں ہوئی",
        "annotated": "امی مجھے آج رات کو تھوڑی دیر ہو جائے گی کیونکہ [حرف] م... م... [/حرف] میٹنگ ابھی ختم نہیں ہوئی",
        "stutter_unit": "م",
        "label": 1,
        "label_name": "syllable_repetition",
    },
    {
        "id": "HARF_004",
        "scenario": "Asking Directions",
        "tts_text": "بھائی صاحب یہ سڑک کہاں جاتی ہے مجھے... س... س... سٹیشن جانا ہے کیا یہ راستہ صحیح ہے",
        "annotated": "بھائی صاحب یہ سڑک کہاں جاتی ہے مجھے [حرف] س... س... [/حرف] سٹیشن جانا ہے کیا یہ راستہ صحیح ہے",
        "stutter_unit": "س",
        "label": 1,
        "label_name": "syllable_repetition",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# لفظ — Word-level stutter (3 samples)
# Pacing: natural flow ... word... word ... natural flow
# ──────────────────────────────────────────────────────────────────────────────
WORD_SAMPLES = [
    {
        "id": "LAFZ_001",
        "scenario": "Ordering Food",
        "tts_text": "مجھے ایک پلیٹ بریانی دے دیں اور ساتھ میں... رائتہ... رائتہ... بھی لگا دیں پلیز",
        "annotated": "مجھے ایک پلیٹ بریانی دے دیں اور ساتھ میں [لفظ] رائتہ... رائتہ... [/لفظ] بھی لگا دیں پلیز",
        "stutter_unit": "رائتہ",
        "label": 2,
        "label_name": "word_repetition",
    },
    {
        "id": "LAFZ_002",
        "scenario": "Talking to Teacher",
        "tts_text": "سر میں نے اسائنمنٹ مکمل کر لی ہے لیکن... مجھے... مجھے... ایک سوال پوچھنا تھا اس کے بارے میں",
        "annotated": "سر میں نے اسائنمنٹ مکمل کر لی ہے لیکن [لفظ] مجھے... مجھے... [/لفظ] ایک سوال پوچھنا تھا اس کے بارے میں",
        "stutter_unit": "مجھے",
        "label": 2,
        "label_name": "word_repetition",
    },
    {
        "id": "LAFZ_003",
        "scenario": "Shopping",
        "tts_text": "یہ شرٹ کتنے کی ہے اور کیا اس میں... رنگ... رنگ... نیلا بھی ملتا ہے یا صرف یہی ہے",
        "annotated": "یہ شرٹ کتنے کی ہے اور کیا اس میں [لفظ] رنگ... رنگ... [/لفظ] نیلا بھی ملتا ہے یا صرف یہی ہے",
        "stutter_unit": "رنگ",
        "label": 2,
        "label_name": "word_repetition",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# بلاک — Block/pause stutter (3 samples)
# Pacing: natural flow ... freeze... recovery ... natural flow
# ──────────────────────────────────────────────────────────────────────────────
BLOCK_SAMPLES = [
    {
        "id": "BLOCK_001",
        "scenario": "Doctor Visit",
        "tts_text": "ڈاکٹر صاحب مجھے کل رات سے بخار ہے اور سر میں بہت... درد ہو رہا... مجھے دوائی لکھ دیں",
        "annotated": "ڈاکٹر صاحب مجھے کل رات سے بخار ہے اور سر میں بہت [بلاک] [/بلاک] درد ہو رہا... مجھے دوائی لکھ دیں",
        "stutter_unit": None,
        "label": 3,
        "label_name": "block",
    },
    {
        "id": "BLOCK_002",
        "scenario": "Apologising to Friend",
        "tts_text": "یار مجھے معلوم ہے میں نے غلطی کی اور میں تم سے... معافی مانگنا... چاہتا ہوں دل سے سچ میں",
        "annotated": "یار مجھے معلوم ہے میں نے غلطی کی اور میں تم سے [بلاک] [/بلاک] معافی مانگنا... چاہتا ہوں دل سے سچ میں",
        "stutter_unit": None,
        "label": 3,
        "label_name": "block",
    },
    {
        "id": "BLOCK_003",
        "scenario": "Phone Call to Family",
        "tts_text": "ابو میں آج یونیورسٹی سے جلدی آ رہا ہوں کیونکہ آج کی... کلاس نہیں... ہوئی سر نے چھٹی دے دی",
        "annotated": "ابو میں آج یونیورسٹی سے جلدی آ رہا ہوں کیونکہ آج کی [بلاک] [/بلاک] کلاس نہیں... ہوئی سر نے چھٹی دے دی",
        "stutter_unit": None,
        "label": 3,
        "label_name": "block",
    },
]

ALL_SAMPLES = SYLLABLE_SAMPLES + WORD_SAMPLES + BLOCK_SAMPLES

# Folder mapping
FOLDER_MAP = {
    "syllable_repetition": "samples/syllable_repetition",
    "word_repetition":     "samples/word_repetition",
    "block":               "samples/block",
}


def call_tts_api(text):
    """Call ElevenLabs TTS API with eleven_v3."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
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
    print("=" * 60)
    print("StutterNet+ — FULL BATCH GENERATION")
    print(f"Voice  : {VOICE_NAME} ({VOICE_ID})")
    print(f"Model  : {MODEL_ID}")
    print(f"Samples: {len(ALL_SAMPLES)} (4 حرف + 3 لفظ + 3 بلاک)")
    print("=" * 60)

    for folder in FOLDER_MAP.values():
        os.makedirs(folder, exist_ok=True)

    annotation_samples = []
    total = len(ALL_SAMPLES)

    for i, sample in enumerate(ALL_SAMPLES, 1):
        sid = sample["id"]
        folder = FOLDER_MAP[sample["label_name"]]
        tts_text = sample["tts_text"]
        mp3_path = f"{folder}/{sid}.mp3"
        wav_path = f"{folder}/{sid}.wav"

        print(f"\n[{i}/{total}] {sid} — {sample.get('scenario', '')} ({sample['label_name']})")
        print(f"  TTS: {tts_text}")

        # Try up to 2 times
        audio_bytes = None
        for attempt in range(1, 3):
            try:
                audio_bytes = call_tts_api(tts_text)
                break
            except requests.exceptions.HTTPError as e:
                print(f"  Attempt {attempt} failed (HTTP {e.response.status_code}): {e}")
                try:
                    print(f"  Response: {e.response.text[:200]}")
                except Exception:
                    pass
                if attempt < 2:
                    print("  Retrying in 3 seconds...")
                    time.sleep(3)
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt < 2:
                    print("  Retrying in 3 seconds...")
                    time.sleep(3)

        if audio_bytes is None:
            print(f"  ERROR: Could not generate {sid} — skipping.")
            continue

        # Save MP3
        with open(mp3_path, "wb") as fout:
            fout.write(audio_bytes)

        # Convert to WAV 16kHz mono 16-bit PCM
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(wav_path, format="wav")
            os.remove(mp3_path)
            duration = len(audio) / 1000.0
            size_kb = os.path.getsize(wav_path) / 1024
            print(f"  Generated {sid} — {duration:.1f}s — {size_kb:.0f}KB")
        except Exception as e:
            print(f"  WARNING: WAV conversion failed: {e}")
            print(f"  MP3 kept at: {mp3_path}")
            duration = 0.0
            size_kb = os.path.getsize(mp3_path) / 1024 if os.path.exists(mp3_path) else 0

        # Annotation entry
        entry = {
            "id":               sid,
            "wav_file":         wav_path,
            "spectrogram_file": f"spectrograms/{sample['label_name']}/{sid}_spectrogram.npy",
            "spectrogram_image": f"spectrograms/{sample['label_name']}/{sid}_spectrogram.png",
            "tts_text":         tts_text,
            "annotated":        sample["annotated"],
            "scenario":         sample.get("scenario", ""),
            "stutter_unit":     sample.get("stutter_unit"),
            "category_urdu":    {"syllable_repetition": "حرف", "word_repetition": "لفظ", "block": "بلاک"}[sample["label_name"]],
            "label":            sample["label"],
            "label_name":       sample["label_name"],
            "language":         "Urdu",
            "duration_seconds": round(duration, 1),
            "voice_used":       VOICE_NAME,
        }
        annotation_samples.append(entry)

        if i < total:
            time.sleep(1.5)

    # ── Save annotations.json ────────────────────────────────────────────────
    os.makedirs("annotations", exist_ok=True)
    annotations = {
        "dataset_info": {
            "name":                   "StutterNet+ Urdu Stuttering Dataset",
            "version":                "0.2 — Urdu script batch",
            "total_samples":          len(annotation_samples),
            "stutter_types_included": ["syllable_repetition", "word_repetition", "block"],
            "stutter_types_urdu":     {"حرف": "syllable_repetition", "لفظ": "word_repetition", "بلاک": "block"},
            "sample_rate":            16000,
            "target_duration_seconds": 7,
            "spectrogram_shape":      [257, 701, 1],
            "language":               "Urdu",
            "tts_model":              MODEL_ID,
            "voice_id":               VOICE_ID,
            "voice_name":             VOICE_NAME,
            "label_convention":       {"syllable_repetition": 1, "word_repetition": 2, "block": 3, "clean": 0},
        },
        "samples": annotation_samples,
    }
    with open("annotations/annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE — {len(annotation_samples)}/{total} files")
    print(f"Annotations saved to: annotations/annotations.json")
    print(f"\nNext step: PATH=\"$HOME/bin:$PATH\" python3 preprocess.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
