"""
StutterNet+ — Test Step: Generate 3 test samples (1 per stutter category).
Run: PATH="$HOME/bin:$PATH" python3 generate_test.py

After running, listen to the 3 WAV files and approve or give feedback.
"""
import requests
import json
import time
import os
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY = "sk_8dc5f216880280427d534d296e25fcee1788403e46bf1144"
VOICE_ID = "A4ASMsAwv9E88KBNEJJy"   # Abdullah (cloned)
VOICE_NAME = "Abdullah"
MODEL_ID = "eleven_v3"               # Latest, best quality
# ──────────────────────────────────────────────────────────────────────────────

# 3 test samples — one per category
TEST_SAMPLES = [
    {
        "id": "TEST_HARF_001",
        "category": "syllable_repetition",
        "category_urdu": "حرف",
        "scenario": "Job Interview",
        "tts_text": "میں نے پچھلے پانچ سال مارکیٹنگ میں... ک... ک... کام کیا ہے اور مجھے اس فیلڈ میں کافی تجربہ ہے",
        "annotated": "میں نے پچھلے پانچ سال مارکیٹنگ میں [حرف] ک... ک... [/حرف] کام کیا ہے اور مجھے اس فیلڈ میں کافی تجربہ ہے",
        "stutter_unit": "ک",
        "label": 1,
        "label_name": "syllable_repetition",
    },
    {
        "id": "TEST_LAFZ_001",
        "category": "word_repetition",
        "category_urdu": "لفظ",
        "scenario": "Ordering Food",
        "tts_text": "مجھے ایک پلیٹ بریانی دے دیں اور ساتھ میں... جوتے... جوتے نہیں رائتہ بھی لگا دیں",
        "annotated": "مجھے ایک پلیٹ بریانی دے دیں اور ساتھ میں [لفظ] جوتے... جوتے [/لفظ] نہیں رائتہ بھی لگا دیں",
        "stutter_unit": "جوتے",
        "label": 2,
        "label_name": "word_repetition",
    },
    {
        "id": "TEST_BLOCK_001",
        "category": "block",
        "category_urdu": "بلاک",
        "scenario": "Doctor Visit",
        "tts_text": "ڈاکٹر صاحب مجھے کل رات سے بخار ہے اور سر میں بہت درد... ہوں... مجھے دوائی لکھ دیں",
        "annotated": "ڈاکٹر صاحب مجھے کل رات سے بخار ہے اور سر میں بہت درد [بلاک] [/بلاک] ہوں... مجھے دوائی لکھ دیں",
        "stutter_unit": None,
        "label": 3,
        "label_name": "block",
    },
]


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
    print("StutterNet+ — TEST BATCH (3 samples, 1 per category)")
    print(f"Voice  : {VOICE_NAME} ({VOICE_ID})")
    print(f"Model  : {MODEL_ID}")
    print("=" * 60)

    os.makedirs("samples/syllable_repetition", exist_ok=True)
    os.makedirs("samples/word_repetition", exist_ok=True)
    os.makedirs("samples/block", exist_ok=True)

    generated_files = []

    for i, sample in enumerate(TEST_SAMPLES, 1):
        sid = sample["id"]
        folder = sample["category"]
        tts_text = sample["tts_text"]
        mp3_path = f"samples/{folder}/{sid}.mp3"
        wav_path = f"samples/{folder}/{sid}.wav"

        print(f"\n[{i}/3] {sid} — {sample['category_urdu']} ({sample['scenario']})")
        print(f"  TTS text: {tts_text}")

        # Try up to 2 times
        audio_bytes = None
        for attempt in range(1, 3):
            try:
                audio_bytes = call_tts_api(tts_text)
                break
            except requests.exceptions.HTTPError as e:
                print(f"  Attempt {attempt} failed (HTTP {e.response.status_code}): {e}")
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
            generated_files.append(wav_path)
        except Exception as e:
            print(f"  WARNING: WAV conversion failed: {e}")
            print(f"  MP3 kept at: {mp3_path}")
            generated_files.append(mp3_path)

        if i < len(TEST_SAMPLES):
            time.sleep(2)

    # ── STOP & WAIT ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST BATCH COMPLETE — HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print("Generated 3 test samples (1 per stutter category):\n")

    for sample in TEST_SAMPLES:
        sid = sample["id"]
        folder = sample["category"]
        print(f"  {sample['category_urdu']} ({sample['label_name']})")
        print(f"    File     : samples/{folder}/{sid}.wav")
        print(f"    Scenario : {sample['scenario']}")
        print(f"    Annotated: {sample['annotated']}")
        print()

    print("ACTION REQUIRED:")
    print("Please listen to each test file and evaluate:")
    for f in generated_files:
        print(f"  {f}")
    print()
    print("Then reply with one of:")
    print('  "APPROVED — generate full batch"')
    print('  "REDO — [feedback about what to fix]"')
    print()
    print("DO NOT PROCEED until human confirms.")
    print("=" * 60)


if __name__ == "__main__":
    main()
