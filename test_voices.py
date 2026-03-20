"""
StutterNet+ — Step 2: Generate test clips for each cloned voice.
Run ONLY after list_voices.py has been run.
Run: python test_voices.py

After running, listen to voice_samples/*.wav and tell the assistant:
    USE VOICE: {voice name}
"""
import requests
import json
import time
import os
import sys

# ─── PASTE YOUR ELEVENLABS API KEY HERE ──────────────────────────────────────
API_KEY = "sk_8dc5f216880280427d534d296e25fcee1788403e46bf1144"
# ─────────────────────────────────────────────────────────────────────────────

TEST_TEXT = "Mujhe mujhe actually pata nahi tha ke yeh project itna mushkil hoga."


def sanitize_filename(name):
    """Remove characters unsafe for filenames."""
    return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name).strip()


def test_voices():
    if API_KEY == "YOUR_ELEVENLABS_API_KEY_HERE":
        print("ERROR: Please set your ElevenLabs API key in test_voices.py (line 12).")
        sys.exit(1)

    os.makedirs("voice_samples", exist_ok=True)

    if not os.path.exists("cloned_voices.json"):
        print("ERROR: cloned_voices.json not found. Run list_voices.py first.")
        sys.exit(1)

    with open("cloned_voices.json", "r", encoding="utf-8") as f:
        cloned_voices = json.load(f)

    if not cloned_voices:
        print("ERROR: cloned_voices.json is empty. Run list_voices.py first.")
        sys.exit(1)

    print(f"Testing {len(cloned_voices)} voice(s)...\n")
    generated_wav_files = []

    for i, voice in enumerate(cloned_voices, 1):
        name     = voice["name"]
        voice_id = voice["voice_id"]
        safe_name = sanitize_filename(name)

        print(f"[{i}/{len(cloned_voices)}] Testing voice: {name} (ID: {voice_id})")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        body = {
            "text": TEST_TEXT,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.85,
                "style": 0.3,
                "use_speaker_boost": True
            }
        }

        try:
            response = requests.post(url, headers=headers, json=body, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"  ERROR: HTTP error for voice '{name}': {e}")
            print(f"  Response: {response.text}")
            continue
        except Exception as e:
            print(f"  ERROR: Failed to generate test clip for '{name}': {e}")
            continue

        # Save MP3
        mp3_path = f"voice_samples/{safe_name}_test.mp3"
        with open(mp3_path, "wb") as f:
            f.write(response.content)
        print(f"  Saved MP3 : {mp3_path}")

        # Convert to WAV 16kHz mono using pydub
        wav_path = f"voice_samples/{safe_name}_test.wav"
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(wav_path, format="wav")
            print(f"  Saved WAV : {wav_path}")
            generated_wav_files.append(wav_path)
        except Exception as e:
            print(f"  WARNING: Could not convert to WAV: {e}")
            print(f"  (ffmpeg may not be installed — MP3 is still available)")
            generated_wav_files.append(mp3_path)

        if i < len(cloned_voices):
            time.sleep(2)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VOICE TEST COMPLETE — HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print("Generated test clips for all cloned voices:")
    for voice in cloned_voices:
        safe = sanitize_filename(voice["name"])
        print(f"  voice_samples/{safe}_test.mp3")
        print(f"  voice_samples/{safe}_test.wav")

    print("\nACTION REQUIRED:")
    print("Please listen to each file in the voice_samples/ folder:")
    for f in generated_wav_files:
        print(f"  {f}")

    print("\nThen tell me which voice sounds the most natural for")
    print("Urdu/Pakistani speech by typing:")
    print('  "USE VOICE: {voice name}"')
    print()
    print("DO NOT PROCEED until the human confirms which voice to use.")
    print("=" * 60)
    # ──────────────────────────────────────────────────────────────────────────
    # STOP HERE — do not call generate_audio.py or any further steps.


if __name__ == "__main__":
    test_voices()
