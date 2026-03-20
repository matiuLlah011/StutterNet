"""
StutterNet+ — Step 1: List all ElevenLabs voices and identify cloned voices.
Run: python list_voices.py
"""
import requests
import json
import sys

# ─── PASTE YOUR ELEVENLABS API KEY HERE ──────────────────────────────────────
API_KEY = "sk_8dc5f216880280427d534d296e25fcee1788403e46bf1144"
# ─────────────────────────────────────────────────────────────────────────────


def list_voices():
    if API_KEY == "YOUR_ELEVENLABS_API_KEY_HERE":
        print("ERROR: Please set your ElevenLabs API key in list_voices.py (line 8).")
        sys.exit(1)

    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": API_KEY}

    print("Fetching voices from ElevenLabs...\n")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    voices = data.get("voices", [])
    print(f"Total voices found: {len(voices)}\n")

    cloned_voices = []

    for voice in voices:
        name        = voice.get("name", "N/A")
        voice_id    = voice.get("voice_id", "N/A")
        category    = voice.get("category", "N/A")
        labels      = voice.get("labels", {})
        description = labels.get("description", voice.get("description", "N/A"))

        print("----------------------------------------")
        print(f"Name        : {name}")
        print(f"Voice ID    : {voice_id}")
        print(f"Category    : {category}")
        print(f"Description : {description}")
        print("----------------------------------------")

        if category == "cloned":
            print(f"\n* CLONED VOICE FOUND *")
            print(f"Name     : {name}")
            print(f"Voice ID : {voice_id}\n")
            cloned_voices.append({"name": name, "voice_id": voice_id})

    # Fallback: if no cloned voices, pick best multilingual premade
    if not cloned_voices:
        print("\nNo cloned voices found. Searching for best multilingual premade voice...\n")
        multilingual_keywords = ["multilingual", "adam", "rachel", "domi", "bella"]
        for voice in voices:
            name = voice.get("name", "").lower()
            labels_str = str(voice.get("labels", {})).lower()
            if any(kw in name or kw in labels_str for kw in multilingual_keywords):
                entry = {"name": voice["name"], "voice_id": voice["voice_id"]}
                cloned_voices.append(entry)
                print(f"Selected fallback voice: {voice['name']} ({voice['voice_id']})")
                break

        # Last resort: first voice
        if not cloned_voices and voices:
            v = voices[0]
            cloned_voices.append({"name": v["name"], "voice_id": v["voice_id"]})
            print(f"Using first available voice: {v['name']}")

    # Save to JSON
    with open("cloned_voices.json", "w", encoding="utf-8") as f:
        json.dump(cloned_voices, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cloned_voices)} voice(s) to cloned_voices.json")
    print(f"\nFound {len(cloned_voices)} cloned voices. Proceeding to voice testing.")
    return cloned_voices


if __name__ == "__main__":
    list_voices()
