#!/usr/bin/env python3
import time
from voice_manager import VoiceManager

def main():
    print("Testing real neural voice cloning...")
    vm = VoiceManager()
    voices = vm.list_voices()
    print("Available voice profiles:", voices)
    if not voices:
        print("No voice profiles found.")
        return

    test_id = voices[0]["id"]
    print(f"Testing real voice clone with profile ID: {test_id} ({voices[0].get('name')})")

    t0 = time.time()
    wav_bytes = vm.generate_tts(
        text="Hello! This is a real neural voice cloning test using OpenVoice tone color conversion.",
        voice_id=test_id,
        prefer_clone=True,
        allow_fallback=True
    )
    dt = time.time() - t0

    if wav_bytes:
        print(f"SUCCESS (Standard fallback): Generated {len(wav_bytes)} bytes in {dt:.2f}s using engine '{vm.last_tts_engine}'!")
    else:
        print("FAILED to generate audio (Standard fallback).")

    print("\nTesting MimicVoice Lite via VoiceManager...")
    t0 = time.time()
    wav_bytes_lite = vm.generate_tts(
        text="Hello.",
        voice_id=test_id,
        prefer_clone=True,
        lightweight=True,
        allow_fallback=False
    )
    dt_lite = time.time() - t0

    if wav_bytes_lite:
        print(f"SUCCESS (Lite): Generated {len(wav_bytes_lite)} bytes in {dt_lite:.2f}s using engine '{vm.last_tts_engine}'!")
    else:
        print("FAILED to generate audio (Lite).")

if __name__ == "__main__":
    main()
