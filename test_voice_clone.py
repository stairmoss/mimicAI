#!/usr/bin/env python3
from voice_manager import VoiceManager
from clonemodel.preset_manager import list_presets
from clonemodel.text_cleaner import clean_tts_text


def run_checks():
    vm = VoiceManager()
    voices = vm.list_voices()
    print("Voice profiles count:", len(voices))

    presets = list_presets()
    print("Presets available:", len(presets))

    # zero-shot voice design check
    res = vm._generate_designed_voice("Hello test", {"gender": "female", "accent": "us"})
    print("Zero shot test:", "OK" if res is not None else "fallback active")

    cleaned = clean_tts_text("Hello [laughter] world")
    print("Cleaned text:", cleaned)


if __name__ == "__main__":
    run_checks()
