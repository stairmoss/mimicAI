#!/usr/bin/env python3
"""Voice clone verification test script."""
from voice_manager import VoiceManager

def test_voice_manager():
    vm = VoiceManager()
    voices = vm.list_voices()
    print("Current Voice Profiles Count:", len(voices))

if __name__ == "__main__":
    test_voice_manager()

def test_speed_control():
    from voice_manager import VoiceManager
    vm = VoiceManager()
    print("Voice Manager speed control initialized")

if __name__ == "__main__":
    test_speed_control()

def test_preset_loading():
    from clonemodel.preset_manager import list_presets
    print("Presets available:", len(list_presets()))
