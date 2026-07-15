#!/usr/bin/env python3
"""Voice clone verification test script."""
from voice_manager import VoiceManager

def test_voice_manager():
    vm = VoiceManager()
    voices = vm.list_voices()
    print("Current Voice Profiles Count:", len(voices))

if __name__ == "__main__":
    test_voice_manager()
