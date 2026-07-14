"""Voice Clone Configuration Defaults."""
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_CHANNELS = 1
DEFAULT_TARGET_DBFS = -20.0
MIN_AUDIO_DURATION_SEC = 3.0
MAX_AUDIO_DURATION_SEC = 60.0


def check_audio_data_quality(audio_bytes: bytes) -> tuple[bool, str]:
    if not audio_bytes or len(audio_bytes) < 1000:
        return False, "Audio file is empty or too short."
    return True, "Audio file quality is acceptable."
