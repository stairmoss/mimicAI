"""Voice Profile Validator."""
import re

def validate_profile_name(name: str) -> tuple[bool, str]:
    if not name or len(name.strip()) == 0:
        return False, "Profile name cannot be empty."
    if len(name) > 50:
        return False, "Profile name is too long (max 50 chars)."
    return True, name.strip()


def validate_audio_file_size(size_bytes: int) -> tuple[bool, str]:
    if size_bytes < 1000:
        return False, "Audio recording is too small (min 1KB required)."
    if size_bytes > 20 * 1024 * 1024:
        return False, "Audio recording exceeds maximum allowed size (20MB)."
    return True, "Audio size is valid."
