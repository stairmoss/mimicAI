"""Voice Profile Validator."""
import re

def validate_profile_name(name: str) -> tuple[bool, str]:
    if not name or len(name.strip()) == 0:
        return False, "Profile name cannot be empty."
    if len(name) > 50:
        return False, "Profile name is too long (max 50 chars)."
    return True, name.strip()
