"""Preset Voice Profiles Manager."""
PRESET_VOICES = [
    {"id": "preset_en_in_f", "name": "Aditi (Indian English)", "language": "en", "gender": "female"},
    {"id": "preset_en_in_m", "name": "Rohan (Indian English)", "language": "en", "gender": "male"},
    {"id": "preset_hi_in_f", "name": "Priya (Hindi)", "language": "hi", "gender": "female"},
]

def list_presets():
    return PRESET_VOICES


def get_preset_by_id(preset_id: str):
    for p in PRESET_VOICES:
        if p["id"] == preset_id:
            return p
    return None
