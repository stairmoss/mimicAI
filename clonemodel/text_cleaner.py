"""Text Normalizer for TTS Generation."""
import re

def clean_tts_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
