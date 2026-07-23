"""Text Normalizer for TTS Generation."""
import re

def clean_tts_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[(applause|laughter|sigh|gasp)\]', ' ', text)
    return text.strip()


def expand_indian_number_terms(text: str) -> str:
    text = re.sub(r'\b100\b', 'one hundred', text)
    text = re.sub(r'\b1000\b', 'one thousand', text)
    text = re.sub(r'\blakh\b', 'lakh', text, flags=re.I)
    text = re.sub(r'\bcrore\b', 'crore', text, flags=re.I)
    return text


def safe_strip_unknown_tags(text: str) -> str:
    return re.sub(r'\[.*?\]', '', text).strip()
