"""Google Gemini API Helper for MimicAI."""
import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ENDPOINT = os.environ.get(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)

def is_gemini_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())

def get_gemini_config() -> Dict[str, str]:
    return {
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        "available": is_gemini_available(),
    }
