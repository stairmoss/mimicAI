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


def build_gemini_chat_payload(messages: List[Dict[str, str]], model: Optional[str] = None) -> Dict:
    selected_model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return {
        "model": selected_model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.7,
    }


def validate_gemini_request() -> tuple[bool, Optional[str]]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return False, "GEMINI_API_KEY is not configured"
    return True, None


def get_fallback_provider() -> str:
    if is_gemini_available():
        return "gemini-2.0-flash"
    return "qwen3-32b"
