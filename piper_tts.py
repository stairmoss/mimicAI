"""Optimized Piper TTS engine for MimicAI.

Implements all best-practice optimisations:
  - Text normalisation  (symbols, abbreviations, unicode)
  - Advanced audio pre-processing (VAD trim, volume normalise to -20 dBFS)
  - Piper PiperVoice Python API with tuned synthesis config
    (noise_scale=0.667, length_scale=1.1, noise_w_scale=0.8)
  - Singleton model cache — only loaded once, thread-safe
  - Auto-discovery: looks for *.onnx in piper_models/ next to this file
"""

import gc
import io
import logging
import os
import re
import struct
import tempfile
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).parent
PIPER_MODELS_DIR = _THIS_DIR / "piper_models"

# Default bundled model (first .onnx in piper_models_dir, preferring "medium")
_DEFAULT_MODEL: Optional[Path] = None

def _find_default_model() -> Optional[Path]:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is not None:
        return _DEFAULT_MODEL
    if not PIPER_MODELS_DIR.exists():
        return None
    candidates = sorted(PIPER_MODELS_DIR.glob("*.onnx"))
    if not candidates:
        return None
    # Prefer models named *medium* or *high* for quality
    for pref in ("high", "medium"):
        for c in candidates:
            if pref in c.name.lower():
                _DEFAULT_MODEL = c
                return c
    _DEFAULT_MODEL = candidates[0]
    return _DEFAULT_MODEL


# ── Model singleton cache ──────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}  # model_path -> PiperVoice
_model_lock = threading.Lock()


def _load_model(model_path: str):
    """Load (or return cached) PiperVoice for *model_path*."""
    with _model_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]
        try:
            from piper import PiperVoice
            logger.info(f"Loading Piper model: {Path(model_path).name}")
            t0 = time.monotonic()
            voice = PiperVoice.load(model_path)
            logger.info(f"Piper model loaded in {time.monotonic() - t0:.1f}s  "
                        f"[sr={voice.config.sample_rate}]")
            _model_cache[model_path] = voice
            return voice
        except Exception as exc:
            logger.error(f"Failed to load Piper model {model_path}: {exc}")
            return None


# ── Text Normalisation ─────────────────────────────────────────────────────────

# Abbreviation expansion — add project-specific terms here
_ABBREVS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bDr\.', re.I),        "Doctor"),
    (re.compile(r'\bMr\.', re.I),        "Mister"),
    (re.compile(r'\bMrs\.', re.I),       "Missus"),
    (re.compile(r'\bMs\.', re.I),        "Miss"),
    (re.compile(r'\bProf\.', re.I),      "Professor"),
    (re.compile(r'\bSt\.', re.I),        "Saint"),
    (re.compile(r'\betc\.', re.I),       "etcetera"),
    (re.compile(r'\be\.g\.', re.I),      "for example"),
    (re.compile(r'\bi\.e\.', re.I),      "that is"),
    (re.compile(r'\bvs\.', re.I),        "versus"),
    (re.compile(r'\bAI\b'),              "A I"),
    (re.compile(r'\bAPI\b'),             "A P I"),
    (re.compile(r'\bTTS\b'),             "T T S"),
    (re.compile(r'\bLLM\b'),             "L L M"),
    (re.compile(r'\bURL\b'),             "U R L"),
    (re.compile(r'\bHTML\b'),            "H T M L"),
    (re.compile(r'\bGPU\b'),             "G P U"),
    (re.compile(r'\bCPU\b'),             "C P U"),
    (re.compile(r'\bRAM\b'),             "ram"),
]


def _expand_numbers(text: str) -> str:
    """Convert digit sequences and common symbols to words."""
    # Percentages: "100%" → "100 percent"
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
    # Currency: "$5.99" → "5 dollars and 99 cents"
    text = re.sub(r'\$(\d+)\.(\d{2})\b', lambda m:
        f"{m.group(1)} dollar{'s' if int(m.group(1)) != 1 else ''} "
        f"and {m.group(2)} cent{'s' if int(m.group(2)) != 1 else ''}", text)
    text = re.sub(r'\$(\d+)\b', lambda m:
        f"{m.group(1)} dollar{'s' if int(m.group(1)) != 1 else ''}", text)
    # Temperature: "37°C" → "37 degrees Celsius"
    text = re.sub(r'(\d+)\s*°\s*[Cc]', r'\1 degrees Celsius', text)
    text = re.sub(r'(\d+)\s*°\s*[Ff]', r'\1 degrees Fahrenheit', text)
    text = re.sub(r'(\d+)\s*°', r'\1 degrees', text)
    # Ordinals: "1st", "2nd", "3rd", "4th" etc.
    text = re.sub(r'\b(\d+)st\b', r'\1 first', text)
    text = re.sub(r'\b(\d+)nd\b', r'\1 second', text)
    text = re.sub(r'\b(\d+)rd\b', r'\1 third', text)
    text = re.sub(r'\b(\d+)th\b', r'\1 th', text)
    return text


def normalize_text(text: str) -> str:
    """Clean and normalise text before sending to Piper.

    1. Strip non-printable / weird Unicode
    2. Expand abbreviations
    3. Convert numbers/symbols to words
    4. Remove emoji and other non-ASCII that espeak-ng can't phonemize
    """
    if not text:
        return text

    # Normalise unicode (NFC) and strip control chars
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))

    # Strip emojis (code points in Symbol/Other emoji ranges)
    text = re.sub(
        r'[\U0001F300-\U0001FFFF'   # Misc symbols, transport, etc.
        r'\U00002702-\U000027B0'
        r'\U0000FE00-\U0000FE0F'    # Variation selectors
        r'\U0001F1E0-\U0001F1FF]',  # Flags
        '', text, flags=re.UNICODE
    )

    # Expand abbreviations
    for pattern, replacement in _ABBREVS:
        text = pattern.sub(replacement, text)

    # Number/symbol expansion
    text = _expand_numbers(text)

    # Replace common symbols that espeak-ng struggles with
    replacements = {
        "&":  " and ",
        "+":  " plus ",
        "=":  " equals ",
        ">":  " greater than ",
        "<":  " less than ",
        "@":  " at ",
        "#":  " hash ",
        "*":  " star ",
        "~":  " approximately ",
        "→":  " to ",
        "←":  " from ",
        "↔":  " bidirectional ",
        "…":  "...",
        "\u2014": ", ",  # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
    }
    for sym, word in replacements.items():
        text = text.replace(sym, word)

    # Collapse multiple spaces / blank lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_text(text: str) -> list[str]:
    """Splits by periods, exclamation marks, or question marks for higher Piper accuracy."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]


# ── Audio Pre-processing ───────────────────────────────────────────────────────

def preprocess_reference_audio(wav_path: str, target_dbfs: float = -20.0) -> str:
    """Pre-process a reference WAV for voice cloning:

    1. Convert to 22050 Hz / Mono / 16-bit
    2. Normalise loudness to *target_dbfs*
    3. Aggressive VAD silence trim (silence_thresh=-40 dBFS)

    Returns path to cleaned file (may be a new temp file).
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence

        audio = AudioSegment.from_file(wav_path)
        # Standardise format
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

        # Loudness normalisation
        change_in_dBFS = target_dbfs - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        # VAD: remove silence at boundaries aggressively
        chunks = split_on_silence(
            audio,
            min_silence_len=300,          # trim silences ≥300ms
            silence_thresh=-40,           # silence below -40 dBFS
            keep_silence=50,              # keep 50ms padding to avoid popping
        )

        if chunks:
            clean_audio = chunks[0]
            for c in chunks[1:]:
                clean_audio += c
        else:
            clean_audio = audio

        # Re-normalise after trimming (trim can shift dBFS)
        change_in_dBFS = target_dbfs - clean_audio.dBFS
        clean_audio = clean_audio.apply_gain(change_in_dBFS)

        out_path = wav_path.replace(".wav", "_clean.wav")
        clean_audio.export(out_path, format="wav",
                           parameters=["-acodec", "pcm_s16le"])
        logger.info(f"Reference audio preprocessed → {out_path}")
        return out_path

    except Exception as exc:
        logger.warning(f"Audio preprocessing skipped ({exc}), using raw file")
        return wav_path


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesize(
    text: str,
    model_path: Optional[str] = None,
    noise_scale: float = 0.667,
    length_scale: float = 1.1,
    noise_w_scale: float = 0.8,
    speaker_id: Optional[int] = None,
) -> Optional[bytes]:
    """Synthesize *text* with Piper and return raw WAV bytes.

    Args:
        text: Raw text (will be normalised automatically).
        model_path: Path to .onnx model. Uses bundled default if None.
        noise_scale: Breathiness / variability. Default 0.667; try 0.7 for
                     more human-like jitter.
        length_scale: Speed multiplier. >1 = slower. 1.1 is recommended for
                      natural prosody.
        noise_w_scale: Phoneme duration variability. Higher = more varied
                       prosody.
        speaker_id: For multi-speaker models; None → use model default.

    Returns:
        WAV bytes (RIFF header + PCM) or None on failure.
    """
    # Resolve model
    if model_path is None:
        default = _find_default_model()
        if default is None:
            logger.error("No Piper model found. Put an .onnx file in piper_models/")
            return None
        model_path = str(default)

    if not os.path.exists(model_path):
        logger.error(f"Piper model not found: {model_path}")
        return None

    # Load (cached)
    voice = _load_model(model_path)
    if voice is None:
        return None

    # Normalise text
    clean_text = normalize_text(text)
    if not clean_text:
        logger.warning("Empty text after normalisation")
        return None

    try:
        from piper import SynthesisConfig

        syn_cfg = SynthesisConfig(
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=True,
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            t0 = time.monotonic()

            sentences = split_text(clean_text)
            for sentence in sentences:
                # Piper performs better on short sentences
                for chunk in voice.synthesize(sentence, syn_config=syn_cfg):
                    wf.writeframes(chunk.audio_int16_bytes)

            elapsed = time.monotonic() - t0

        wav_bytes = buf.getvalue()
        logger.info(
            f"Piper synthesized {len(clean_text)} chars → "
            f"{len(wav_bytes)//1024}KB in {elapsed:.2f}s  "
            f"[ns={noise_scale} ls={length_scale} nw={noise_w_scale}]"
        )
        return wav_bytes

    except Exception as exc:
        logger.error(f"Piper synthesis failed: {exc}", exc_info=True)
        return None


def synthesize_to_file(
    text: str,
    output_path: str,
    model_path: Optional[str] = None,
    noise_scale: float = 0.667,
    length_scale: float = 1.1,
    noise_w_scale: float = 0.8,
) -> bool:
    """Synthesize text and write WAV to *output_path*. Returns True on success."""
    wav_bytes = synthesize(
        text=text,
        model_path=model_path,
        noise_scale=noise_scale,
        length_scale=length_scale,
        noise_w_scale=noise_w_scale,
    )
    if wav_bytes is None:
        return False
    try:
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return True
    except OSError as exc:
        logger.error(f"Failed to write WAV to {output_path}: {exc}")
        return False


def is_available(model_path: Optional[str] = None) -> bool:
    """Return True if piper-tts is importable and a valid model (+ json config) exists."""
    try:
        import piper  # noqa: F401

        if model_path is not None:
            target = Path(model_path)
        else:
            target = _find_default_model()

        if target is None:
            return False

        # Piper requires both the .onnx file and .onnx.json config
        json_path = Path(f"{target}.json")
        return target.exists() and json_path.exists()
    except ImportError:
        return False


def get_default_model_path() -> Optional[str]:
    """Return the path of the auto-discovered default Piper model, or None."""
    m = _find_default_model()
    return str(m) if m else None


def get_sample_rate(model_path: Optional[str] = None) -> int:
    """Return the sample rate of the given (or default) model."""
    if model_path is None:
        model_path_obj = _find_default_model()
        if model_path_obj is None:
            return 22050
        model_path = str(model_path_obj)
    voice = _load_model(model_path)
    if voice is None:
        return 22050
    return voice.config.sample_rate
