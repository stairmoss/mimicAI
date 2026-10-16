import io
import logging
import os
import re
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

this_dir = Path(__file__).parent
models_dir = this_dir / "piper_models"

_default_model = None
_model_cache = {}
_model_lock = threading.Lock()


def _find_default_model():
    global _default_model
    if _default_model is not None:
        return _default_model
    if not models_dir.exists():
        return None
    files = sorted(models_dir.glob("*.onnx"))
    if not files:
        return None
    for pref in ("high", "medium"):
        for f in files:
            if pref in f.name.lower():
                _default_model = f
                return f
    _default_model = files[0]
    return _default_model


def _load_model(model_path: str):
    with _model_lock:
        if model_path in _model_cache:
            return _model_cache[model_path]
        try:
            from piper import PiperVoice
            t0 = time.monotonic()
            voice = PiperVoice.load(model_path)
            logger.info(f"Loaded Piper model in {time.monotonic() - t0:.1f}s")
            _model_cache[model_path] = voice
            return voice
        except Exception as exc:
            logger.error(f"Failed loading Piper model {model_path}: {exc}")
            return None


# TODO: refactor abbreviation list into a json config later
abbrevs = [
    (re.compile(r'\bDr\.', re.I), "Doctor"),
    (re.compile(r'\bMr\.', re.I), "Mister"),
    (re.compile(r'\bMrs\.', re.I), "Missus"),
    (re.compile(r'\bMs\.', re.I), "Miss"),
    (re.compile(r'\bProf\.', re.I), "Professor"),
    (re.compile(r'\bSt\.', re.I), "Saint"),
    (re.compile(r'\betc\.', re.I), "etcetera"),
    (re.compile(r'\be\.g\.', re.I), "for example"),
    (re.compile(r'\bi\.e\.', re.I), "that is"),
    (re.compile(r'\bvs\.', re.I), "versus"),
    (re.compile(r'\bAI\b'), "A I"),
    (re.compile(r'\bAPI\b'), "A P I"),
    (re.compile(r'\bTTS\b'), "T T S"),
    (re.compile(r'\bLLM\b'), "L L M"),
    (re.compile(r'\bURL\b'), "U R L"),
    (re.compile(r'\bHTML\b'), "H T M L"),
    (re.compile(r'\bGPU\b'), "G P U"),
    (re.compile(r'\bCPU\b'), "C P U"),
    (re.compile(r'\bRAM\b'), "ram"),
]


def _expand_numbers(text: str) -> str:
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
    text = re.sub(r'\$(\d+)\.(\d{2})\b', lambda m:
        f"{m.group(1)} dollar{'s' if int(m.group(1)) != 1 else ''} and {m.group(2)} cent{'s' if int(m.group(2)) != 1 else ''}", text)
    text = re.sub(r'\$(\d+)\b', lambda m:
        f"{m.group(1)} dollar{'s' if int(m.group(1)) != 1 else ''}", text)
    text = re.sub(r'(\d+)\s*°\s*[Cc]', r'\1 degrees Celsius', text)
    text = re.sub(r'(\d+)\s*°\s*[Ff]', r'\1 degrees Fahrenheit', text)
    text = re.sub(r'(\d+)\s*°', r'\1 degrees', text)
    text = re.sub(r'\b(\d+)st\b', r'\1 first', text)
    text = re.sub(r'\b(\d+)nd\b', r'\1 second', text)
    text = re.sub(r'\b(\d+)rd\b', r'\1 third', text)
    text = re.sub(r'\b(\d+)th\b', r'\1 th', text)
    return text


def normalize_text(text: str) -> str:
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))

    # strip emoji ranges espeak cannot phonemize
    text = re.sub(
        r'[\U0001F300-\U0001FFFF\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0001F1E0-\U0001F1FF]',
        '', text, flags=re.UNICODE
    )

    for pat, rep in abbrevs:
        text = pat.sub(rep, text)

    text = _expand_numbers(text)

    symbols = {
        "&": " and ",
        "+": " plus ",
        "=": " equals ",
        ">": " greater than ",
        "<": " less than ",
        "@": " at ",
        "#": " hash ",
        "*": " star ",
        "~": " approximately ",
        "→": " to ",
        "←": " from ",
        "↔": " bidirectional ",
        "…": "...",
        "\u2014": ", ",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        '\u201c': '"',
        '\u201d': '"',
    }
    for sym, val in symbols.items():
        text = text.replace(sym, val)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_text(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?]) +', text)
    return [p.strip() for p in parts if p.strip()]


def preprocess_reference_audio(wav_path: str, target_dbfs: float = -20.0) -> str:
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence

        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

        gain = target_dbfs - audio.dBFS
        audio = audio.apply_gain(gain)

        # trim leading/trailing quiet spaces
        chunks = split_on_silence(
            audio,
            min_silence_len=300,
            silence_thresh=-40,
            keep_silence=50,
        )

        clean = chunks[0] if chunks else audio
        if chunks and len(chunks) > 1:
            for c in chunks[1:]:
                clean += c

        gain = target_dbfs - clean.dBFS
        clean = clean.apply_gain(gain)

        out_path = wav_path.replace(".wav", "_clean.wav")
        clean.export(out_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        return out_path
    except Exception as exc:
        logger.warning(f"Audio preprocess failed ({exc}), using raw file")
        return wav_path


def synthesize(
    text: str,
    model_path: Optional[str] = None,
    noise_scale: float = 0.667,
    length_scale: float = 1.1,
    noise_w_scale: float = 0.8,
    speaker_id: Optional[int] = None,
) -> Optional[bytes]:
    if model_path is None:
        default = _find_default_model()
        if default is None:
            logger.error("No Piper model found")
            return None
        model_path = str(default)

    if not os.path.exists(model_path):
        logger.error(f"Piper model not found at {model_path}")
        return None

    voice = _load_model(model_path)
    if voice is None:
        return None

    from clonemodel.text_cleaner import clean_tts_text, expand_indian_number_terms
    clean_text = expand_indian_number_terms(clean_tts_text(normalize_text(text)))
    if not clean_text:
        return None

    try:
        from piper import SynthesisConfig

        cfg = SynthesisConfig(
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=True,
        )

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)

            # piper has better prosody with sentence chunks
            for sent in split_text(clean_text):
                for chunk in voice.synthesize(sent, syn_config=cfg):
                    wf.writeframes(chunk.audio_int16_bytes)

        return buf.getvalue()
    except Exception as exc:
        logger.error(f"Piper synthesis error: {exc}", exc_info=True)
        return None


def synthesize_to_file(
    text: str,
    output_path: str,
    model_path: Optional[str] = None,
    noise_scale: float = 0.667,
    length_scale: float = 1.1,
    noise_w_scale: float = 0.8,
) -> bool:
    data = synthesize(
        text=text,
        model_path=model_path,
        noise_scale=noise_scale,
        length_scale=length_scale,
        noise_w_scale=noise_w_scale,
    )
    if data is None:
        return False
    try:
        with open(output_path, "wb") as f:
            f.write(data)
        return True
    except OSError as exc:
        logger.error(f"Failed writing wav to {output_path}: {exc}")
        return False


def is_available(model_path: Optional[str] = None) -> bool:
    try:
        import piper  # noqa: F401
        target = Path(model_path) if model_path else _find_default_model()
        if target is None:
            return False
        json_cfg = Path(f"{target}.json")
        return target.exists() and json_cfg.exists()
    except ImportError:
        return False


def get_default_model_path() -> Optional[str]:
    m = _find_default_model()
    return str(m) if m else None


def get_sample_rate(model_path: Optional[str] = None) -> int:
    if model_path is None:
        target = _find_default_model()
        if target is None:
            return 22050
        model_path = str(target)
    voice = _load_model(model_path)
    if voice is None:
        return 22050
    return voice.config.sample_rate
