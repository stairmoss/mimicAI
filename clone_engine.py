"""Voice cloning engine for MimicAI.

Pipeline:
  1. Generate base speech via Piper TTS (primary — local, fast, high-quality)
     - Text is normalised (symbols, abbreviations, emoji stripped)
     - Synthesis uses tuned params: noise_scale=0.667, length_scale=1.1, noise_w=0.8
  2. (Optional) Extract speaker embedding from base speech
  3. (Optional) Extract speaker embedding from reference audio (cached)
  4. (Optional) Apply OpenVoice tone color conversion: base → cloned voice

Fallback chain: Piper → Kokoro → espeak-ng
"""

import gc
import io
import logging
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_converter = None
_converter_lock = threading.Lock()
_se_cache: dict[str, torch.Tensor] = {}


def _get_converter():
    """Load ToneColorConverter singleton (thread-safe, ~10MB model)."""
    global _converter
    with _converter_lock:
        if _converter is not None:
            return _converter

        try:
            from openvoice_cli.api import ToneColorConverter
            from openvoice_cli.downloader import download_checkpoint

            pkg_dir = os.path.dirname(os.path.realpath(
                __import__("openvoice_cli").__file__
            ))
            ckpt_dir = os.path.join(pkg_dir, "checkpoints", "converter")

            # Auto-download checkpoint if missing
            if not os.path.exists(os.path.join(ckpt_dir, "checkpoint.pth")):
                os.makedirs(ckpt_dir, exist_ok=True)
                logger.info("Downloading OpenVoice converter checkpoint…")
                download_checkpoint(ckpt_dir)

            t0 = time.monotonic()
            conv = ToneColorConverter(
                os.path.join(ckpt_dir, "config.json"), device="cpu"
            )
            conv.load_ckpt(os.path.join(ckpt_dir, "checkpoint.pth"))
            logger.info(f"OpenVoice converter loaded in {time.monotonic()-t0:.1f}s")
            _converter = conv
            return conv

        except Exception as exc:
            logger.error(f"Failed to load OpenVoice converter: {exc}")
            return None


def _extract_se(audio_path: str, converter) -> Optional[torch.Tensor]:
    """Extract speaker embedding directly from audio (no VAD/Whisper needed)."""
    try:
        import librosa
        from openvoice_cli.mel_processing import spectrogram_torch

        hps = converter.hps
        audio_ref, sr = librosa.load(audio_path, sr=hps.data.sampling_rate)
        y = torch.FloatTensor(audio_ref).unsqueeze(0).to(converter.device)
        y = spectrogram_torch(
            y,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        ).to(converter.device)
        with torch.no_grad():
            g = converter.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        return g
    except Exception as exc:
        logger.error(f"SE extraction failed for {audio_path}: {exc}")
        return None


def _get_target_se(ref_audio_path: str, converter) -> Optional[torch.Tensor]:
    """Get cached target speaker embedding."""
    cache_key = ref_audio_path
    if cache_key in _se_cache:
        return _se_cache[cache_key]

    se = _extract_se(ref_audio_path, converter)
    if se is not None:
        _se_cache[cache_key] = se
    return se


def clone_voice(
    text: str,
    ref_audio_path: str,
    language: str = "en",
    progress_callback=None,
    piper_model_path: Optional[str] = None,
    skip_openvoice: bool = False,
) -> Optional[bytes]:
    """Generate speech that sounds like the reference speaker.

    Args:
        text: Text to speak
        ref_audio_path: Path to reference WAV file
        language: Language code for local fallback engines
        progress_callback: Optional fn(phase, step, total, msg)
        piper_model_path: Optional custom Piper ONNX model
        skip_openvoice: If True, bypass ToneColorConverter (useful for fully custom Piper models)

    Returns:
        WAV bytes or None on failure
    """

    def _progress(phase, step, total, msg):
        if progress_callback:
            try:
                progress_callback(phase, step, total, msg)
            except Exception:
                pass

    base_wav_path = None
    out_path = None
    try:
        # Step 1: Load converter (if not skipping)
        converter = None
        if not skip_openvoice:
            _progress("init", 0, 4, "Loading voice converter…")
            converter = _get_converter()
            if converter is None:
                logger.error("OpenVoice converter not available")
                return None

        # Step 2: Generate base speech with Piper TTS
        _progress("tts", 1, 4, "Generating base speech…")
        base_wav_path = _generate_base_tts(text, language, piper_model_path)
        if base_wav_path is None:
            return None

        # If skipping OpenVoice, just return the Piper generated audio
        if skip_openvoice:
            _progress("done", 4, 4, "Speech generated via custom model!")
            with open(base_wav_path, "rb") as f:
                wav_bytes = f.read()
            return wav_bytes

        # Step 3: Extract speaker embeddings
        _progress("embed", 2, 4, "Extracting voice features…")
        source_se = _extract_se(base_wav_path, converter)
        target_se = _get_target_se(ref_audio_path, converter)

        if source_se is None or target_se is None:
            logger.error("Failed to extract speaker embeddings")
            return None

        # Step 4: Convert voice color
        _progress("convert", 3, 4, "Cloning voice…")
        out_path = base_wav_path.replace(".wav", "_cloned.wav")
        t0 = time.monotonic()

        converter.convert(
            audio_src_path=base_wav_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
        )
        logger.info(f"Voice conversion done in {time.monotonic()-t0:.1f}s")

        _progress("done", 4, 4, "Voice cloned!")

        # Read output
        with open(out_path, "rb") as f:
            wav_bytes = f.read()

        return wav_bytes

    except Exception as exc:
        logger.error(f"Voice cloning error: {exc}", exc_info=True)
        return None
    finally:
        if base_wav_path:
            _cleanup_temp(base_wav_path)
        if out_path:
            _cleanup_temp(out_path)


_kokoro_model = None
_kokoro_lock = threading.Lock()

def _get_kokoro():
    """Load Kokoro TTS singleton."""
    global _kokoro_model
    with _kokoro_lock:
        if _kokoro_model is not None:
            return _kokoro_model
        try:
            from kokoro_onnx import Kokoro
            model_path = "/mnt/18A660FBA660DB30/voiceclone_AI/mimicAI/kokoro_models/kokoro-v1.0.int8.onnx"
            voices_path = "/mnt/18A660FBA660DB30/voiceclone_AI/mimicAI/kokoro_models/voices-v1.0.bin"
            if not os.path.exists(model_path) or not os.path.exists(voices_path):
                logger.error("Kokoro models not found in kokoro_models directory.")
                return None
            logger.info("Loading Kokoro TTS model (82M)...")
            t0 = time.monotonic()
            _kokoro_model = Kokoro(model_path, voices_path)
            logger.info(f"Kokoro loaded in {time.monotonic()-t0:.1f}s")
            return _kokoro_model
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            return None


def _generate_base_tts(text: str, language: str, piper_model_path: Optional[str] = None) -> Optional[str]:
    """Generate base TTS → WAV file.

    Priority:
      1. Custom fine-tuned Piper ONNX model (if piper_model_path is set)
      2. Bundled Piper model via piper_tts module (preferred default)
      3. Kokoro TTS (fallback if Piper unavailable)
    """
    try:
        # ── 1. Custom or default Piper (primary) ────────────────────────────
        try:
            import piper_tts as _pt

            # Resolve which model to use
            model = piper_model_path if (piper_model_path and os.path.exists(piper_model_path)) else None

            if _pt.is_available(model):
                wav_path = tempfile.mktemp(suffix=".wav", prefix="mimicai_base_")
                ok = _pt.synthesize_to_file(
                    text=text[:500],
                    output_path=wav_path,
                    model_path=model,         # None → auto-discover bundled model
                    noise_scale=0.667,        # adds natural breathiness
                    length_scale=1.1,         # slightly slower = more natural
                    noise_w_scale=0.8,        # varied phoneme duration prosody
                )
                if ok and os.path.exists(wav_path):
                    logger.info(f"Piper base TTS ({'custom' if model else 'default'}) OK")
                    return wav_path
                logger.warning("Piper synthesis returned empty — trying Kokoro fallback")
        except Exception as exc:
            logger.warning(f"Piper TTS error: {exc} — trying Kokoro fallback")

        # ── 2. Kokoro TTS (secondary fallback) ──────────────────────────────
        kokoro = _get_kokoro()
        if kokoro is not None:
            import soundfile as sf
            import librosa

            wav_path = tempfile.mktemp(suffix=".wav", prefix="mimicai_base_")
            samples, sample_rate = kokoro.create(text[:500], voice="af_sarah", speed=1.0, lang="en-us")

            # OpenVoice expects exactly 22050 Hz
            if sample_rate != 22050:
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=22050)
                sample_rate = 22050

            sf.write(wav_path, samples, sample_rate)
            logger.info("Kokoro fallback TTS OK")
            return wav_path

        logger.error("All TTS engines unavailable")
        return None

    except Exception as exc:
        logger.error(f"Base TTS generation failed: {exc}")
        return None


def _cleanup_temp(path: str):
    """Remove temp file silently."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def is_available(model_path: Optional[str] = None) -> bool:
    """Check if voice cloning pipeline is available (Piper is the minimum requirement)."""
    try:
        import piper_tts
        if piper_tts.is_available(model_path):
            return True
    except ImportError:
        pass
    # Also accept if OpenVoice is available even without Piper (legacy path)
    try:
        import openvoice_cli  # noqa: F401
        return True
    except ImportError:
        return False


def preload():
    """Pre-load the models in background."""
    def _load():
        _get_converter()
        try:
            import piper_tts
            if piper_tts.is_available():
                # This will cache the default model in memory
                default_model = piper_tts.get_default_model_path()
                if default_model:
                    piper_tts._load_model(default_model)
        except Exception:
            pass
    t = threading.Thread(target=_load, daemon=True)
    t.start()
