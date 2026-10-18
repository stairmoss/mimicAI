import io
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_converter = None
_converter_lock = threading.Lock()
_se_cache = {}


def _get_converter():
    global _converter
    with _converter_lock:
        if _converter is not None:
            return _converter

        try:
            from openvoice_cli.api import ToneColorConverter
            from openvoice_cli.downloader import download_checkpoint

            pkg_dir = os.path.dirname(os.path.realpath(__import__("openvoice_cli").__file__))
            ckpt_dir = os.path.join(pkg_dir, "checkpoints", "converter")

            if not os.path.exists(os.path.join(ckpt_dir, "checkpoint.pth")):
                os.makedirs(ckpt_dir, exist_ok=True)
                download_checkpoint(ckpt_dir)

            t0 = time.monotonic()
            conv = ToneColorConverter(
                os.path.join(ckpt_dir, "config.json"), device="cpu"
            )
            conv.load_ckpt(os.path.join(ckpt_dir, "checkpoint.pth"))
            logger.info(f"ToneColorConverter ready in {time.monotonic() - t0:.1f}s")
            _converter = conv
            return conv
        except Exception as exc:
            logger.error(f"Failed to load OpenVoice converter: {exc}")
            return None


def _extract_se(audio_path: str, converter) -> Optional[torch.Tensor]:
    try:
        import librosa
        from openvoice_cli.mel_processing import spectrogram_torch

        hps = converter.hps
        ref_audio, sr = librosa.load(audio_path, sr=hps.data.sampling_rate)
        y = torch.FloatTensor(ref_audio).unsqueeze(0).to(converter.device)
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


def _get_target_se(ref_path: str, converter):
    if ref_path in _se_cache:
        return _se_cache[ref_path]

    se = _extract_se(ref_path, converter)
    if se is not None:
        _se_cache[ref_path] = se
    return se


def clone_voice(
    text: str,
    ref_audio_path: str,
    language: str = "en",
    progress_callback=None,
    piper_model_path: Optional[str] = None,
    skip_openvoice: bool = False,
) -> Optional[bytes]:
    def progress(phase, step, total, msg):
        if progress_callback:
            try:
                progress_callback(phase, step, total, msg)
            except Exception:
                pass

    base_wav = None
    out_wav = None
    try:
        converter = None
        if not skip_openvoice:
            progress("init", 0, 4, "Loading voice converter...")
            converter = _get_converter()
            if converter is None:
                logger.error("OpenVoice converter not available")
                return None

        progress("tts", 1, 4, "Generating base speech...")
        base_wav = _generate_base_tts(text, language, piper_model_path)
        if base_wav is None:
            return None

        if skip_openvoice:
            progress("done", 4, 4, "Speech generated!")
            with open(base_wav, 'rb') as f:
                return f.read()

        progress("embed", 2, 4, "Extracting voice features...")
        src_se = _extract_se(base_wav, converter)
        tgt_se = _get_target_se(ref_audio_path, converter)

        if src_se is None or tgt_se is None:
            logger.error("Failed to extract speaker embeddings")
            return None

        progress("convert", 3, 4, "Cloning voice...")
        out_wav = base_wav.replace(".wav", "_cloned.wav")
        t0 = time.monotonic()

        converter.convert(
            audio_src_path=base_wav,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=out_wav,
        )
        logger.info(f"Voice conversion took {time.monotonic() - t0:.1f}s")

        progress("done", 4, 4, "Voice cloned!")
        with open(out_wav, "rb") as f:
            return f.read()

    except Exception as exc:
        logger.error(f"Voice cloning error: {exc}", exc_info=True)
        return None
    finally:
        if base_wav:
            _cleanup_temp(base_wav)
        if out_wav:
            _cleanup_temp(out_wav)


_kokoro_model = None
_kokoro_lock = threading.Lock()


def _get_kokoro():
    global _kokoro_model
    with _kokoro_lock:
        if _kokoro_model is not None:
            return _kokoro_model
        try:
            from kokoro_onnx import Kokoro
            base_dir = Path(__file__).parent
            m_path = base_dir / "kokoro_models" / "kokoro-v1.0.int8.onnx"
            v_path = base_dir / "kokoro_models" / "voices-v1.0.bin"
            if not m_path.exists() or not v_path.exists():
                try:
                    import urllib.request
                    os.makedirs(base_dir / "kokoro_models", exist_ok=True)
                    if not m_path.exists():
                        urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx", str(m_path))
                    if not v_path.exists():
                        urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", str(v_path))
                except Exception as exc:
                    logger.warning(f"Kokoro auto-download failed: {exc}")
                    return None

            t0 = time.monotonic()
            _kokoro_model = Kokoro(str(m_path), str(v_path))
            logger.info(f"Kokoro loaded in {time.monotonic() - t0:.1f}s")
            return _kokoro_model
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            return None


def _generate_base_tts(text: str, language: str, piper_model_path: Optional[str] = None) -> Optional[str]:
    lang = str(language).lower().strip()
    voice_name = "af_sarah"
    kokoro_lang = "en-us"
    has_kokoro = False

    if lang in ("en", "en-us", "us"):
        voice_name, kokoro_lang, has_kokoro = "af_sarah", "en-us", True
    elif lang in ("en-gb", "gb", "uk"):
        voice_name, kokoro_lang, has_kokoro = "bf_emma", "en-gb", True
    elif lang in ("ja", "jp", "japanese"):
        voice_name, kokoro_lang, has_kokoro = "jf_alpha", "ja", True
    elif lang in ("zh", "cn", "chinese"):
        voice_name, kokoro_lang, has_kokoro = "zf_xiaobei", "zh", True

    try:
        if piper_model_path and os.path.exists(piper_model_path):
            try:
                import piper_tts as pt
                if pt.is_available(piper_model_path):
                    wav_path = tempfile.mktemp(suffix=".wav", prefix="mimic_base_")
                    ok = pt.synthesize_to_file(
                        text=text[:500],
                        output_path=wav_path,
                        model_path=piper_model_path,
                    )
                    if ok and os.path.exists(wav_path):
                        return wav_path
            except Exception as exc:
                logger.warning(f"Custom Piper error: {exc}, trying fallbacks")

        if has_kokoro:
            try:
                kokoro = _get_kokoro()
                if kokoro is not None:
                    import soundfile as sf
                    import librosa

                    wav_path = tempfile.mktemp(suffix=".wav", prefix="mimic_base_")
                    samples, sr = kokoro.create(text[:500], voice=voice_name, speed=1.0, lang=kokoro_lang)

                    if samples is not None and len(samples) > 0:
                        if sr != 22050:
                            samples = librosa.resample(samples, orig_sr=sr, target_sr=22050)
                            sr = 22050

                        sf.write(wav_path, samples, sr)
                        return wav_path
            except Exception as exc:
                logger.warning(f"Kokoro base TTS failed: {exc}")

        try:
            import piper_tts as pt
            if pt.is_available():
                wav_path = tempfile.mktemp(suffix=".wav", prefix="mimic_base_")
                ok = pt.synthesize_to_file(text=text[:500], output_path=wav_path)
                if ok and os.path.exists(wav_path):
                    return wav_path
        except Exception as exc:
            logger.warning(f"Piper base TTS fallback failed: {exc}")

        if not has_kokoro:
            try:
                kokoro = _get_kokoro()
                if kokoro is not None:
                    import soundfile as sf
                    import librosa

                    wav_path = tempfile.mktemp(suffix=".wav", prefix="mimic_base_")
                    samples, sr = kokoro.create(text[:500], voice="af_sarah", speed=1.0, lang="en-us")
                    if samples is not None and len(samples) > 0:
                        if sr != 22050:
                            samples = librosa.resample(samples, orig_sr=sr, target_sr=22050)
                            sr = 22050
                        sf.write(wav_path, samples, sr)
                        return wav_path
            except Exception as exc:
                logger.warning(f"Kokoro fallback failed: {exc}")

        return None
    except Exception as exc:
        logger.error(f"Base TTS generation failed: {exc}")
        return None


def _cleanup_temp(path: str):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def is_available(model_path: Optional[str] = None) -> bool:
    try:
        import piper_tts
        if piper_tts.is_available(model_path):
            return True
    except ImportError:
        pass
    try:
        import openvoice_cli
        return True
    except ImportError:
        return False


def preload():
    def load():
        _get_converter()
        try:
            import piper_tts
            if piper_tts.is_available():
                m = piper_tts.get_default_model_path()
                if m:
                    piper_tts._load_model(m)
        except Exception:
            pass

    t = threading.Thread(target=load, daemon=True)
    t.start()
