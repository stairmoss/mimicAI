import gc
import io
import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

def _is_safe_id(profile_id: Optional[str]) -> bool:
    if not profile_id:
        return False
    import re
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", profile_id))

VOICES_DIR = Path(__file__).parent / "voices"
SHARD_DIR = Path(os.path.expanduser("~/.clonemodel_lite/shards"))

_model_lock = threading.Lock()
_omnivoice_model = None
_omnivoice_loaded = False
_omnivoice_full_model = None
_omnivoice_full_loaded = False


def _load_omnivoice_model():
    global _omnivoice_model, _omnivoice_loaded
    with _model_lock:
        if _omnivoice_loaded:
            return _omnivoice_model
        try:
            from clonemodel.lite.omnivoice_lite import OmniVoiceLite
            if not SHARD_DIR.exists():
                logger.warning(f"Shard dir {SHARD_DIR} missing. Model not loaded.")
                _omnivoice_loaded = True
                return None
            logger.info("Initializing OmniVoice Lite (CPU)...")
            t0 = time.monotonic()
            _omnivoice_model = OmniVoiceLite.from_pretrained(
                shard_dir=str(SHARD_DIR), max_memory_gb=3.0,
                device="cpu", enable_prefetch=True,
            )
            logger.info(f"OmniVoice Lite ready in {time.monotonic()-t0:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load OmniVoice Lite: {exc}", exc_info=True)
            _omnivoice_model = None
        finally:
            _omnivoice_loaded = True
        return _omnivoice_model


def _load_omnivoice_full_model():
    global _omnivoice_full_model, _omnivoice_full_loaded
    with _model_lock:
        if _omnivoice_full_loaded:
            return _omnivoice_full_model
        try:
            from clonemodel import OmniVoice

            model_name = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            logger.info("Initializing full OmniVoice on %s from %s...", device, model_name)
            t0 = time.monotonic()
            _omnivoice_full_model = OmniVoice.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=dtype,
                load_asr=False,
            )
            logger.info("Full OmniVoice ready in %.1fs", time.monotonic() - t0)
        except Exception as exc:
            logger.error("Failed to load full OmniVoice: %s", exc, exc_info=True)
            _omnivoice_full_model = None
        finally:
            _omnivoice_full_loaded = True
        return _omnivoice_full_model


def _float_audio_to_wav_bytes(audio, sample_rate: int) -> bytes:
    import soundfile as sf

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class VoiceManager:
    """Manages voice profiles and TTS generation.

    TTS priority (generate_tts):
      1. F5-TTS voice cloning using the selected reference voice.
      2. OmniVoice / OmniVoice Lite voice cloning.
      3. OpenVoice tone conversion over local Piper/Kokoro base speech.
      4. Piper direct local fallback.
    """

    def __init__(self, voices_dir: Optional[str] = None):
        self.voices_dir = Path(voices_dir) if voices_dir else VOICES_DIR
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.last_tts_engine = "none"
        try:
            from clone_engine import preload
            preload()
        except Exception:
            pass

    # -- Profile CRUD ----------------------------------------------------------

    def list_voices(self) -> List[Dict]:
        profiles = []
        for d in sorted(self.voices_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["id"] = d.name
                meta["has_audio"] = len(list(d.glob("reference.*"))) > 0
                profiles.append(meta)
            except Exception:
                pass
        return profiles

    def create_voice(self, audio_data: bytes, name: str, language: str = "en",
                     file_ext: str = "webm") -> Dict:
        profile_id = uuid.uuid4().hex[:12]
        profile_dir = self.voices_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        audio_path = profile_dir / f"reference.{file_ext}"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        wav_path = self._convert_to_wav(audio_path, profile_dir)
        if wav_path is None:
            raise RuntimeError("Audio conversion failed. Make sure ffmpeg is installed.")

        meta = {
            "name": name,
            "language": language,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_file": wav_path.name,
            "original_file": audio_path.name,
            "clone_ready": True,
        }
        with open(profile_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        meta["id"] = profile_id
        logger.info(f"Created voice profile: {name} ({profile_id})")
        return meta

    def delete_voice(self, profile_id: str) -> bool:
        if not _is_safe_id(profile_id):
            logger.warning(f"Unsafe profile_id rejected in delete_voice: {profile_id}")
            return False
        profile_dir = self.voices_dir / profile_id
        if not profile_dir.exists():
            return False
        shutil.rmtree(profile_dir, ignore_errors=True)
        try:
            from clone_engine import _se_cache
            ref = str(profile_dir / "reference.wav")
            _se_cache.pop(ref, None)
        except Exception:
            pass
        logger.info(f"Deleted voice profile: {profile_id}")
        return True

    def get_reference_audio_path(self, profile_id: str) -> Optional[str]:
        if not _is_safe_id(profile_id):
            logger.warning(f"Unsafe profile_id rejected in get_reference_audio_path: {profile_id}")
            return None
        profile_dir = self.voices_dir / profile_id
        if not profile_dir.exists():
            return None
        meta_path = profile_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                audio_file = meta.get("audio_file", "reference.wav")
                p = (profile_dir / audio_file).resolve()
                p.relative_to(profile_dir.resolve())
                if p.exists():
                    return str(p)
            except (ValueError, Exception):
                pass
        for ext in ("wav", "webm", "mp3", "ogg", "flac"):
            try:
                p = (profile_dir / f"reference.{ext}").resolve()
                p.relative_to(profile_dir.resolve())
                if p.exists():
                    return str(p)
            except (ValueError, Exception):
                pass
        return None

    def get_voice_metadata(self, profile_id: str) -> Optional[Dict]:
        if not _is_safe_id(profile_id):
            logger.warning(f"Unsafe profile_id rejected in get_voice_metadata: {profile_id}")
            return None
        meta_path = self.voices_dir / profile_id / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["id"] = profile_id
            return meta
        except Exception:
            return None

    # -- TTS Generation --------------------------------------------------------

    def generate_tts(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        prefer_clone: bool = False,
        lightweight: bool = False,
        allow_fallback: bool = True,
        progress_callback: Optional[Callable] = None,
        voice_design: Optional[dict] = None,
    ) -> Optional[bytes]:
        self.last_tts_engine = "none"

        if not voice_id and voice_design:
            result = self._generate_designed_voice(text, voice_design)
            if result is not None:
                return result

        if voice_id:
            if not _is_safe_id(voice_id):
                logger.warning(f"Unsafe voice_id rejected in generate_tts: {voice_id}")
                if not allow_fallback:
                    return None
                voice_id = None
            else:
                profile = self.get_voice_metadata(voice_id) or {}
                if not language or language == "auto":
                    language = profile.get("language", "en")

        # 1. Clone path
        if prefer_clone and voice_id:
            result = self._generate_f5_tts(text, voice_id, progress_callback, lightweight=lightweight)
            if result is not None:
                self.last_tts_engine = "f5_tts"
                return result
            if not allow_fallback:
                return None

            if lightweight:
                result = self._generate_omnivoice_lite(text, voice_id, language, progress_callback)
                if result is not None:
                    self.last_tts_engine = "omnivoice_lite"
                return result

            result = self._generate_omnivoice_full(text, voice_id, language, progress_callback)
            if result is not None:
                self.last_tts_engine = "omnivoice"
                return result

            result = self._generate_cloned(text, voice_id, language, progress_callback)
            if result is not None:
                self.last_tts_engine = "openvoice"
                return result
            if not allow_fallback:
                return None
        # 2. Piper (Forced for everything)
        piper_result = self._generate_piper(text)
        if piper_result is not None:
            self.last_tts_engine = "piper"
            return piper_result

        logger.error("Real voice TTS failed.")
        return None

    def _generate_designed_voice(self, text: str, design: dict) -> Optional[bytes]:
        """Synthesize speech using designed voice attributes (zero-shot synthesis via Kokoro)."""
        if not isinstance(design, dict):
            logger.warning("Invalid voice_design type, must be a dict.")
            return None

        gender = str(design.get("gender", "female")).lower()
        if gender not in ["male", "female"]:
            gender = "female"

        accent = str(design.get("accent", "us")).lower()
        if accent not in ["us", "uk", "jp"]:
            accent = "us"

        try:
            speed = float(design.get("speed", 1.0))
        except (ValueError, TypeError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))
        
        # Map gender + accent to Kokoro voice
        if accent == "uk":
            voice = "bf_emma" if gender == "female" else "bm_george"
            lang = "en-gb"
        elif accent == "jp":
            voice = "jf_alpha"
            lang = "ja"
        else: # US default
            voice = "af_sarah" if gender == "female" else "am_adam"
            lang = "en-us"
            
        try:
            from clone_engine import _get_kokoro
            kokoro = _get_kokoro()
            if kokoro is not None:
                speed = max(0.5, min(2.0, speed))
                samples, sample_rate = kokoro.create(text[:500], voice=voice, speed=speed, lang=lang)
                if samples is not None and len(samples) > 0:
                    self.last_tts_engine = "kokoro"
                    return _float_audio_to_wav_bytes(samples, sample_rate)
        except Exception as exc:
            logger.warning(f"Voice design synthesis via Kokoro failed: {exc}. Trying Piper design...")
            
        # Fallback to Piper with adjusted speed/pitch
        try:
            length_scale = 1.0 / speed if speed > 0 else 1.0
            noise_scale = 0.667
            import piper_tts
            if piper_tts.is_available():
                self.last_tts_engine = "piper"
                return piper_tts.synthesize(
                    text=text[:500],
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                )
        except Exception:
            pass
            
        return None

    # -- F5-TTS ----------------------------------------------------------------

    def _generate_f5_tts(
        self,
        text: str,
        voice_id: str,
        progress_callback: Optional[Callable] = None,
        lightweight: bool = False,
    ) -> Optional[bytes]:
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                logger.warning(f"No reference audio for voice {voice_id}")
                return None

            f5_cli = self._find_executable("f5-tts_infer-cli")
            if not f5_cli:
                logger.warning("F5-TTS CLI not found. Install with: pip install f5-tts")
                return None

            if progress_callback:
                progress_callback("init", 0, 3, "Starting F5-TTS...")

            with tempfile.TemporaryDirectory(prefix="mimicai_f5_") as tmp_dir:
                out_name = "speech.wav"
                cmd = [
                    f5_cli,
                    "--model", os.environ.get("F5_TTS_MODEL", "F5TTS_v1_Base"),
                    "--ref_audio", ref_path,
                    "--ref_text", "",
                    "--gen_text", text[:500],
                    "--output_dir", tmp_dir,
                    "--output_file", out_name,
                    "--remove_silence",
                    "--nfe_step", os.environ.get(
                        "F5_TTS_NFE_STEP_LIGHT" if lightweight else "F5_TTS_NFE_STEP",
                        "16" if lightweight else "32",
                    ),
                ]

                if progress_callback:
                    progress_callback("generate", 1, 3, "Generating with F5-TTS...")

                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=int(os.environ.get("F5_TTS_TIMEOUT", "600")),
                )
                if proc.returncode != 0:
                    logger.warning(
                        "F5-TTS failed: %s",
                        proc.stderr.decode("utf-8", errors="ignore")[-2000:],
                    )
                    return None

                out_path = Path(tmp_dir) / out_name
                if not out_path.exists():
                    wavs = sorted(Path(tmp_dir).rglob("*.wav"))
                    out_path = wavs[-1] if wavs else out_path
                if not out_path.exists():
                    logger.warning("F5-TTS finished but no WAV output was found")
                    return None

                if progress_callback:
                    progress_callback("done", 3, 3, "F5-TTS generated speech.")

                return out_path.read_bytes()
        except Exception as exc:
            logger.error(f"F5-TTS generation error: {exc}", exc_info=True)
            return None

    def _find_executable(self, name: str) -> Optional[str]:
        path = shutil.which(name)
        if path:
            return path
        local = Path(__file__).parent / "venv" / "bin" / name
        if local.exists():
            return str(local)
        return None

    # -- OmniVoice -------------------------------------------------------------

    def _generate_omnivoice_full(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                logger.warning(f"No reference audio for voice {voice_id}")
                return None

            if progress_callback:
                progress_callback("init", 0, 3, "Loading full OmniVoice...")

            model = _load_omnivoice_full_model()
            if model is None:
                return None

            if progress_callback:
                progress_callback("generate", 1, 3, "Generating with full OmniVoice...")

            audios = model.generate(
                text=text[:500],
                language=language,
                ref_audio=ref_path,
                num_step=16,
                guidance_scale=2.0,
            )
            audio = audios[0] if isinstance(audios, list) else audios
            sample_rate = int(getattr(model, "sampling_rate", 24000) or 24000)

            if progress_callback:
                progress_callback("done", 3, 3, "Full OmniVoice generated speech.")

            return _float_audio_to_wav_bytes(audio, sample_rate)
        except Exception as exc:
            logger.error(f"Full OmniVoice generation error: {exc}", exc_info=True)
            return None

    def _generate_omnivoice_lite(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                logger.warning(f"No reference audio for voice {voice_id}")
                return None

            model = _load_omnivoice_model()
            if model is None:
                return None

            audio = model.generate(
                text=text[:500],
                ref_audio=ref_path,
                language=language,
                quality="balanced",
                progress_callback=progress_callback,
            )
            sample_rate = int(getattr(model, "sampling_rate", 24000) or 24000)
            return _float_audio_to_wav_bytes(audio, sample_rate)
        except Exception as exc:
            logger.error(f"OmniVoice Lite generation error: {exc}", exc_info=True)
            return None

    # -- Piper TTS (primary local engine) --------------------------------------

    def _generate_piper(self, text: str, model_path: Optional[str] = None) -> Optional[bytes]:
        """Synthesize with tuned Piper params.

        Spec params: noise_scale=0.667, length_scale=1.1, noise_w_scale=0.8
        """
        try:
            import piper_tts
            if not piper_tts.is_available(model_path):
                logger.warning(f"Piper model not available for {model_path or 'default'}")
                return None

            return piper_tts.synthesize(
                text=text[:500],
                model_path=model_path,
                noise_scale=0.667,
                length_scale=1.1,
                noise_w_scale=0.8,
            )
        except Exception as exc:
            logger.warning(f"Piper TTS failed: {exc}")
            return None

    # -- Voice Cloning (Piper + OpenVoice) -------------------------------------

    def _generate_cloned(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        """Three-strategy cloning pipeline.

        A. Custom fine-tuned ONNX in piper_models/{voice_id or name}.onnx
        B. Bundled Piper base + OpenVoice ToneColorConverter
        C. Piper direct (no tone cloning) when OpenVoice unavailable
        """
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                logger.warning(f"No reference audio for voice {voice_id}")
                return None

            meta = self.get_voice_metadata(voice_id)
            voice_name = meta.get("name", "") if meta else ""
            piper_models_dir = Path(__file__).parent / "piper_models"

            custom_model_path = None
            id_model = piper_models_dir / f"{voice_id}.onnx"
            name_model = piper_models_dir / f"{voice_name}.onnx"

            if id_model.exists():
                custom_model_path = str(id_model)
                logger.info(f"Found custom Piper model by ID: {custom_model_path}")
            elif name_model.exists():
                custom_model_path = str(name_model)
                logger.info(f"Found custom Piper model by Name: {custom_model_path}")

            # Strategy A: custom fine-tuned Piper model
            if custom_model_path:
                import piper_tts
                if piper_tts.is_available(custom_model_path):
                    return self._generate_piper(text, model_path=custom_model_path)
                else:
                    logger.warning(f"Custom model path {custom_model_path} missing files. Falling back.")

            # Strategy B: OpenVoice ToneColorConverter
            try:
                from clone_engine import clone_voice, is_available as ce_available
                if ce_available():
                    t0 = time.monotonic()
                    # Pre-process reference audio: VAD trim + volume normalise
                    try:
                        import piper_tts
                        clean_ref = piper_tts.preprocess_reference_audio(ref_path)
                    except Exception:
                        clean_ref = ref_path

                    wav_bytes = clone_voice(
                        text=text[:500],
                        ref_audio_path=clean_ref,
                        language=language,
                        progress_callback=progress_callback,
                        piper_model_path=None,
                        skip_openvoice=False,
                    )
                    if wav_bytes:
                        logger.info(f"OpenVoice clone done in {time.monotonic()-t0:.1f}s")
                        return wav_bytes
            except Exception as exc:
                logger.warning(f"OpenVoice clone error: {exc}")

            # Strategy C: Piper direct
            logger.info("OpenVoice unavailable -- using Piper direct (no tone cloning)")
            return self._generate_piper(text)

        except Exception as exc:
            logger.error(f"Clone generation error: {exc}", exc_info=True)
            return None

    # -- espeak-ng -------------------------------------------------------------

    def _generate_espeak(self, text: str, language: str) -> Optional[bytes]:
        espeak_path = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak_path:
            return None
        lang_map = {
            "en": "en", "hi": "hi", "ml": "ml", "ta": "ta", "te": "te",
            "kn": "kn", "bn": "bn", "mr": "mr", "gu": "gu", "pa": "pa",
            "ur": "ur", "fr": "fr-fr", "de": "de", "es": "es", "pt": "pt",
            "ru": "ru", "ja": "ja", "ko": "ko", "zh": "zh", "ar": "ar",
        }
        voice = lang_map.get(language, "en")
        out_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name
            try:
                subprocess.run(
                    [espeak_path, "-v", voice, "-s", "165", "-w", out_path, text[:500]],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=20,
                )
            except subprocess.CalledProcessError:
                if voice != "en":
                    subprocess.run(
                        [espeak_path, "-v", "en", "-s", "165", "-w", out_path, text[:500]],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=20,
                    )
            with open(out_path, "rb") as f:
                return f.read()
        except Exception as exc:
            logger.error(f"espeak fallback error: {exc}")
            return None
        finally:
            if out_path:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

    # -- Audio Utilities -------------------------------------------------------

    def _convert_to_wav(self, input_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert recorded audio to 22050 Hz / Mono / 16-bit WAV.

        Also applies loudness normalisation (-20 dBFS) and VAD silence trimming
        to match Piper/OpenVoice input requirements.
        """
        wav_path = output_dir / "reference.wav"

        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence

            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

            # Normalise loudness to -20 dBFS
            audio = audio.apply_gain(-20.0 - audio.dBFS)

            # Aggressive VAD trim
            chunks = split_on_silence(
                audio, min_silence_len=300, silence_thresh=-40, keep_silence=100,
            )
            if chunks:
                clean = chunks[0]
                for c in chunks[1:]:
                    clean += c
                clean = clean.apply_gain(-20.0 - clean.dBFS)
                audio = clean

            audio.export(str(wav_path), format="wav",
                         parameters=["-acodec", "pcm_s16le"])
            logger.info(f"Converted & preprocessed to WAV via pydub: {wav_path}")
            return wav_path
        except Exception as exc:
            logger.warning(f"pydub conversion/preprocessing failed: {exc}")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            try:
                subprocess.run(
                    [ffmpeg_path, "-y", "-i", str(input_path),
                     "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", str(wav_path)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                logger.info(f"Converted to WAV via ffmpeg: {wav_path}")
                return wav_path
            except Exception as exc:
                logger.warning(f"ffmpeg conversion failed: {exc}")

        return None
