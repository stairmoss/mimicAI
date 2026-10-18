import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
import torch

from clonemodel.memory_utils import free_unused_memory
from clonemodel.preset_manager import list_presets
from clonemodel.voice_config import check_audio_data_quality

logger = logging.getLogger(__name__)


def _is_safe_id(profile_id: Optional[str]) -> bool:
    if not profile_id:
        return False
    import re
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", profile_id))


VOICES_DIR = Path(__file__).parent / "voices"
SHARD_DIR = Path(os.path.expanduser("~/.clonemodel_lite/shards"))

_model_lock = threading.Lock()
_mimic_model = None
_mimic_loaded = False
_mimic_full_model = None
_mimic_full_loaded = False


def _load_MimicAI_model():
    global _mimic_model, _mimic_loaded
    with _model_lock:
        if _mimic_loaded:
            return _mimic_model
        try:
            from clonemodel.lite.mimicvoice_lite import MimicVoiceLite
            if not SHARD_DIR.exists():
                _mimic_loaded = True
                return None
            t0 = time.monotonic()
            _mimic_model = MimicVoiceLite.from_pretrained(
                shard_dir=str(SHARD_DIR), max_memory_gb=3.0,
                device="cpu", enable_prefetch=True,
            )
            logger.info(f"MimicVoice Lite ready in {time.monotonic() - t0:.1f}s")
        except Exception as exc:
            logger.error(f"Failed loading MimicVoice Lite: {exc}", exc_info=True)
            _mimic_model = None
        finally:
            _mimic_loaded = True
        return _mimic_model


def _load_MimicAI_full_model():
    global _mimic_full_model, _mimic_full_loaded
    with _model_lock:
        if _mimic_full_loaded:
            return _mimic_full_model
        try:
            from clonemodel import MimicVoice

            m_name = os.environ.get("MimicAI_MODEL", "k2-fsa/OmniVoice")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            t0 = time.monotonic()
            _mimic_full_model = MimicVoice.from_pretrained(
                m_name,
                device_map=device,
                torch_dtype=dtype,
                load_asr=False,
                local_files_only=True,
            )
            logger.info(f"Full MimicVoice loaded in {time.monotonic() - t0:.1f}s")
        except Exception as exc:
            logger.error(f"Failed loading full MimicVoice: {exc}", exc_info=True)
            _mimic_full_model = None
        finally:
            _mimic_full_loaded = True
        return _mimic_full_model


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
    def __init__(self, voices_dir: Optional[str] = None):
        self.voices_dir = Path(voices_dir) if voices_dir else VOICES_DIR
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.last_tts_engine = "none"
        try:
            from clone_engine import preload
            preload()
        except Exception:
            pass

    def list_voices(self) -> List[Dict]:
        profiles = []
        for d in self.voices_dir.iterdir():
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
        profiles.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return profiles

    def create_voice(self, audio_data: bytes, name: str, language: str = "en", file_ext: str = "webm", gender: str = "male") -> Dict:
        profile_id = 'v_' + uuid.uuid4().hex[:10]
        profile_dir = self.voices_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        audio_path = profile_dir / f"original_reference.{file_ext}"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        wav_path = self._convert_to_wav(audio_path, profile_dir)
        if wav_path is None:
            raise RuntimeError("Audio conversion failed. Make sure ffmpeg is installed.")

        ref_text = ""
        try:
            import whisper
            model = whisper.load_model("base")
            res = model.transcribe(str(wav_path))
            ref_text = res.get("text", "").strip()
        except Exception as exc:
            logger.warning(f"Whisper transcription skipped: {exc}")

        meta = {
            "name": name,
            "language": language,
            "gender": gender or "male",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_file": wav_path.name,
            "original_file": audio_path.name,
            "clone_ready": True,
            "transcript": ref_text,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(profile_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        meta["id"] = profile_id
        return meta

    def delete_voice(self, profile_id: str) -> bool:
        if not _is_safe_id(profile_id):
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
        return True

    def get_reference_audio_path(self, profile_id: str) -> Optional[str]:
        if not _is_safe_id(profile_id):
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
            res = self._generate_remote_design(text, voice_design)
            if res is not None:
                return res
            res = self._generate_designed_voice(text, voice_design)
            if res is not None:
                return res

        if voice_id:
            if not _is_safe_id(voice_id):
                if not allow_fallback:
                    return None
                voice_id = None
            else:
                profile = self.get_voice_metadata(voice_id) or {}
                if not language or language == "auto":
                    language = profile.get("language", "en")

        if prefer_clone and voice_id:
            res = self._generate_remote_clone(text, voice_id, language=language)
            if res is not None:
                return res

            ref_path = self.get_reference_audio_path(voice_id)
            if ref_path and os.path.exists(ref_path):
                # A CPU-only installation is still able to clone a voice.  Do not
                # silently skip it just because CUDA is unavailable: doing so
                # turns every failed remote request into a generic TTS fallback.
                try:
                    from clone_engine import clone_voice
                    profile = self.get_voice_metadata(voice_id) or {}
                    gender = profile.get("gender", "male")
                    local_clone = clone_voice(
                        text=text,
                        ref_audio_path=ref_path,
                        language=language,
                        gender=gender,
                        progress_callback=progress_callback,
                    )
                    if local_clone:
                        self.last_tts_engine = "openvoice"
                        return local_clone
                except Exception as exc:
                    logger.warning(f"Local clone error: {exc}")

            if not allow_fallback:
                return None

        # Gender-matched fallback so male speakers NEVER hear a female voice
        profile = self.get_voice_metadata(voice_id) or {} if voice_id else {}
        gender = profile.get("gender", "male")
        try:
            from clone_engine import _get_kokoro
            kokoro = _get_kokoro()
            if kokoro is not None:
                is_male = str(gender).lower() in ("male", "m")
                voice_name = "am_adam" if is_male else "af_sarah"
                samples, sr = kokoro.create(text[:500], voice=voice_name, speed=1.0, lang="en-us")
                if samples is not None and len(samples) > 0:
                    self.last_tts_engine = "kokoro"
                    return _float_audio_to_wav_bytes(samples, sr)
        except Exception as exc:
            logger.warning(f"Kokoro gender fallback failed: {exc}")

        piper_res = self._generate_piper(text)
        if piper_res is not None:
            self.last_tts_engine = "piper"
            return piper_res

        return None

    def _generate_remote_clone(self, text: str, voice_id: str, language: str = "en") -> Optional[bytes]:
        ref_path = self.get_reference_audio_path(voice_id)
        if not ref_path or not os.path.exists(ref_path):
            return None

        profile = self.get_voice_metadata(voice_id) or {}
        gender = profile.get("gender", "male")

        try:
            from gradio_client import Client, handle_file
            if not hasattr(self, '_gradio_client') or self._gradio_client is None:
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if not hf_token:
                    try:
                        import huggingface_hub
                        hf_token = huggingface_hub.get_token()
                    except Exception:
                        pass
                self._gradio_client = Client("stairmoss/MimicAI", token=hf_token)

            client = self._gradio_client
            try:
                job = client.submit(
                    ref_audio=handle_file(str(ref_path)),
                    text=text[:500],
                    gender=gender,
                    api_name="/clone"
                )
                # ZeroGPU cold starts and queued jobs routinely take longer than
                # three seconds.  The previous value made cloning fail almost
                # deterministically before inference could begin.
                res_path = job.result(timeout=int(os.environ.get("REMOTE_CLONE_TIMEOUT", "45")))
            except Exception as e:
                if "gender" in str(e).lower():
                    job = client.submit(
                        ref_audio=handle_file(str(ref_path)),
                        text=text[:500],
                        api_name="/clone"
                    )
                    res_path = job.result(timeout=int(os.environ.get("REMOTE_CLONE_TIMEOUT", "45")))
                else:
                    raise
            if res_path and os.path.exists(res_path):
                with open(res_path, "rb") as f:
                    data = f.read()
                if len(data) > 100:
                    self.last_tts_engine = "remote_zerogpu"
                    return data
        except Exception as exc:
            logger.warning(f"Remote clone via Gradio client: {exc}")

        # Keep the REST endpoint as a second remote route.  It is particularly
        # useful when the Gradio client cannot negotiate a queued ZeroGPU job.
        remote_url = os.environ.get("REMOTE_VOICE_API", "https://stairmoss-mimicai.hf.space").rstrip("/")
        if remote_url:
            try:
                headers = {}
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                with open(ref_path, "rb") as f:
                    files = {"audio": (os.path.basename(ref_path), f, "audio/wav")}
                    data = {"text": text[:500], "language": language, "gender": gender}
                    resp = requests.post(
                        f"{remote_url}/voice/clone",
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=int(os.environ.get("REMOTE_CLONE_HTTP_TIMEOUT", "45")),
                    )
                if resp.status_code == 200 and len(resp.content) > 100:
                    self.last_tts_engine = "remote_zerogpu"
                    return resp.content
                logger.warning("Remote clone HTTP returned status %s", resp.status_code)
            except Exception as exc:
                logger.warning(f"Remote clone via HTTP: {exc}")
        return None

    def _generate_remote_design(self, text: str, design: dict) -> Optional[bytes]:
        remote_url = os.environ.get("REMOTE_VOICE_API", "https://stairmoss-mimicai.hf.space").rstrip("/")
        if not remote_url:
            return None

        try:
            payload = {
                "text": text[:500],
                "gender": str(design.get("gender", "female")),
                "accent": str(design.get("accent", "us")),
                "speed": float(design.get("speed", 1.0)),
                "pitch": float(design.get("pitch", 1.0)),
            }
            resp = requests.post(f"{remote_url}/voice/design", json=payload, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 100:
                self.last_tts_engine = "remote_zerogpu"
                return resp.content
        except Exception as exc:
            logger.warning(f"Remote voice design fallback: {exc}")
        return None

    def _generate_designed_voice(self, text: str, design: dict) -> Optional[bytes]:
        if not isinstance(design, dict):
            return None

        gender = str(design.get("gender", "female")).lower()
        if gender not in ["male", "female"]:
            gender = "female"

        accent = str(design.get("accent", "us")).lower()
        if accent == "in":
            accent = "us"
        if accent not in ["us", "uk", "jp"]:
            accent = "us"

        try:
            speed = float(design.get("speed", 1.0))
        except (ValueError, TypeError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))

        if accent == "uk":
            voice = "bf_emma" if gender == "female" else "bm_george"
            lang = "en-gb"
        elif accent == "jp":
            voice = "jf_alpha"
            lang = "ja"
        else:
            voice = "af_sarah" if gender == "female" else "am_adam"
            lang = "en-us"

        try:
            from clone_engine import _get_kokoro
            kokoro = _get_kokoro()
            if kokoro is not None:
                samples, sr = kokoro.create(text[:500], voice=voice, speed=speed, lang=lang)
                if samples is not None and len(samples) > 0:
                    self.last_tts_engine = "kokoro"
                    return _float_audio_to_wav_bytes(samples, sr)
        except Exception as exc:
            logger.warning(f"Kokoro design failed: {exc}, trying piper")

        try:
            scale = 1.0 / speed if speed > 0 else 1.0
            import piper_tts
            if piper_tts.is_available():
                self.last_tts_engine = "piper"
                return piper_tts.synthesize(text=text[:500], noise_scale=0.667, length_scale=scale)
        except Exception:
            pass

        return None

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
                return None

            meta = self.get_voice_metadata(voice_id) or {}
            ref_text = meta.get("transcript")
            if ref_text is None:
                try:
                    import whisper
                    model = whisper.load_model("base")
                    res = model.transcribe(str(ref_path))
                    ref_text = res.get("text", "").strip()
                    meta["transcript"] = ref_text
                    with open(self.voices_dir / voice_id / "metadata.json", "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception as exc:
                    logger.warning(f"Whisper transcript fallback failed: {exc}")
                    ref_text = ""

            cli = self._find_executable("f5-tts_infer-cli")
            if not cli:
                return None

            if progress_callback:
                progress_callback("init", 0, 3, "Starting F5-TTS...")

            with tempfile.TemporaryDirectory(prefix="mimic_f5_") as tmp_dir:
                out_name = "speech.wav"
                cmd = [
                    cli,
                    "--model", os.environ.get("F5_TTS_MODEL", "F5TTS_v1_Base"),
                    "--ref_audio", ref_path,
                    "--ref_text", ref_text,
                    "--gen_text", text[:500],
                    "--output_dir", tmp_dir,
                    "--output_file", out_name,
                    "--remove_silence",
                    "--nfe_step", os.environ.get(
                        "F5_TTS_NFE_STEP_LIGHT" if lightweight else "F5_TTS_NFE_STEP",
                        "16" if lightweight else "32",
                    ),
                ]
                if not torch.cuda.is_available():
                    cmd.extend(["--device", "cpu"])

                if progress_callback:
                    progress_callback("generate", 1, 3, "Generating with F5-TTS...")

                import multiprocessing
                env = os.environ.copy()
                cores = str(max(1, multiprocessing.cpu_count()))
                env["OMP_NUM_THREADS"] = cores
                env["MKL_NUM_THREADS"] = cores

                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=int(os.environ.get("F5_TTS_TIMEOUT", "1800")),
                )
                if proc.returncode != 0:
                    logger.warning(f"F5-TTS exited with code {proc.returncode}")
                    return None

                out_path = Path(tmp_dir) / out_name
                if not out_path.exists():
                    wavs = sorted(Path(tmp_dir).rglob("*.wav"))
                    out_path = wavs[-1] if wavs else out_path
                if not out_path.exists():
                    return None

                if progress_callback:
                    progress_callback("done", 3, 3, "F5-TTS generated speech.")

                return out_path.read_bytes()
        except Exception as exc:
            logger.error(f"F5-TTS generation error: {exc}", exc_info=True)
            return None

    def _find_executable(self, name: str) -> Optional[str]:
        p = shutil.which(name)
        if p:
            return p
        local = Path(__file__).parent / "venv" / "bin" / name
        if local.exists():
            return str(local)
        return None

    def _generate_MimicAI_full(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                return None

            if progress_callback:
                progress_callback("init", 0, 3, "Loading full MimicVoice...")

            model = _load_MimicAI_full_model()
            if model is None:
                return None

            if progress_callback:
                progress_callback("generate", 1, 3, "Generating with full MimicVoice...")

            audios = model.generate(
                text=text[:500],
                language=language,
                ref_audio=ref_path,
                num_step=16,
                guidance_scale=2.0,
            )
            audio = audios[0] if isinstance(audios, list) else audios
            sr = int(getattr(model, "sampling_rate", 24000) or 24000)

            if progress_callback:
                progress_callback("done", 3, 3, "Full MimicVoice generated speech.")

            return _float_audio_to_wav_bytes(audio, sr)
        except Exception as exc:
            logger.error(f"Full MimicVoice error: {exc}", exc_info=True)
            return None

    def _generate_MimicAI_lite(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                return None

            model = _load_MimicAI_model()
            if model is None:
                return None

            audio = model.generate(
                text=text[:500],
                ref_audio=ref_path,
                language=language,
                quality="balanced",
                progress_callback=progress_callback,
            )
            sr = int(getattr(model, "sampling_rate", 24000) or 24000)
            return _float_audio_to_wav_bytes(audio, sr)
        except Exception as exc:
            logger.error(f"MimicVoice Lite error: {exc}", exc_info=True)
            return None

    def _generate_piper(self, text: str, model_path: Optional[str] = None) -> Optional[bytes]:
        try:
            import piper_tts
            if not piper_tts.is_available(model_path):
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

    def _generate_cloned(
        self,
        text: str,
        voice_id: str,
        language: str,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        try:
            ref_path = self.get_reference_audio_path(voice_id)
            if not ref_path:
                return None

            meta = self.get_voice_metadata(voice_id)
            voice_name = meta.get("name", "") if meta else ""
            piper_dir = Path(__file__).parent / "piper_models"

            custom_path = None
            id_model = piper_dir / f"{voice_id}.onnx"
            name_model = piper_dir / f"{voice_name}.onnx"

            if id_model.exists():
                custom_path = str(id_model)
            elif name_model.exists():
                custom_path = str(name_model)

            if custom_path:
                import piper_tts
                if piper_tts.is_available(custom_path):
                    return self._generate_piper(text, model_path=custom_path)

            try:
                from clone_engine import clone_voice, is_available as ce_avail
                if ce_avail():
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
                        return wav_bytes
            except Exception as exc:
                logger.warning(f"OpenVoice clone error: {exc}")

            return self._generate_piper(text)

        except Exception as exc:
            logger.error(f"Clone generation error: {exc}", exc_info=True)
            return None

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

    def _convert_to_wav(self, input_path: Path, output_dir: Path) -> Optional[Path]:
        wav_path = output_dir / "reference.wav"

        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence

            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

            audio = audio.apply_gain(-20.0 - audio.dBFS)

            chunks = split_on_silence(
                audio, min_silence_len=300, silence_thresh=-40, keep_silence=100,
            )
            if chunks:
                clean = chunks[0]
                for c in chunks[1:]:
                    clean += c
                clean = clean.apply_gain(-20.0 - clean.dBFS)
                audio = clean

            audio.export(str(wav_path), format="wav", parameters=["-acodec", "pcm_s16le"])
            return wav_path
        except Exception as exc:
            logger.warning(f"pydub wav conversion failed: {exc}")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            try:
                subprocess.run(
                    [ffmpeg_path, "-y", "-i", str(input_path),
                     "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", str(wav_path)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return wav_path
            except Exception as exc:
                logger.warning(f"ffmpeg conversion failed: {exc}")

        return None
