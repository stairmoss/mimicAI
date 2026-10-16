#!/usr/bin/env python3
import json
import logging
import os
import queue
import threading
import time
import traceback
import uuid

import requests
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from voice_manager import VoiceManager
from clonemodel.profile_validator import validate_profile_name, validate_audio_file_size
from gemini_helper import is_gemini_available, GEMINI_ENDPOINT, GEMINI_MODEL, GEMINI_API_KEY
from clonemodel.memory_utils import free_unused_memory

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_URL = os.environ.get(
    'GEMINI_API_URL',
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
HACKCLUB_API_URL = "https://ai.hackclub.com/proxy/v1/chat/completions"
HACKCLUB_API_KEY = os.environ.get("HACKCLUB_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", 'qwen/qwen3-32b')

SYSTEM_PROMPT = (
    "You are MimicAI, a friendly and concise AI assistant. "
    "Keep responses short and helpful — 2-3 sentences max unless asked for detail. "
    "Be warm but not overly chatty. Use simple, clear language. "
    "Do NOT include any thinking or reasoning tags in your response."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mimicai")

FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1" or os.environ.get("FLASK_ENV", "production") == "development"


def _error_response(exc, message):
    logger.error(f"{message}: {exc}", exc_info=True)
    err_msg = str(exc) if FLASK_DEBUG else f"Internal server error: {message.lower()}"
    return jsonify({"error": err_msg}), 500


FALLBACK_LANGUAGES = [
    {"id": "en", "name": "English"},
    {"id": "hi", "name": "Hindi"},
    {"id": "ml", "name": "Malayalam"},
    {"id": "ta", "name": "Tamil"},
    {"id": "te", "name": "Telugu"},
    {"id": "kn", "name": "Kannada"},
    {"id": "bn", "name": "Bengali"},
    {"id": "mr", "name": "Marathi"},
    {"id": "gu", "name": "Gujarati"},
    {"id": "pa", "name": "Punjabi"},
    {"id": "ur", "name": "Urdu"},
    {"id": "fr", "name": "French"},
    {"id": "de", "name": "German"},
    {"id": "es", "name": "Spanish"},
    {"id": "pt", "name": "Portuguese"},
    {"id": "ru", "name": "Russian"},
    {"id": "ja", "name": "Japanese"},
    {"id": "ko", "name": "Korean"},
    {"id": "zh", "name": "Chinese"},
    {"id": "ar", "name": "Arabic"},
]

app = Flask(__name__)
cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
if cors_origins != "*":
    cors_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": cors_origins}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

voice_manager = VoiceManager()

jobs = {}
jobs_lock = threading.Lock()


def _new_job(job_id: str):
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "progress": [],
            "audio": None,
            "engine": None,
            "error": None,
            "queue": queue.Queue(),
            'created_at': time.time(),
        }


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            job[k] = v
        try:
            job["queue"].put_nowait(kwargs)
        except queue.Full:
            pass


def _get_job(job_id: str):
    with jobs_lock:
        return jobs.get(job_id)


def _get_job_snapshot(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return None
        return {
            "status": job.get("status"),
            "engine": job.get("engine"),
            "error": job.get("error"),
            "last_progress": job.get("last_progress", {}),
        }


def _cleanup_old_jobs():
    with jobs_lock:
        now = time.time()
        stale = [jid for jid, j in jobs.items() if j.get('created_at', now) < now - 300]
        for jid in stale:
            del jobs[jid]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "No message provided"}), 400

    user_msg = data["message"].strip()
    history = data.get("history", [])

    if history:
        last = history[-1]
        if last.get("role") == "user" and (last.get("content") or "").strip() == user_msg:
            history = history[:-1]

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-10:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_msg})

    def _err_chunk(msg: str):
        payload = json.dumps({"choices": [{"delta": {"content": msg}}]}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    def stream():
        try:
            target_url = GEMINI_API_URL if os.environ.get("GEMINI_API_KEY") else HACKCLUB_API_URL
            api_key = os.environ.get("GEMINI_API_KEY") or HACKCLUB_API_KEY
            model_name = GEMINI_MODEL if os.environ.get("GEMINI_API_KEY") else CHAT_MODEL
            resp = requests.post(
                target_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": msgs,
                    "stream": True,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
                stream=True,
                timeout=45,
            )

            if resp.status_code != 200:
                yield from _err_chunk(f"API error {resp.status_code}. Please try again.")
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                raw = line[6:] if line.startswith("data: ") else line
                if raw.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if not content:
                        continue
                    try:
                        content = content.encode("latin-1").decode("utf-8")
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        pass
                    clean = json.dumps({"choices": [{"delta": {"content": content}}]}, ensure_ascii=False)
                    yield f"data: {clean}\n\n"
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

        except requests.exceptions.Timeout:
            yield from _err_chunk("The chat service timed out. Try again in a moment.")
        except requests.exceptions.ConnectionError:
            yield from _err_chunk("Cannot reach the chat service. Check your connection.")
        except Exception:
            logger.error(f"Chat stream error:\n{traceback.format_exc()}")
            yield from _err_chunk("The chat service hit an error. Please try again.")

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()[:500]
    voice_id = data.get("voice_id") or None
    lang = data.get("language", "en")
    prefer_clone = bool(data.get("prefer_clone"))
    lightweight = bool(data.get("lightweight"))
    allow_fallback = not bool(data.get("strict_clone"))
    design = data.get("voice_design") or None

    try:
        audio = voice_manager.generate_tts(
            text=text,
            voice_id=voice_id,
            language=lang,
            prefer_clone=prefer_clone,
            lightweight=lightweight,
            allow_fallback=allow_fallback if not prefer_clone else True,
            voice_design=design,
        )
    except Exception as exc:
        return _error_response(exc, "TTS generation failed")

    if not audio:
        return jsonify({"error": "TTS generation failed"}), 500

    return Response(
        audio,
        mimetype=_detect_mimetype(audio),
        headers={"X-TTS-Engine": voice_manager.last_tts_engine},
    )


@app.route("/api/tts/async", methods=["POST"])
def tts_async_start():
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()[:500]
    voice_id = data.get("voice_id") or None
    lang = data.get("language", "en")
    prefer_clone = bool(data.get("prefer_clone"))
    lightweight = bool(data.get("lightweight"))
    allow_fallback = not bool(data.get("strict_clone"))
    design = data.get("voice_design") or None

    job_id = uuid.uuid4().hex[:16]
    _new_job(job_id)

    def worker():
        def progress(phase, step, total, msg):
            _update_job(job_id, status="running", last_progress={"phase": phase, "step": step, "total": total, "msg": msg})

        try:
            _update_job(job_id, status="running")
            audio = voice_manager.generate_tts(
                text=text,
                voice_id=voice_id,
                language=lang,
                prefer_clone=prefer_clone,
                lightweight=lightweight,
                allow_fallback=allow_fallback,
                progress_callback=progress,
                voice_design=design,
            )
            if audio:
                _update_job(job_id, status="done", audio=audio, engine=voice_manager.last_tts_engine)
            else:
                _update_job(job_id, status="error", error="TTS generation failed")
        except Exception as exc:
            logger.error(f"Async TTS worker error: {exc}", exc_info=True)
            err_msg = str(exc) if FLASK_DEBUG else "Internal server error during synthesis"
            _update_job(job_id, status="error", error=err_msg)

    free_unused_memory()
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    _cleanup_old_jobs()
    return jsonify({"job_id": job_id})


@app.route("/api/tts/async/<job_id>/status")
def tts_async_status(job_id):
    def event_stream():
        start = time.time()
        while True:
            job = _get_job_snapshot(job_id)
            if job is None:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                return

            status = job.get("status", "pending")
            prog = job.get("last_progress", {})
            payload = {"status": status, "progress": prog, "timestamp": time.time()}

            if status == "done":
                payload["engine"] = job.get("engine", "unknown")
                yield f"data: {json.dumps(payload)}\n\n"
                return
            elif status == "error":
                payload["error"] = job.get("error", "Unknown error")
                yield f"data: {json.dumps(payload)}\n\n"
                return
            else:
                yield f"data: {json.dumps(payload)}\n\n"

            if time.time() - start > 180:
                yield f"data: {json.dumps({'status': 'timeout'})}\n\n"
                return

            time.sleep(1.0)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/tts/async/<job_id>/audio")
def tts_async_audio(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job.get("status") != "done":
            return jsonify({"error": "Job not ready", "status": job.get("status")}), 202
        audio = job.get("audio")
        engine = job.get("engine", "unknown")

    if not audio:
        return jsonify({"error": "No audio data"}), 500

    return Response(
        audio,
        mimetype=_detect_mimetype(audio),
        headers={"X-TTS-Engine": engine},
    )


@app.route("/api/voices", methods=["GET"])
def list_voices():
    return jsonify({"voices": voice_manager.list_voices()})


@app.route("/api/voices/record", methods=["POST"])
def record_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    name = request.form.get("name", "Untitled Voice").strip()
    lang = request.form.get("language", "en").strip()

    valid_name, name_res = validate_profile_name(name)
    if not valid_name:
        return jsonify({"error": name_res}), 400
    name = name_res

    data = audio_file.read()
    valid_sz, sz_err = validate_audio_file_size(len(data))
    if not valid_sz:
        return jsonify({"error": sz_err}), 400

    fn = audio_file.filename or "recording.webm"
    ext = fn.rsplit(".", 1)[-1] if "." in fn else "webm"

    try:
        profile = voice_manager.create_voice(audio_data=data, name=name, language=lang, file_ext=ext)
        return jsonify({"success": True, "voice": profile})
    except Exception as exc:
        return _error_response(exc, "Voice creation failed")


@app.route("/api/voices/<voice_id>/delete", methods=["DELETE"])
def delete_voice(voice_id):
    if voice_manager.delete_voice(voice_id):
        return jsonify({"success": True})
    return jsonify({"error": "Voice not found"}), 404


@app.route("/api/voices/<voice_id>/preview", methods=["POST"])
def preview_voice(voice_id):
    profile = voice_manager.get_voice_metadata(voice_id)
    if not profile:
        return jsonify({"error": "Voice not found"}), 404

    data = request.get_json(silent=True) or {}
    lang = data.get("language") or profile.get("language", "en")
    text = (data.get("text") or f"Hello, I am {profile.get('name', 'your assistant')}.").strip()[:180]

    try:
        audio = voice_manager.generate_tts(
            text=text,
            voice_id=voice_id,
            language=lang,
            prefer_clone=True,
            allow_fallback=True,
        )
    except Exception as exc:
        return _error_response(exc, "Voice preview failed")

    if not audio:
        return jsonify({"error": "Voice preview failed"}), 500

    return Response(
        audio,
        mimetype=_detect_mimetype(audio),
        headers={"X-TTS-Engine": voice_manager.last_tts_engine},
    )


@app.route("/api/languages", methods=["GET"])
def list_languages():
    return jsonify({"languages": FALLBACK_LANGUAGES, "total": len(FALLBACK_LANGUAGES)})


@app.route("/api/status", methods=["GET"])
def get_status():
    status = {}

    try:
        import piper_tts
        status["gemini"] = {
            "available": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "loaded": bool(os.environ.get("GEMINI_API_KEY", "").strip())
        }
        status["piper"] = {
            "available": piper_tts.is_available(),
            "loaded": getattr(piper_tts, "_loaded_model_path", None) is not None
        }
    except Exception:
        status["gemini"] = {
            "available": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "loaded": bool(os.environ.get("GEMINI_API_KEY", "").strip())
        }
        status["piper"] = {"available": False, "loaded": False}

    try:
        import kokoro_onnx
        m_path = "/mnt/18A660FBA660DB30/voiceclone_AI/mimicAI/kokoro_models/kokoro-v1.0.int8.onnx"
        v_path = "/mnt/18A660FBA660DB30/voiceclone_AI/mimicAI/kokoro_models/voices-v1.0.bin"
        from clone_engine import _kokoro_model
        status["kokoro"] = {
            "available": os.path.exists(m_path) and os.path.exists(v_path),
            "loaded": _kokoro_model is not None
        }
    except Exception:
        status["kokoro"] = {"available": False, "loaded": False}

    try:
        from clone_engine import is_available as ce_avail
        status["openvoice"] = {
            "available": ce_avail(),
            "loaded": True
        }
    except Exception:
        status["openvoice"] = {"available": False, "loaded": False}

    try:
        from clonemodel import MimicVoice
        from voice_manager import _MimicAI_full_model
        status["mimicvoice"] = {
            "available": True,
            "loaded": _MimicAI_full_model is not None
        }
    except Exception:
        status["mimicvoice"] = {"available": False, "loaded": False}

    try:
        from clonemodel.lite.mimicvoice_lite import MimicVoiceLite
        from voice_manager import _MimicAI_model
        status["mimicvoice_lite"] = {
            "available": True,
            "loaded": _MimicAI_model is not None
        }
    except Exception:
        status["mimicvoice_lite"] = {"available": False, "loaded": False}

    return jsonify(status)


def _detect_mimetype(data: bytes) -> str:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if len(data) >= 4 and data[:4] == b"fLaC":
        return "audio/flac"
    if len(data) >= 3 and data[:3] == b"ID3":
        return "audio/mpeg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    return "audio/mpeg"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    print(f"\nMimicAI starting on http://localhost:{port}\n")
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except OSError as exc:
        if "Address already in use" in str(exc):
            alt = port + 1
            print(f"Port {port} busy, trying {alt}...")
            app.run(host=host, port=alt, debug=False, threaded=True)
        else:
            raise
