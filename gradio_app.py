import os
import tempfile
import time
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn

try:
    import spaces
except ImportError:
    class _MockSpaces:
        def GPU(self, fn=None, duration=None):
            if fn is None:
                return lambda f: f
            return fn
    spaces = _MockSpaces()

from voice_manager import VoiceManager
from clone_engine import clone_voice
import piper_tts

vm = VoiceManager()

api_app = FastAPI(title="MimicAI Headless Voice API", version="1.0.0")

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@spaces.GPU
def run_voice_clone(clean_ref_path: str, text: str, language: str = "en") -> bytes:
    audio_bytes = clone_voice(
        text=text.strip()[:500],
        ref_audio_path=clean_ref_path,
        language=language,
    )
    if not audio_bytes:
        audio_bytes = piper_tts.synthesize(text=text.strip()[:500])
    return audio_bytes


@spaces.GPU
def run_voice_design(text: str, gender: str, accent: str, speed: float, pitch: float) -> bytes:
    design = {
        "gender": gender.lower(),
        "accent": "us" if "american" in accent.lower() else ("in" if "indian" in accent.lower() else ("uk" if "british" in accent.lower() else "jp")),
        "speed": float(speed),
        "pitch": float(pitch),
    }
    audio_bytes = vm._generate_designed_voice(text.strip()[:500], design)
    if not audio_bytes:
        audio_bytes = piper_tts.synthesize(text=text.strip()[:500], length_scale=1.0 / max(0.5, speed))
    return audio_bytes


@api_app.get("/")
@api_app.get("/health")
def health_check():
    return {
        "service": "MimicAI Headless API",
        "status": "online",
        "version": "1.0.0",
        "hardware": "ZeroGPU enabled" if hasattr(spaces, "zero") else "CPU",
        "endpoints": {
            "clone": {
                "method": "POST",
                "path": "/api/clone",
                "type": "multipart/form-data",
                "params": ["audio (file)", "text (string)", "language (optional)"]
            },
            "design": {
                "method": "POST",
                "path": "/api/design",
                "type": "application/json or form",
                "params": ["text", "gender", "accent", "speed", "pitch"]
            },
            "tts": {
                "method": "POST",
                "path": "/api/tts",
                "type": "application/json or form",
                "params": ["text", "speed"]
            },
            "health": {
                "method": "GET",
                "path": "/health"
            }
        }
    }


@api_app.post("/api/clone")
async def api_clone(
    audio: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("en"),
):
    if not text or not text.strip():
        return JSONResponse({"error": "text parameter is required"}, status_code=400)

    suffix = os.path.splitext(audio.filename or "sample.wav")[1] or ".wav"
    raw_tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await audio.read()
        raw_tmp.write(content)
        raw_tmp.close()

        clean_ref = piper_tts.preprocess_reference_audio(raw_tmp.name)
        audio_bytes = run_voice_clone(clean_ref, text.strip(), language)

        if not audio_bytes:
            return JSONResponse({"error": "Voice cloning failed"}, status_code=500)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="clone.wav"'},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        try:
            if os.path.exists(raw_tmp.name):
                os.remove(raw_tmp.name)
        except Exception:
            pass


@api_app.post("/api/design")
async def api_design(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            form = await request.form()
            data = dict(form)

        text = str(data.get("text", "")).strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)

        gender = str(data.get("gender", "female"))
        accent = str(data.get("accent", "us"))
        speed = float(data.get("speed", 1.0))
        pitch = float(data.get("pitch", 1.0))

        audio_bytes = run_voice_design(text, gender, accent, speed, pitch)
        if not audio_bytes:
            return JSONResponse({"error": "Voice design synthesis failed"}, status_code=500)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="design.wav"'},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@api_app.post("/api/tts")
async def api_tts(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
        else:
            form = await request.form()
            data = dict(form)

        text = str(data.get("text", "")).strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)

        speed = float(data.get("speed", 1.0))
        audio_bytes = piper_tts.synthesize(text=text[:500], length_scale=1.0 / max(0.5, speed))

        if not audio_bytes:
            return JSONResponse({"error": "TTS synthesis failed"}, status_code=500)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="tts.wav"'},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


with gr.Blocks(title="MimicAI Headless API") as dummy_block:
    gr.Markdown("MimicAI Headless API is online.")

app = gr.mount_gradio_app(api_app, dummy_block, path="/_gradio")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
