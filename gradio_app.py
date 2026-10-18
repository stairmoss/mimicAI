import os
import tempfile
import time
from fastapi import UploadFile, File, Form, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

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


@spaces.GPU
def run_voice_clone(ref_audio: str, text: str, gender: str = "male", language: str = "en") -> str:
    if not ref_audio or not text or not text.strip():
        return None
    clean_ref = piper_tts.preprocess_reference_audio(ref_audio)
    audio_bytes = clone_voice(
        text=text.strip()[:500],
        ref_audio_path=clean_ref,
        language=language,
        gender=gender,
    )
    if not audio_bytes:
        audio_bytes = piper_tts.synthesize(text=text.strip()[:500])

    if audio_bytes:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    return None


@spaces.GPU
def run_voice_design(text: str, gender: str = "female", accent: str = "us", speed: float = 1.0, pitch: float = 1.0) -> str:
    if not text or not text.strip():
        return None
    design = {
        "gender": gender.lower(),
        "accent": "us" if "american" in accent.lower() else ("in" if "indian" in accent.lower() else ("uk" if "british" in accent.lower() else "jp")),
        "speed": float(speed),
        "pitch": float(pitch),
    }
    audio_bytes = vm._generate_designed_voice(text.strip()[:500], design)
    if not audio_bytes:
        audio_bytes = piper_tts.synthesize(text=text.strip()[:500], length_scale=1.0 / max(0.5, speed))

    if audio_bytes:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    return None


@spaces.GPU
def run_tts(text: str, speed: float = 1.0) -> str:
    if not text or not text.strip():
        return None
    audio_bytes = piper_tts.synthesize(text=text.strip()[:500], length_scale=1.0 / max(0.5, speed))
    if audio_bytes:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    return None


theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#202124",
    body_text_color="#e8eaed",
    block_background_fill="#292a2d",
    block_border_color="#3c4043",
    input_background_fill="#303134",
    button_primary_background_fill="#8ab4f8",
    button_primary_text_color="#202124",
)

with gr.Blocks(title="MimicAI Headless API", theme=theme) as demo:
    gr.HTML(
        """
        <div style="background:#292a2d;border:1px solid #3c4043;border-radius:14px;padding:24px;margin-top:10px;color:#e8eaed;font-family:system-ui,sans-serif;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <div style="width:12px;height:12px;border-radius:50%;background:#81c995;box-shadow:0 0 10px #81c995;"></div>
            <h2 style="margin:0;font-size:22px;color:#fff;">MimicAI Headless REST API</h2>
            <span style="font-size:11px;background:#303134;color:#8ab4f8;border:1px solid #3c4043;padding:3px 8px;border-radius:12px;">ONLINE</span>
          </div>
          <p style="color:#9aa0a6;margin:0 0 18px 0;font-size:14px;">
            High-performance zero-shot voice cloning and neural TTS API running on Hugging Face ZeroGPU.
          </p>
          <div style="background:#202124;border:1px solid #3c4043;border-radius:10px;padding:16px;font-family:monospace;font-size:13px;color:#8ab4f8;line-height:1.7;">
            <div><b>POST</b> /voice/clone &nbsp;&nbsp;<span style="color:#9aa0a6;">(multipart/form-data: audio=@sample.wav, text="...")</span></div>
            <div><b>POST</b> /voice/design <span style="color:#9aa0a6;">(JSON: {"text":"...", "gender":"female", "accent":"us"})</span></div>
            <div><b>POST</b> /voice/tts &nbsp;&nbsp;&nbsp;<span style="color:#9aa0a6;">(JSON: {"text":"...", "speed":1.0})</span></div>
            <div><b>GET</b> &nbsp;/health &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#9aa0a6;">(Health check and system status)</span></div>
          </div>
        </div>
        """
    )

    with gr.Row(visible=False):
        c_audio = gr.Audio(type="filepath")
        c_text = gr.Textbox()
        c_gender = gr.Textbox(value="male")
        c_out = gr.Audio()
        c_btn = gr.Button()
        c_btn.click(fn=run_voice_clone, inputs=[c_audio, c_text, c_gender], outputs=[c_out], api_name="clone")

        d_text = gr.Textbox()
        d_gender = gr.Textbox(value="female")
        d_accent = gr.Textbox(value="us")
        d_speed = gr.Number(value=1.0)
        d_pitch = gr.Number(value=1.0)
        d_out = gr.Audio()
        d_btn = gr.Button()
        d_btn.click(fn=run_voice_design, inputs=[d_text, d_gender, d_accent, d_speed, d_pitch], outputs=[d_out], api_name="design")

        t_text = gr.Textbox()
        t_speed = gr.Number(value=1.0)
        t_out = gr.Audio()
        t_btn = gr.Button()
        t_btn.click(fn=run_tts, inputs=[t_text, t_speed], outputs=[t_out], api_name="tts")


demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@demo.app.get("/health")
def health():
    return {
        "service": "MimicAI Headless API",
        "status": "online",
        "version": "1.0.0",
        "hardware": "ZeroGPU enabled" if hasattr(spaces, "zero") else "CPU",
        "endpoints": {
            "clone": "/voice/clone",
            "design": "/voice/design",
            "tts": "/voice/tts",
        }
    }


@demo.app.post("/voice/clone")
async def api_clone(
    audio: UploadFile = File(...),
    text: str = Form(...),
    gender: str = Form("male"),
    language: str = Form("en"),
):
    if not text or not text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)

    suffix = os.path.splitext(audio.filename or "sample.wav")[1] or ".wav"
    raw_tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await audio.read()
        raw_tmp.write(content)
        raw_tmp.close()

        out_path = run_voice_clone(raw_tmp.name, text.strip(), gender=gender, language=language)
        if not out_path or not os.path.exists(out_path):
            return JSONResponse({"error": "Voice cloning failed"}, status_code=500)

        with open(out_path, "rb") as f:
            audio_bytes = f.read()

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


@demo.app.post("/voice/design")
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

        out_path = run_voice_design(text, gender, accent, speed, pitch)
        if not out_path or not os.path.exists(out_path):
            return JSONResponse({"error": "Voice design failed"}, status_code=500)

        with open(out_path, "rb") as f:
            audio_bytes = f.read()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="design.wav"'},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@demo.app.post("/voice/tts")
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
        out_path = run_tts(text, speed)
        if not out_path or not os.path.exists(out_path):
            return JSONResponse({"error": "TTS failed"}, status_code=500)

        with open(out_path, "rb") as f:
            audio_bytes = f.read()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="tts.wav"'},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
