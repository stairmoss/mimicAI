import os
import tempfile
import time
import gradio as gr
import numpy as np

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
def clone_speech(ref_audio, text):
    if not ref_audio:
        return None, "Please record or upload a reference audio sample."
    if not text or not text.strip():
        return None, "Please enter some text to speak."

    t0 = time.time()
    try:
        clean_ref = piper_tts.preprocess_reference_audio(ref_audio)
        audio_bytes = clone_voice(
            text=text.strip()[:500],
            ref_audio_path=clean_ref,
            language="en",
        )
        if not audio_bytes:
            audio_bytes = piper_tts.synthesize(text=text.strip()[:500])

        if audio_bytes:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            dt = time.time() - t0
            return tmp.name, f"Generated in {dt:.2f}s"
        return None, "Voice cloning failed"
    except Exception as exc:
        return None, f"Error: {exc}"


@spaces.GPU
def design_speech(text, gender, accent, speed, pitch):
    if not text or not text.strip():
        return None, "Please enter some text."

    t0 = time.time()
    try:
        design = {
            "gender": gender.lower(),
            "accent": "us" if "american" in accent.lower() else ("in" if "indian" in accent.lower() else ("uk" if "british" in accent.lower() else "jp")),
            "speed": float(speed),
            "pitch": float(pitch),
        }
        audio_bytes = vm._generate_designed_voice(text.strip()[:500], design)
        if not audio_bytes:
            audio_bytes = piper_tts.synthesize(text=text.strip()[:500], length_scale=1.0/max(0.5, speed))

        if audio_bytes:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            dt = time.time() - t0
            return tmp.name, f"Synthesized in {dt:.2f}s using {vm.last_tts_engine}"
        return None, "Synthesis failed"
    except Exception as exc:
        return None, f"Error: {exc}"


def chat_and_speak(message, history, voice_mode, ref_audio, gender, accent, speed):
    if not message or not message.strip():
        return history or [], None

    from gemini_helper import is_gemini_available, GEMINI_API_KEY, GEMINI_ENDPOINT, GEMINI_MODEL
    import requests

    history = list(history or [])
    reply = "I'm MimicAI. Nice to meet you!"
    try:
        if is_gemini_available():
            msgs = [{"role": "system", "content": "You are MimicAI, a helpful and concise voice AI assistant. Keep responses under 2 sentences."}]
            for h in history:
                if isinstance(h, dict):
                    msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")})
                elif isinstance(h, (list, tuple)) and len(h) >= 2:
                    msgs.append({"role": "user", "content": h[0]})
                    msgs.append({"role": "assistant", "content": h[1]})
            msgs.append({"role": "user", "content": message.strip()})

            resp = requests.post(
                GEMINI_ENDPOINT,
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"},
                json={"model": GEMINI_MODEL, "messages": msgs, "max_tokens": 150},
                timeout=20,
            )
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    history.append({"role": "user", "content": message.strip()})
    history.append({"role": "assistant", "content": reply})

    audio_file = None
    if voice_mode == "Cloned Voice" and ref_audio:
        audio_file, _ = clone_speech(ref_audio, reply)
    else:
        audio_file, _ = design_speech(reply, gender, accent, speed, 1.0)

    return history, audio_file


def add_cue(current_text, cue):
    current = current_text or ""
    if current and not current.endswith(" "):
        current += " "
    return current + cue + " "


CUSTOM_CSS = """
:root {
  --bg-dark: #202124;
  --bg-card: #292a2d;
  --bg-input: #303134;
  --border-color: #3c4043;
  --accent-blue: #8ab4f8;
  --accent-hover: #aecbfa;
  --text-primary: #e8eaed;
  --text-secondary: #9aa0a6;
}

body, .gradio-container {
  background-color: var(--bg-dark) !important;
  color: var(--text-primary) !important;
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}

.mimic-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 22px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  margin-top: 8px;
  margin-bottom: 20px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
}

.mimic-brand {
  display: flex;
  align-items: center;
  gap: 16px;
}

.mimic-title-group {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.mimic-title {
  font-size: 22px;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.3px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.mimic-badge {
  font-size: 10px;
  font-weight: 700;
  background: rgba(138, 180, 248, 0.16);
  color: var(--accent-blue);
  border: 1px solid rgba(138, 180, 248, 0.35);
  padding: 2px 8px;
  border-radius: 20px;
  text-transform: uppercase;
}

.mimic-sub {
  font-size: 13px;
  color: var(--text-secondary);
}

.mimic-pills {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.mimic-pill {
  font-size: 11px;
  font-weight: 500;
  background: var(--bg-input);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  padding: 5px 12px;
  border-radius: 20px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.dot-online {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #81c995;
  box-shadow: 0 0 8px rgba(129, 201, 149, 0.7);
  display: inline-block;
}

.block, .gradio-group {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 14px !important;
}

div[role="tablist"] {
  border-bottom: 1px solid var(--border-color) !important;
  gap: 6px !important;
  background: transparent !important;
  padding: 0 4px !important;
  margin-bottom: 16px !important;
}

button[role="tab"] {
  background: transparent !important;
  color: var(--text-secondary) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  padding: 10px 18px !important;
  border-radius: 6px 6px 0 0 !important;
}

button[role="tab"][aria-selected="true"], button[role="tab"].selected {
  color: var(--accent-blue) !important;
  border-bottom: 2px solid var(--accent-blue) !important;
  font-weight: 600 !important;
  background: rgba(138, 180, 248, 0.08) !important;
}

input, textarea, select {
  background-color: var(--bg-input) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
}

input:focus, textarea:focus, select:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 2px rgba(138, 180, 248, 0.25) !important;
}

button.primary, button[variant="primary"] {
  background: var(--accent-blue) !important;
  color: #202124 !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 8px !important;
  box-shadow: 0 2px 6px rgba(138, 180, 248, 0.2) !important;
}

button.primary:hover, button[variant="primary"]:hover {
  background: var(--accent-hover) !important;
}

.cues-container {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 6px;
  margin-bottom: 10px;
}

.cue-btn {
  font-size: 12px !important;
  padding: 4px 10px !important;
  background: var(--bg-input) !important;
  color: var(--accent-blue) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 16px !important;
}

.cue-btn:hover {
  background: var(--border-color) !important;
}
"""

CUSTOM_HEADER = """
<style>""" + CUSTOM_CSS + """</style>
<div class="mimic-header">
  <div class="mimic-brand">
    <svg width="52" height="52" viewBox="0 0 68 68" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="wave1" x1="12" y1="34" x2="56" y2="34" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#8ab4f8"/>
          <stop offset="100%" stop-color="#80d8ff"/>
        </linearGradient>
        <linearGradient id="wave2" x1="12" y1="34" x2="56" y2="34" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#aecbfa"/>
          <stop offset="100%" stop-color="#8ab4f8"/>
        </linearGradient>
        <radialGradient id="centerGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="rgba(138, 180, 248, 0.35)"/>
          <stop offset="100%" stop-color="rgba(138, 180, 248, 0)"/>
        </radialGradient>
      </defs>
      <rect width="68" height="68" rx="19" fill="#292a2d" stroke="#3c4043" stroke-width="1.6"/>
      <circle cx="34" cy="34" r="28" fill="url(#centerGlow)"/>
      <line x1="17.5" y1="26" x2="17.5" y2="42" stroke="#8ab4f8" stroke-width="2" stroke-linecap="round" opacity="0.35"/>
      <line x1="23" y1="19" x2="23" y2="49" stroke="#8ab4f8" stroke-width="2.5" stroke-linecap="round" opacity="0.45"/>
      <line x1="45" y1="19" x2="45" y2="49" stroke="#80d8ff" stroke-width="2.5" stroke-linecap="round" opacity="0.45"/>
      <line x1="50.5" y1="26" x2="50.5" y2="42" stroke="#80d8ff" stroke-width="2" stroke-linecap="round" opacity="0.35"/>
      <path d="M12 34 C17 16, 29 16, 34 34 C39 52, 51 52, 56 34" stroke="url(#wave1)" stroke-width="3.5" stroke-linecap="round"/>
      <path d="M12 34 C17 52, 29 52, 34 34 C39 16, 51 16, 56 34" stroke="url(#wave2)" stroke-width="3.5" stroke-linecap="round"/>
      <circle cx="23" cy="17" r="3.2" fill="#8ab4f8"/>
      <circle cx="23" cy="51" r="3.2" fill="#aecbfa"/>
      <circle cx="45" cy="17" r="3.2" fill="#80d8ff"/>
      <circle cx="45" cy="51" r="3.2" fill="#8ab4f8"/>
      <circle cx="12" cy="34" r="2.2" fill="#8ab4f8"/>
      <circle cx="56" cy="34" r="2.2" fill="#80d8ff"/>
      <path d="M34 27 Q34 34 27 34 Q34 34 34 41 Q34 34 41 34 Q34 34 34 27 Z" fill="#ffffff"/>
      <circle cx="34" cy="34" r="1.6" fill="#8ab4f8"/>
    </svg>
    <div class="mimic-title-group">
      <div class="mimic-title">MimicAI <span class="mimic-badge">AI CLONE</span></div>
      <div class="mimic-sub">Zero-Shot Neural Voice Cloning & Custom Speech Studio</div>
    </div>
  </div>
  <div class="mimic-pills">
    <span class="mimic-pill"><span class="dot-online"></span> ZeroGPU A10G</span>
    <span class="mimic-pill">Piper · Kokoro · F5-TTS</span>
    <span class="mimic-pill">Chrome Dark Mode</span>
  </div>
</div>
"""

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

with gr.Blocks(title="MimicAI — Voice Cloning & TTS", theme=theme) as demo:
    gr.HTML(CUSTOM_HEADER)

    with gr.Tabs():
        with gr.Tab("🎙️ Voice Cloning"):
            with gr.Row():
                with gr.Column(scale=1):
                    clone_ref = gr.Audio(label="Reference Voice (Record or Upload 5-10s)", type="filepath")
                    clone_text = gr.Textbox(label="Text to Speak in Cloned Voice", lines=3, placeholder="Type what you want the clone to say...")
                    clone_btn = gr.Button("Clone Voice", variant="primary")
                with gr.Column(scale=1):
                    clone_out = gr.Audio(label="Cloned Voice Output", type="filepath")
                    clone_status = gr.Textbox(label="Status", lines=1)

            clone_btn.click(fn=clone_speech, inputs=[clone_ref, clone_text], outputs=[clone_out, clone_status])

        with gr.Tab("🎨 Voice Designer"):
            with gr.Row():
                with gr.Column(scale=1):
                    design_text = gr.Textbox(label="Text to Synthesize", lines=3, placeholder="Enter text to speak...")
                    with gr.Row():
                        gender_el = gr.Radio(["Female", "Male"], label="Gender", value="Female")
                        accent_el = gr.Dropdown(["American (US)", "Indian Accent (IN)", "British (UK)", "Japanese (JP)"], label="Accent", value="American (US)")
                    with gr.Row():
                        speed_el = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                        pitch_el = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Pitch")
                    design_btn = gr.Button("Synthesize Voice", variant="primary")
                with gr.Column(scale=1):
                    design_out = gr.Audio(label="Generated Audio", type="filepath")
                    design_status = gr.Textbox(label="Status", lines=1)

            design_btn.click(fn=design_speech, inputs=[design_text, gender_el, accent_el, speed_el, pitch_el], outputs=[design_out, design_status])

        with gr.Tab("💬 Voice Chatbot"):
            chatbot = gr.Chatbot(label="MimicAI Conversation", height=380)

            with gr.Row():
                gr.Markdown("**Expressiveness Cues:**")
                cue_laugh = gr.Button("😄 + Laughter", size="sm", elem_classes=["cue-btn"])
                cue_applause = gr.Button("👏 + Applause", size="sm", elem_classes=["cue-btn"])
                cue_sigh = gr.Button("💨 + Sigh", size="sm", elem_classes=["cue-btn"])
                cue_gasp = gr.Button("😲 + Gasp", size="sm", elem_classes=["cue-btn"])

            with gr.Row():
                chat_input = gr.Textbox(label="Your Message", placeholder="Type a message or click an expressive cue...", scale=4)
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            chat_audio = gr.Audio(label="AI Voice Reply", type="filepath", autoplay=True)

            cue_laugh.click(fn=lambda t: add_cue(t, "[laughter]"), inputs=[chat_input], outputs=[chat_input])
            cue_applause.click(fn=lambda t: add_cue(t, "[applause]"), inputs=[chat_input], outputs=[chat_input])
            cue_sigh.click(fn=lambda t: add_cue(t, "[sigh]"), inputs=[chat_input], outputs=[chat_input])
            cue_gasp.click(fn=lambda t: add_cue(t, "[gasp]"), inputs=[chat_input], outputs=[chat_input])

            with gr.Accordion("Voice Settings for Chat", open=False):
                chat_mode = gr.Radio(["Designed Voice", "Cloned Voice"], label="Voice Type", value="Designed Voice")
                chat_ref = gr.Audio(label="Reference Voice for Chat", type="filepath")

            chat_btn.click(
                fn=chat_and_speak,
                inputs=[chat_input, chatbot, chat_mode, chat_ref, gender_el, accent_el, speed_el],
                outputs=[chatbot, chat_audio],
            )
            chat_input.submit(
                fn=chat_and_speak,
                inputs=[chat_input, chatbot, chat_mode, chat_ref, gender_el, accent_el, speed_el],
                outputs=[chatbot, chat_audio],
            )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, css=CUSTOM_CSS)
