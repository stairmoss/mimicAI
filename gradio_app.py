import os
import tempfile
import time
import gradio as gr
import numpy as np

from voice_manager import VoiceManager
from clone_engine import clone_voice
import piper_tts

vm = VoiceManager()


def clone_speech(ref_audio, text):
    if not ref_audio:
        return None, "Please record or upload a reference audio sample."
    if not text or not text.strip():
        return None, "Please enter some text to speak."

    t0 = time.time()
    try:
        # preprocess reference audio
        clean_ref = piper_tts.preprocess_reference_audio(ref_audio)
        audio_bytes = clone_voice(
            text=text.strip()[:500],
            ref_audio_path=clean_ref,
            language="en",
        )
        if not audio_bytes:
            # fallback to piper
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

    # simple AI response using gemini helper or local fallback
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

    # synthesize reply voice
    audio_file = None
    if voice_mode == "Cloned Voice" and ref_audio:
        audio_file, _ = clone_speech(ref_audio, reply)
    else:
        audio_file, _ = design_speech(reply, gender, accent, speed, 1.0)

    return history, audio_file



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
    gr.Markdown(
        """
        # 🎙️ MimicAI
        ### Zero-shot neural voice cloning, custom voice design, and AI chatbot.
        """
    )

    with gr.Tabs():
        with gr.Tab("Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    clone_ref = gr.Audio(label="Reference Voice (Record or Upload 5-10s)", type="filepath")
                    clone_text = gr.Textbox(label="Text to Speak in Cloned Voice", lines=3, placeholder="Type what you want the clone to say...")
                    clone_btn = gr.Button("Clone Voice", variant="primary")
                with gr.Column():
                    clone_out = gr.Audio(label="Cloned Voice Output", type="filepath")
                    clone_status = gr.Textbox(label="Status", lines=1)

            clone_btn.click(fn=clone_speech, inputs=[clone_ref, clone_text], outputs=[clone_out, clone_status])

        with gr.Tab("Voice Designer"):
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label="Text to Synthesize", lines=3, placeholder="Enter text to speak...")
                    with gr.Row():
                        gender_el = gr.Radio(["Female", "Male"], label="Gender", value="Female")
                        accent_el = gr.Dropdown(["American (US)", "Indian Accent (IN)", "British (UK)", "Japanese (JP)"], label="Accent", value="American (US)")
                    with gr.Row():
                        speed_el = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                        pitch_el = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Pitch")
                    design_btn = gr.Button("Synthesize Voice", variant="primary")
                with gr.Column():
                    design_out = gr.Audio(label="Generated Audio", type="filepath")
                    design_status = gr.Textbox(label="Status", lines=1)

            design_btn.click(fn=design_speech, inputs=[design_text, gender_el, accent_el, speed_el, pitch_el], outputs=[design_out, design_status])

        with gr.Tab("Voice Chatbot"):
            chatbot = gr.Chatbot(label="Conversation", height=350)
            with gr.Row():
                chat_input = gr.Textbox(label="Message", placeholder="Type a message...", scale=4)
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            chat_audio = gr.Audio(label="AI Voice Reply", type="filepath", autoplay=True)

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
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
