import argparse
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(prog="omnivoice-demo-lite", description="OmniVoice Lite demo")
    parser.add_argument("--shard-dir", default=os.path.expanduser("~/.clonemodel_lite/shards"))
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max-memory", type=float, default=3.5)
    return parser


def build_demo(model):
    import gradio as gr

    try:
        languages = ["Auto"] + model.supported_languages
    except Exception:
        languages = ["Auto", "English", "Malayalam", "Hindi", "Tamil", "Telugu", "Kannada", "Bengali", "Marathi"]

    dialect_choices = ["Auto (detect from audio)"]
    for lang, dialects in model.supported_dialects.items():
        for d in dialects:
            dialect_choices.append(f"{lang.title()}: {d.title()}")

    def generate_speech(text, ref_audio, ref_text, language, dialect_choice, quality, instruct, speed):
        if not text or not text.strip():
            return None, "Please enter text to synthesize."

        start = time.time()
        dialect = None
        if dialect_choice and ":" in dialect_choice:
            dialect = dialect_choice.split(":")[1].strip().lower()

        lang = language if language != "Auto" else None

        try:
            audio = model.generate(
                text=text.strip(), ref_audio=ref_audio if ref_audio else None,
                ref_text=ref_text if ref_text else None, language=lang,
                dialect=dialect, quality=quality.lower(),
                instruct=instruct if instruct else None,
                speed=float(speed) if speed != 1.0 else None,
            )
            elapsed = time.time() - start
            waveform = (audio * 32767).astype(np.int16)
            mem = model.memory_manager.get_memory_stats()
            status = f"Done in {elapsed:.1f}s | RAM: {mem.rss_mb:.0f}MB / {mem.ceiling_mb:.0f}MB"
            return (24000, waveform), status
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

    with gr.Blocks(title="OmniVoice Lite") as demo:
        gr.Markdown("# OmniVoice Lite — Ultra-Low-Memory Voice Cloning\n**600+ languages** | **4GB RAM** | **Layer-wise inference** | **Accent-aware cloning**")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter text in any of 600+ languages...")
                ref_audio = gr.Audio(label="Reference Audio (for voice cloning)", type="filepath")
                ref_text = gr.Textbox(label="Reference Text (optional)", lines=2, placeholder="Transcript of reference audio...")

                with gr.Row():
                    language = gr.Dropdown(label="Language", choices=languages, value="Auto")
                    quality = gr.Radio(label="Quality", choices=["Fast", "Balanced", "Best"], value="Balanced")

                with gr.Accordion("Advanced", open=False):
                    dialect = gr.Dropdown(label="Regional Dialect", choices=dialect_choices, value="Auto (detect from audio)")
                    instruct = gr.Textbox(label="Voice Design Instruct", placeholder="e.g., female, low pitch, indian accent")
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")

                gen_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Generated Audio", type="numpy")
                status = gr.Textbox(label="Status", lines=2)

        gen_btn.click(generate_speech, inputs=[text_input, ref_audio, ref_text, language, dialect, quality, instruct, speed], outputs=[output_audio, status])

    return demo


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    shard_dir = args.shard_dir
    if not os.path.exists(shard_dir):
        logger.info("Shards not found. Running first-time setup...")
        from clonemodel.lite.setup_lite import run_setup
        run_setup(output_dir=shard_dir)

    logger.info(f"Loading OmniVoice Lite from {shard_dir}...")
    from clonemodel.lite.omnivoice_lite import OmniVoiceLite
    model = OmniVoiceLite.from_pretrained(shard_dir, max_memory_gb=args.max_memory)

    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb > 6:
            logger.info("Enough RAM detected, loading standard audio tokenizer...")
            from transformers import HiggsAudioV2TokenizerModel
            tokenizer = HiggsAudioV2TokenizerModel.from_pretrained("eustlb/higgs-audio-v2-tokenizer", device_map="cpu")
            model.set_audio_tokenizer(tokenizer)
    except Exception as e:
        logger.info(f"Using layer-wise audio tokenizer: {e}")

    demo = build_demo(model)
    demo.queue().launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
