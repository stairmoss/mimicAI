import os
import csv
import logging
from pathlib import Path
import subprocess

try:
    import whisper
    from pydub import AudioSegment
except ImportError:
    print("Please install required packages first:")
    print("pip install openai-whisper pydub")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RAW_AUDIO_DIR = Path("raw_audio")
DATASET_DIR = Path("dataset")
WAVS_DIR = DATASET_DIR / "wavs"
METADATA_FILE = DATASET_DIR / "metadata.csv"

# Ensure directories exist
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
WAVS_DIR.mkdir(parents=True, exist_ok=True)


def convert_to_wav(input_path: Path) -> Path:
    """Convert audio to 22050Hz Mono WAV (required for Piper)."""
    tmp_path = input_path.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "22050", str(tmp_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return tmp_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to convert {input_path} with ffmpeg: {e}")
        return input_path


def process_audio():
    if not any(RAW_AUDIO_DIR.iterdir()):
        logging.error(f"No audio files found in {RAW_AUDIO_DIR}. Please place your voice recordings there.")
        return

    logging.info("Loading Whisper model (base)... This may take a moment.")
    model = whisper.load_model("base")  # Light enough for 4GB RAM

    metadata_entries = []
    clip_counter = 1

    for audio_file in RAW_AUDIO_DIR.iterdir():
        if audio_file.is_file() and audio_file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            logging.info(f"Processing {audio_file.name}...")

            # 1. Convert to 22050Hz Mono wav for processing
            base_wav = convert_to_wav(audio_file)

            # 2. Transcribe with Whisper to get timestamps
            logging.info("  Transcribing audio and detecting segments...")
            result = model.transcribe(str(base_wav), word_timestamps=False)

            # 3. Load audio with pydub for slicing
            full_audio = AudioSegment.from_wav(str(base_wav))

            # 4. Slice audio based on whisper segments
            for segment in result["segments"]:
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                text = segment["text"].strip()

                # Filter out too short or too long segments (Piper likes 2-10 seconds)
                duration_ms = end_ms - start_ms
                if duration_ms < 1500 or duration_ms > 12000 or not text:
                    continue

                # Slice audio
                chunk = full_audio[start_ms:end_ms]

                # Save chunk
                clip_filename = f"clip_{clip_counter:04d}.wav"
                chunk_path = WAVS_DIR / clip_filename
                chunk.export(str(chunk_path), format="wav")

                # Piper format: filename|transcription
                # Using csv writer to ensure correct formatting without escaping issues
                metadata_entries.append((clip_filename, text))

                clip_counter += 1

            # Clean up temp file
            if base_wav != audio_file and base_wav.exists():
                base_wav.unlink()

    # Write metadata.csv
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for filename, text in metadata_entries:
            writer.writerow([filename, text])

    logging.info("==================================================")
    logging.info(f"Dataset preparation complete!")
    logging.info(f"Generated {len(metadata_entries)} clips in {DATASET_DIR}")
    logging.info("Next Step: Upload the 'dataset' folder to Google Drive for Colab training.")
    logging.info("==================================================")


if __name__ == "__main__":
    process_audio()
