# MimicAI 

MimicAI is a production-hardened, state-of-the-art zero-shot text-to-speech (TTS) and voice cloning system. It features a modern, responsive web dashboard with advanced styling controls, real-time diagnostics, thread-safe asynchronous task runners, and secure API endpoints.

## Key Features

- **Voice Cloning Profiles**: Build, save, and manage custom cloned voice profiles from short (3-10s) reference clips.
- **Zero-Shot Voice Design (New)**: Style voices dynamically using the designer dashboard (configure Gender, Accent, Speed, and Pitch in real-time) with automatic multi-engine fallbacks (Kokoro / Piper).
- **Inline Non-Verbal Controls**: A quick-insert cues panel to seamlessly add expressiveness like `[laughter]`, `[sigh]`, or `[surprise-ah]` directly into text.
- **Audio Playback Controls**: Dedicated play, stop, and mute button controls to manage audio streams in the chat interface.
- **System Diagnostics Badge**: A real-time system status widget displaying availability and load status of synthesis engines (Piper, Kokoro, eSpeak-ng).
- **Theme Switcher**: Smooth toggle between clean light mode and premium dark mode.
- **Production-Hardened Security**: Protected endpoints preventing path traversals, configurable CORS headers, and secure environment variable loaders.

---

## Installation

### Step 1: Clone and Set Up Dependencies
Use python virtual environments (`venv`) or `uv` to sync dependencies:

```bash
git clone https://github.com/stairmoss/mimicAI.git
cd mimicAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory:

```env
PORT=8000
HOST=127.0.0.1
CORS_ORIGINS=*
HACKCLUB_API_KEY=your_optional_api_key
```

---

## Quick Start

Launch the local web application server:

```bash
python3 app.py
```

Open `http://127.0.0.1:8000` in your web browser to access the MimicAI dashboard.

---
