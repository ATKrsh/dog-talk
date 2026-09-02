# Dog Talk 🐕‍🦺

**AI-Powered Dog Behavior Interpreter** — Point your phone at any dog and understand what they're feeling, saying, and about to do.

## Architecture

- **Android App** (Kotlin + Jetpack Compose) — Premium dark UI with real-time camera & audio
- **Python Server** (FastAPI + WebSocket) — Runs on your PC, processes video & audio
- **AI Pipeline** — YOLOv8 pose estimation + YAMNet audio classification + behavior knowledge base + local LLM

## Quick Start

### 1. Start the Server (on your PC)

```bash
cd server
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

### 2. Install the Android App

```bash
cd android
# Build and install via Android CLI or Android Studio
android run
```

### 3. Connect

- Ensure phone and PC are on the same Wi-Fi network
- The app will auto-discover the server, or enter the IP manually in Settings

## Requirements

- **Phone**: Android 11+ (API 30+), tested on Redmi Note 10 Pro Max
- **PC**: Python 3.10+, GPU recommended (NVIDIA with CUDA)
- **Optional**: Ollama with a vision model for enhanced natural language interpretation

## License

MIT
