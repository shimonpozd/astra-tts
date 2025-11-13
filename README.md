# TTS Dispatcher Service

This directory contains the standalone FastAPI-based TTS dispatcher used by the Astra stack. It exposes the same `/stream`, `/tts/synthesize`, `/tts/voices`, and `/speak` endpoints that the rest of the project already depends on, and it can proxy between ElevenLabs and Yandex SpeechKit (the only providers we deploy on the server).

## Features

- Selects the provider via `ASTRA_TTS_PROVIDER` (default: `xtts`) and wires the correct API client/payload.
- Mirrors the streaming endpoint that front-end components such as `StreamingAudioMessage` and `useStreamingTTS` consume.
- Automatically handles Yandex IAM tokens (service account or API key) and splits long text when `unsafeMode` is enabled.
- Includes a worker queue + optional local audio playback for CLI/test automation via `state.queue`.

## Setup

```bash
cd tts
python -m venv .venv
source .venv/bin/activate          # OR `.venv\\Scripts\\activate`
pip install -r requirements.txt
```

## Configuration

Environment variables are read from `audio.settings` (which merges `config/defaults.toml` + overrides + `overrides.toml`). Key overrides (only ElevenLabs and Yandex are required in production):

- `ASTRA_TTS_PROVIDER` – one of `xtts`, `elevenlabs`, `orpheus`, `yandex`.
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`, `ELEVENLABS_OUTPUT_FORMAT`.
- `YANDEX_API_KEY`, `YANDEX_IAM_TOKEN`, `YANDEX_FOLDER_ID`, `YANDEX_VOICE`, `YANDEX_FORMAT`, `YANDEX_SAMPLE_RATE`, `YANDEX_USE_V3_REST`, `YANDEX_SA_KEY_PATH`.
- `REDIS_URL` (optional; only used if Redis behavior is enabled later).

For Docker deployments see `deploy/tts/tts.env.example` and `deploy/tts/docker-compose.yml`.

## Running locally

```bash
cd tts
uvicorn tts.main:app --host 0.0.0.0 --port 7010
```

You can then call:

- `POST /stream` (streaming response)
- `POST /tts/synthesize` / `POST /tts/voices` / `POST /speak` (compatibility)

Use the admin frontend at `/api/tts/stream` or any `authorizedFetch` call defined under `astra-web-client/src/services/ttsService.ts`.

## Docker deployment

See `deploy/tts/README.md` for the production-ready recipe (Dockerfile, compose, env example, smoke tests).

## Testing

Manual tests (ElevenLabs + Yandex only):

- `curl -X POST http://localhost:7010/stream -H "Content-Type: application/json" -d '{"text":"Привет","language":"ru"}' > sample.ogg`
- Ensure `ELEVENLABS_API_KEY` and `YANDEX_FOLDER_ID` are populated in `deploy/tts/tts.env` or `.env`, then rerun.
