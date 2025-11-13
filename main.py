import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
import logging_utils
import queue
import threading
import time
import numpy as np
import asyncio
import json
from typing import Optional

from audio.settings import (
    REDIS_URL as SETTINGS_REDIS_URL,
    TTS_PROVIDER as SETTINGS_TTS_PROVIDER,
    XTTS_API_URL as SETTINGS_XTTS_API_URL,
    XTTS_SPEAKER_WAV as SETTINGS_XTTS_SPEAKER_WAV,
    ELEVENLABS_API_KEY as SETTINGS_ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID as SETTINGS_ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL_ID as SETTINGS_ELEVENLABS_MODEL_ID,
    ELEVENLABS_OUTPUT_FORMAT as SETTINGS_ELEVENLABS_OUTPUT_FORMAT,
    ORPHEUS_API_URL as SETTINGS_ORPHEUS_API_URL,
)

import redis
try:
    import sounddevice as sd
except ImportError:
    sd = None
import uvicorn
import requests
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel

# --- Provider-specific Imports ---
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play as elevenlabs_play
except ImportError:
    ElevenLabs, elevenlabs_play = None, None

# --- Configuration ---
logger = logging_utils.get_logger("tts-dispatcher-service", service="tts")

REDIS_URL = SETTINGS_REDIS_URL

# --- Provider Configuration ---
TTS_PROVIDER = SETTINGS_TTS_PROVIDER

# XTTS API Config
XTTS_API_URL = SETTINGS_XTTS_API_URL
XTTS_SPEAKER_WAV_PATH = SETTINGS_XTTS_SPEAKER_WAV

# ElevenLabs Config
ELEVENLABS_API_KEY = SETTINGS_ELEVENLABS_API_KEY
ELEVENLABS_VOICE_ID = SETTINGS_ELEVENLABS_VOICE_ID
ELEVENLABS_MODEL_ID = SETTINGS_ELEVENLABS_MODEL_ID
ELEVENLABS_OUTPUT_FORMAT = SETTINGS_ELEVENLABS_OUTPUT_FORMAT

# Orpheus Proxy Config
ORPHEUS_PROXY_URL = SETTINGS_ORPHEUS_API_URL

# Yandex SpeechKit (from audio.settings via env/config)
from audio.settings import (
    YANDEX_API_KEY,
    YANDEX_IAM_TOKEN,
    YANDEX_FOLDER_ID,
    YANDEX_VOICE,
    YANDEX_FORMAT,
    YANDEX_SAMPLE_RATE,
    YANDEX_USE_V3_REST,
)

# --- Global State ---
class ServiceState:
    def __init__(self):
        self.tts_client = None # Can be proxy string or ElevenLabs client
        self.redis_client: redis.Redis | None = None
        self.queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None

state = ServiceState()

# --- Data Models ---
class SpeakRequest(BaseModel):
    text: str

# --- Core Logic ---
def initialize_tts_client():
    """Initializes the appropriate TTS client or proxy based on configuration."""
    logger.info(f"Selected TTS Provider: {TTS_PROVIDER}")

    if TTS_PROVIDER == "xtts":
        logger.info(f"Configured to use XTTS API server at: {XTTS_API_URL}")
        try:
            response = requests.get(f"{XTTS_API_URL}/speakers_list", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to XTTS API server.")
            state.tts_client = "xtts_api_proxy"
        except requests.RequestException as e:
            logger.error(f"Could not connect to XTTS API server. Error: {e}")
            state.tts_client = None

    elif TTS_PROVIDER == "elevenlabs":
        if not ElevenLabs:
            raise RuntimeError("ElevenLabs provider selected, but 'elevenlabs' library is not installed.")
        if not ELEVENLABS_API_KEY:
            raise RuntimeError("ElevenLabs provider selected, but ELEVENLABS_API_KEY is not set.")
        logger.info("Initializing ElevenLabs client...")
        state.tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    elif TTS_PROVIDER == "orpheus":
        logger.info(f"Configured to use Orpheus TTS service at: {ORPHEUS_PROXY_URL}")
        try:
            response = requests.get(f"{ORPHEUS_PROXY_URL}/v1/healthz", timeout=3.05)
            response.raise_for_status()
            logger.info("Successfully connected to Orpheus TTS service.")
            state.tts_client = "orpheus_api_proxy"
        except requests.RequestException as e:
            logger.error(f"Could not connect to Orpheus TTS service. Error: {e}")
            state.tts_client = None
    elif TTS_PROVIDER == "yandex":
        logger.info("Configured to use Yandex SpeechKit via REST API")
        if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
            logger.error("YANDEX_API_KEY and YANDEX_FOLDER_ID are required for Yandex provider.")
            state.tts_client = None
        else:
            state.tts_client = "yandex_rest"
    else:
        raise RuntimeError(f"Invalid TTS_PROVIDER: '{TTS_PROVIDER}'. Choose from [xtts, elevenlabs, orpheus, yandex]")

def speech_worker():
    """Worker thread that processes text from the queue and synthesizes speech."""
    while True:
        try:
            text = state.queue.get()
            if text is None: break
            
            logger.info(f"Processing text for TTS: '{text[:50]}...'")
            audio_np = None
            samplerate = 24000 # Default sample rate

            if TTS_PROVIDER == 'xtts':
                if state.tts_client != "xtts_api_proxy": continue
                try:
                    params = {"text": text, "speaker_wav": XTTS_SPEAKER_WAV_PATH, "language": "ru"}
                    response = requests.get(f"{XTTS_API_URL}/tts_stream", params=params, stream=True, timeout=30)
                    response.raise_for_status()
                    audio_data = b"".join([chunk for chunk in response.iter_content(chunk_size=1024)])
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                except requests.RequestException as e:
                    logger.error(f"Failed to call XTTS API server: {e}")

            elif TTS_PROVIDER == 'elevenlabs':
                try:
                    audio_generator = state.tts_client.text_to_speech.convert(text=text, voice_id=ELEVENLABS_VOICE_ID)
                    audio_data = b"".join([chunk for chunk in audio_generator])
                    # ElevenLabs uses MP3, need to decode it. This is complex, skipping for now.
                    # For now, we assume the `play` function handles it.
                    elevenlabs_play(audio_data)
                except Exception as e:
                    logger.error(f"Failed during ElevenLabs synthesis: {e}")

            elif TTS_PROVIDER == 'orpheus':
                if state.tts_client != "orpheus_api_proxy": continue
                try:
                    payload = {"text": text} # Add other params from ENV later
                    response = requests.post(f"{ORPHEUS_PROXY_URL}/v1/tts/synthesize", json=payload, timeout=30)
                    response.raise_for_status()
                    audio_np = np.frombuffer(response.content, dtype=np.int16)
                except requests.RequestException as e:
                    logger.error(f"Failed to call Orpheus service: {e}")

            # Generic playback for numpy audio data
            if audio_np is not None:
                if sd is None:
                    logger.debug("sounddevice missing; skipping local playback.")
                else:
                    try:
                        sd.play(audio_np, samplerate)
                        time.sleep(len(audio_np) / samplerate * 1.05)
                        logger.info("Playback finished.")
                    except Exception as e:
                        logger.error(f"Error playing audio: {e}")

            state.queue.task_done()
        except Exception as e:
            logger.error(f"Error in speech worker: {e}", exc_info=True)
            state.queue.task_done()

# --- FastAPI App ---
app = FastAPI(title="TTS Dispatcher Service", version="3.0.0")

@app.on_event("startup")
def startup_event():
    initialize_tts_client()
    state.worker_thread = threading.Thread(target=speech_worker, daemon=True)
    state.worker_thread.start()
    logger.info("Speech worker thread started.")

class TTSStreamRequest(BaseModel):
    text: str
    language: str = "ru"


async def _provider_stream(text: str, language: str):
    # Avoid logging full unicode text to prevent Windows console encoding issues
    logger.debug(f"Streaming TTS using {TTS_PROVIDER}; length={len(text)} chars")
    
    if TTS_PROVIDER == 'xtts':
        try:
            params = {"text": text, "speaker_wav": XTTS_SPEAKER_WAV_PATH, "language": language}
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", f"{XTTS_API_URL}/tts_stream", params=params, timeout=60) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except Exception as e:
            logger.error(f"Failed to stream from XTTS API server: {e}", exc_info=True)
        return

    elif TTS_PROVIDER == 'elevenlabs':
        if not ElevenLabs:
            logger.error("ElevenLabs provider selected, but library not installed.")
            return
        try:
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            # Use keyword arguments via lambda to satisfy SDK signature
            model_id = os.getenv("ELEVENLABS_MODEL_ID", ELEVENLABS_MODEL_ID)
            output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", ELEVENLABS_OUTPUT_FORMAT)
            audio_generator = await asyncio.to_thread(
                lambda: client.text_to_speech.convert(
                    voice_id=ELEVENLABS_VOICE_ID,
                    text=text,
                    model_id=model_id,
                    output_format=output_format,
                )
            )
            for chunk in audio_generator:
                yield chunk
        except Exception as e:
            logger.error(f"Failed during ElevenLabs synthesis: {e}", exc_info=True)
        return

    elif TTS_PROVIDER == 'orpheus':
        try:
            payload = {"text": text}
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{ORPHEUS_PROXY_URL}/v1/tts/synthesize", json=payload, timeout=60) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except Exception as e:
            logger.error(f"Failed to call Orpheus service: {e}", exc_info=True)
        return

    elif TTS_PROVIDER == 'yandex':
        # Helpers for Yandex auth
        def _read_sa_key(path: str) -> Optional[dict]:
            import json
            try:
                # Try absolute path first
                if not os.path.isabs(path):
                    abs_path = os.path.abspath(path)
                else:
                    abs_path = path
                logger.info(f"Attempting to read SA key from: {abs_path}")
                with open(abs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"SA key loaded successfully, service_account_id: {data.get('service_account_id', 'unknown')}")
                    return data
            except FileNotFoundError:
                logger.error(f"SA key file not found: {abs_path}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in SA key file: {e}")
                return None
            except Exception as e:
                logger.error(f"Error reading SA key file: {e}")
                return None

        def _issue_yandex_iam_token_from_sa(sa_key: dict) -> Optional[str]:
            try:
                import jwt  # PyJWT
                import requests
            except Exception:
                logger.warning("PyJWT/requests not available; cannot auto-issue Yandex IAM token")
                return None

            now = int(time.time())
            payload = {
                'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
                'iss': sa_key.get('service_account_id'),
                'iat': now,
                'exp': now + 3600,
            }
            private_key = sa_key.get('private_key')
            key_id = sa_key.get('id')
            if not private_key or not key_id:
                return None
            headers_local = {'kid': key_id}
            try:
                jwt_token = jwt.encode(payload, private_key, algorithm='PS256', headers=headers_local)
                resp = requests.post('https://iam.api.cloud.yandex.net/iam/v1/tokens', json={'jwt': jwt_token}, timeout=20)
                if resp.ok:
                    return resp.json().get('iamToken')
            except Exception as e:
                logger.warning(f"Failed to issue Yandex IAM token: {e}")
            return None

        def _ensure_yandex_auth_headers(headers: dict) -> dict:
            token = os.getenv('YANDEX_IAM_TOKEN') or (YANDEX_IAM_TOKEN or None)
            api_key = os.getenv('YANDEX_API_KEY') or (YANDEX_API_KEY or None)

            if token:
                headers["Authorization"] = f"Bearer {token}"
                logger.info("Using YANDEX_IAM_TOKEN from environment")
            elif api_key:
                headers["Authorization"] = f"Api-Key {api_key}"
                logger.info("Using YANDEX_API_KEY from environment")
            else:
                sa_path = os.getenv('YANDEX_SA_KEY_PATH', 'authorized_key.json')
                logger.info(f"Trying to read SA key from: {sa_path}")
                sa_key = _read_sa_key(sa_path)
                if sa_key:
                    logger.info("SA key loaded, attempting to issue IAM token")
                    issued = _issue_yandex_iam_token_from_sa(sa_key)
                    if issued:
                        headers["Authorization"] = f"Bearer {issued}"
                        logger.info("IAM token issued successfully")
                    else:
                        logger.error("Failed to issue IAM token from SA key")
                else:
                    logger.error(f"Could not read SA key from {sa_path}")
            # Log minimal diagnostics (no secrets)
            try:
                auth_used = 'iam_token' if 'Bearer ' in headers.get('Authorization','') else ('api_key' if 'Api-Key ' in headers.get('Authorization','') else 'none')
                logger.info(f"Yandex auth: {auth_used}; folder_id set: {bool(YANDEX_FOLDER_ID)}; v3={YANDEX_USE_V3_REST}")
            except Exception:
                pass
            return headers

        try:
            # v3 or v1 endpoint
            url = 'https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis' if YANDEX_USE_V3_REST else 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
            headers = {}
            headers = _ensure_yandex_auth_headers(headers)
            
            # With unsafeMode: true, Yandex handles text splitting automatically
            logger.info(f"Yandex TTS: sending text of {len(text)} chars with unsafeMode enabled")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                if YANDEX_USE_V3_REST:
                    # Build v3 JSON body
                    body = {
                        "text": text,
                        "hints": [{"voice": YANDEX_VOICE}, {"speed": "1.0"}],
                        "outputAudioSpec": (
                            {"containerAudio": {"containerAudioType": "OGG_OPUS"}} if YANDEX_FORMAT == 'oggopus' else
                            {"containerAudio": {"containerAudioType": "MP3"}} if YANDEX_FORMAT == 'mp3' else
                            {"rawAudio": {"audioEncoding": "LINEAR16_PCM", "sampleRateHertz": YANDEX_SAMPLE_RATE}}
                        ),
                        "loudnessNormalizationType": "LUFS",
                        "unsafeMode": True,  # Enable automatic text splitting
                    }
                    req_headers = {**headers, "Content-Type": "application/json", "Accept": "application/json"}
                    if YANDEX_FOLDER_ID:
                        # Some services expect this header name, include both variants for safety
                        req_headers["x-folder-id"] = YANDEX_FOLDER_ID
                        req_headers["X-YaCloud-FolderId"] = YANDEX_FOLDER_ID
                    resp = await client.post(url, headers=req_headers, json=body)
                    logger.info(f"Yandex v3 response status: {resp.status_code}")
                    if resp.status_code >= 400:
                        try:
                            logger.error({"yandex_v3_error": resp.text})
                        except Exception:
                            pass
                    resp.raise_for_status()
                    # v3 REST returns JSON with base64 audio bytes in audioChunk.data
                    try:
                        # Try to parse as single JSON first
                        payload = resp.json()
                        logger.info(f"Yandex v3 response keys: {list(payload.keys())}")
                        data_b64 = payload.get("audioChunk", {}).get("data")
                        if not data_b64:
                            result = payload.get("result", {})
                            if isinstance(result, dict):
                                data_b64 = result.get("audioChunk", {}).get("data") or result.get("data")
                        if data_b64:
                            import base64
                            chunk_bytes = base64.b64decode(data_b64)
                            logger.info(f"Yandex v3: decoded {len(chunk_bytes)} bytes of audio")
                            yield chunk_bytes
                        else:
                            logger.warning("Yandex v3: no audioChunk.data found, using raw content")
                            yield resp.content
                    except Exception as e:
                        logger.warning(f"Yandex v3: JSON parsing failed ({e}), trying to parse as streaming JSON")
                        # Try to parse as streaming JSON (multiple JSON objects)
                        try:
                            import json
                            content = resp.text
                            logger.info(f"Yandex v3: processing streaming response of {len(content)} chars")
                            # Split by newlines and parse each JSON object
                            lines = content.split('\n')
                            logger.info(f"Yandex v3: found {len(lines)} lines in response")
                            for i, line in enumerate(lines):
                                if line.strip():
                                    try:
                                        chunk_payload = json.loads(line)
                                        logger.info(f"Yandex v3: parsed line {i+1}, keys: {list(chunk_payload.keys())}")
                                        data_b64 = None
                                        if "audioChunk" in chunk_payload:
                                            data_b64 = chunk_payload.get("audioChunk", {}).get("data")
                                        if not data_b64 and "result" in chunk_payload:
                                            result = chunk_payload.get("result", {})
                                            if isinstance(result, dict):
                                                data_b64 = result.get("audioChunk", {}).get("data") or result.get("data")
                                        if data_b64:
                                            import base64
                                            chunk_bytes = base64.b64decode(data_b64)
                                            logger.info(f"Yandex v3: decoded {len(chunk_bytes)} bytes of audio from streaming line {i+1}")
                                            yield chunk_bytes
                                        else:
                                            logger.warning(f"Yandex v3: no audio data found in line {i+1}, structure: {chunk_payload}")
                                    except json.JSONDecodeError as je:
                                        logger.warning(f"Yandex v3: JSON decode error in line {i+1}: {je}")
                                        continue
                        except Exception as stream_e:
                            logger.error(f"Yandex v3: streaming JSON parsing also failed: {stream_e}")
                            # Fallback to raw content
                            yield resp.content
                else:
                    # Legacy v1 form-data
                    data = {
                        'text': text,
                        'voice': YANDEX_VOICE,
                        'format': YANDEX_FORMAT,
                        'sampleRateHertz': YANDEX_SAMPLE_RATE,
                        'folderId': YANDEX_FOLDER_ID,
                        'lang': 'ru-RU' if language.startswith('ru') else 'en-US',
                        'speed': '1.0',
                    }
                    # For v1 also ensure folder header variants
                    v1_headers = dict(headers)
                    if YANDEX_FOLDER_ID:
                        v1_headers["x-folder-id"] = YANDEX_FOLDER_ID
                        v1_headers["X-YaCloud-FolderId"] = YANDEX_FOLDER_ID
                    resp = await client.post(url, headers=v1_headers, data=data)
                    resp.raise_for_status()
                    yield resp.content
        except Exception as e:
            logger.error(f"Failed to call Yandex SpeechKit: {e}", exc_info=True)
            return
    else:
        logger.error(f"TTS provider '{TTS_PROVIDER}' not configured for streaming.")
        return


@app.post("/stream")
async def tts_stream_handler(request: TTSStreamRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    media_type = (
        "audio/mpeg" if TTS_PROVIDER == 'elevenlabs' else (
            "audio/ogg" if TTS_PROVIDER == 'yandex' and YANDEX_FORMAT == 'oggopus' else "audio/wav"
        )
    )
    return StreamingResponse(_provider_stream(request.text, request.language), media_type=media_type)


# === Compatibility endpoints expected by brain proxy ===
class TTSSynthesizeRequest(BaseModel):
    text: str
    language: str | None = None
    voiceId: str | None = None
    modelId: str | None = None
    outputFormat: str | None = None


@app.post("/tts/synthesize")
async def tts_synthesize(req: TTSSynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    language = req.language or "ru"
    media_type = (
        "audio/mpeg" if TTS_PROVIDER == 'elevenlabs' else (
            "audio/ogg" if TTS_PROVIDER == 'yandex' and YANDEX_FORMAT == 'oggopus' else "audio/wav"
        )
    )
    return StreamingResponse(_provider_stream(req.text, language), media_type=media_type)


@app.get("/tts/voices")
async def tts_voices():
    voices = []
    if TTS_PROVIDER == 'elevenlabs':
        # Minimal single-voice listing using configured voice id
        voices.append({
            "id": ELEVENLABS_VOICE_ID or "default",
            "name": (ELEVENLABS_VOICE_ID or "Default"),
            "provider": "elevenlabs",
            "language": "en"
        })
    elif TTS_PROVIDER == 'xtts':
        voices.append({
            "id": "xtts_default",
            "name": "XTTS Default",
            "provider": "xtts",
            "language": "ru"
        })
    elif TTS_PROVIDER == 'orpheus':
        voices.append({
            "id": "orpheus_default",
            "name": "Orpheus Default",
            "provider": "orpheus",
            "language": "en"
        })
    elif TTS_PROVIDER == 'yandex':
        voices.append({
            "id": YANDEX_VOICE or 'oksana',
            "name": (YANDEX_VOICE or 'oksana').capitalize(),
            "provider": "yandex",
            "language": "ru"
        })
    return JSONResponse({"voices": voices})

@app.post("/speak")
def speak(request: SpeakRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    state.queue.put(request.text)
    logger.info(f"Added to TTS queue: '{request.text[:50]}...'")
    return {"status": "ok"}

@app.post("/shutdown")
def shutdown():
    logger.info("Shutdown endpoint called. Initiating graceful shutdown.")
    try:
        if state.redis_client:
            state.redis_client.close()
            logger.info("Redis client closed.")
        if state.worker_thread and state.worker_thread.is_alive():
            # Signal worker to stop (assume queue.put(None) or event)
            state.queue.put(None)
            state.worker_thread.join(timeout=5)
            if state.worker_thread.is_alive():
                logger.warning("Worker thread did not stop gracefully, forcing.")
            logger.info("Worker thread stopped.")
        # Close sounddevice if needed
        logger.info("All resources cleaned up. Exiting.")
        import sys
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7010)
