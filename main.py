
import argparse
import base64
import json
import logging
import mimetypes
import os
import tempfile
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from intelligence_engine import IntelligenceEngine
from model_loader import get_model
from sarvam_client import SarvamClient
from speech_router import speech_pipeline as run_speech_pipeline
from stt import transcribe_audio
from tts import generate_speech
from voice_input import record_microphone_to_wav

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logger = logging.getLogger("Main")
DOWNLOADS_DIR = Path("downloads")
API_PREFIX = "/api/v1"


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logger.info("[BOOT] Logging initialized at level=%s", logging.getLevelName(level))


def _ensure_env_loaded() -> None:
    load_dotenv()


@lru_cache(maxsize=1)
def get_intelligence_engine() -> IntelligenceEngine:
    return IntelligenceEngine(protocols_path="disease_protocols.json")


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    _ensure_env_loaded()
    logger.info("[APP] Warming up model and intelligence engine")
    get_model()
    get_intelligence_engine()
    logger.info("[APP] FastAPI app started")
    yield


app = FastAPI(title="Crop Disease Voice API", version="1.0.0", lifespan=lifespan)


class ChatbotRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1)
    disease_json: dict = Field(default_factory=dict)


class ChatbotResponse(BaseModel):
    reply: str
    language: str
    session_id: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified pipeline: run ML prediction locally or start voice pipeline."
    )
    parser.add_argument(
        "--pipeline",
        default="prompt",
        choices=["prompt", "ml", "voice", "speech", "chatbot"],
        help="Which pipeline to run: image ML pipeline or speech chatbot pipeline.",
    )
    parser.add_argument("--image", default="leaf.jpg", help="Input image for ML pipeline (default: leaf.jpg)")
    parser.add_argument("--crop-area", type=float, default=1.0, help="Crop area in acres (default: 1.0)")
    parser.add_argument("--market-price", type=float, default=1500.0, help="Market price Rs per quintal (default: 1500.0)")
    parser.add_argument("--audio", default=None, help="Input audio file path")
    parser.add_argument("--session-id", default="local_session", help="Chatbot session id")
    parser.add_argument("--disease-json", default=None, help="Optional disease JSON string for chatbot context")
    parser.add_argument(
        "--input-source",
        default="prompt",
        choices=["prompt", "file", "mic"],
        help="Where speech input comes from: prompt asks interactively, file uses a wav file, mic records from microphone",
    )
    parser.add_argument("--record-seconds", type=int, default=5, help="Mic recording length in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Mic recording sample rate")
    parser.add_argument(
        "--stt-only",
        action="store_true",
        help="Stop after speech-to-text and do not generate speech output",
    )
    parser.add_argument(
        "--mode",
        default="prompt",
        choices=["prompt", "auto", "online", "offline"],
        help="Pipeline mode: prompt asks interactively, online uses Sarvam only, offline uses Whisper only, auto tries Sarvam then Whisper",
    )
    parser.add_argument("--stt-lang", default="en-IN", help="STT language code")
    parser.add_argument("--stt-model", default="saarika:v2.5", help="Sarvam STT model")
    parser.add_argument("--tts-lang", default="en-IN", help="TTS language code")
    parser.add_argument("--tts-model", default="bulbul:v2", help="Sarvam TTS model")
    parser.add_argument("--speaker", default="anushka", help="Sarvam TTS speaker")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument(
        "--whisper-language",
        default=None,
        help="Optional Whisper language hint like en/hi/mr",
    )
    parser.add_argument(
        "--response-text",
        default=None,
        help="Optional response text to send to TTS; default uses transcript-based template",
    )
    parser.add_argument("--output-audio", default="response_output.wav", help="Output TTS audio path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def _prompt_with_default(message: str, default: str) -> str:
    value = input(f"{message} [{default}]: ").strip()
    return value or default


def prompt_for_interactive_config(args: argparse.Namespace) -> argparse.Namespace:
    logger.info("[INPUT] Interactive mode enabled")

    if args.mode == "prompt":
        args.mode = _prompt_with_default("Choose STT mode (online/offline/auto)", "auto")

    if args.input_source == "prompt":
        args.input_source = _prompt_with_default("Choose input source (mic/file)", "mic")

    if args.input_source == "file":
        args.audio = _prompt_with_default("Enter input WAV file path", args.audio or "input.wav")
    elif args.input_source == "mic":
        args.record_seconds = int(_prompt_with_default("Record duration in seconds", str(args.record_seconds)))
        args.sample_rate = int(_prompt_with_default("Sample rate", str(args.sample_rate)))

    stt_only_answer = _prompt_with_default("Stop after STT and print transcript only? (y/n)", "y")
    args.stt_only = stt_only_answer.lower().startswith("y")

    return args


def prompt_for_pipeline_choice(args: argparse.Namespace) -> argparse.Namespace:
    if args.pipeline == "prompt":
        choice = _prompt_with_default("Choose pipeline (image/speech)", "image").lower()
        args.pipeline = "ml" if choice in {"image", "ml"} else "chatbot"
    return args


def _parse_disease_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("disease_json must be a JSON object")
        return parsed
    except Exception as exc:
        raise ValueError(f"Invalid --disease-json: {exc}") from exc


def _ensure_downloads_dir() -> Path:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return DOWNLOADS_DIR


def run_chatbot_voice_pipeline(args: argparse.Namespace) -> dict:
    logger.info("[CHATBOT VOICE] Starting STT -> Chatbot -> TTS pipeline")
    _ensure_env_loaded()

    if args.mode == "prompt" or args.input_source == "prompt":
        args = prompt_for_interactive_config(args)

    if args.input_source == "mic":
        logger.info("[CHATBOT VOICE] Recording speech from microphone")
        args.audio = record_microphone_to_wav(
            output_path=args.audio or "input.wav",
            duration_seconds=args.record_seconds,
            sample_rate=args.sample_rate,
        )

    if not args.audio:
        raise ValueError("No audio input provided. Use --audio, --input-source file, or --input-source mic.")
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Input audio file not found: {args.audio}")

    sarvam_client = SarvamClient.from_env()
    if sarvam_client is None:
        raise RuntimeError("Sarvam client is not configured. Set SARVAM or SARVAM_API_KEY in .env")

    stt_mode = args.mode if args.mode != "prompt" else "online"
    stt_result = transcribe_audio(
        file_path=args.audio,
        mode=stt_mode,
        sarvam_client=sarvam_client,
        stt_language_code=args.stt_lang,
        stt_model=args.stt_model,
        whisper_model_size=args.whisper_model,
        whisper_language_hint=args.whisper_language,
    )
    if stt_result.get("error"):
        return {
            "status": "error",
            "stage": "stt",
            "error": stt_result.get("error"),
            "stt": stt_result,
        }

    transcript = stt_result.get("transcript", "")
    if not transcript.strip():
        return {
            "status": "error",
            "stage": "stt",
            "error": "Empty transcript from STT",
            "stt": stt_result,
        }

    from chatbot_engine import chatbot_reply
    disease_context = _parse_disease_json(args.disease_json)
    chat = chatbot_reply(args.session_id, transcript, disease_context)
    reply_text = chat.get("reply", "").strip()
    if not reply_text:
        return {
            "status": "error",
            "stage": "chatbot",
            "error": "Chatbot returned empty response",
            "stt": stt_result,
        }

    out_dir = _ensure_downloads_dir()
    output_file = out_dir / f"chatbot_tts_{uuid4().hex}.wav"
    tts_result = generate_speech(
        text=reply_text,
        output_path=str(output_file),
        sarvam_client=sarvam_client,
        tts_language_code=args.tts_lang,
        tts_model=args.tts_model,
        tts_speaker=args.speaker,
    )
    if tts_result.get("error"):
        return {
            "status": "error",
            "stage": "tts",
            "error": tts_result.get("error"),
            "stt": stt_result,
            "chatbot_reply": reply_text,
        }

    return {
        "status": "success",
        "stage": "complete",
        "session_id": args.session_id,
        "transcript": transcript,
        "chatbot_reply": reply_text,
        "stt": stt_result,
        "tts": tts_result,
        "audio_download_url": f"/voice/download/{output_file.name}",
        "audio_file": output_file.name,
    }


def run_ml_pipeline(args: argparse.Namespace) -> dict:
    from intelligence_engine import print_intelligence_report

    logger.info("[ML PIPELINE] Starting local ML execution")
    _ensure_env_loaded()

    image_path = args.image
    if not os.path.exists(image_path):
        logger.error(f"[ML PIPELINE] Request failed: Input image not found: {image_path}")
        raise FileNotFoundError(f"Input image not found: {image_path}")

    logger.info(f"[ML PIPELINE] Loading image: {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    logger.info("[ML PIPELINE] Warming up model (MobileNetV2)")
    model = get_model()
    
    logger.info("[ML PIPELINE] Warming up intelligence engine")
    engine = get_intelligence_engine()

    logger.info("[ML PIPELINE] Layer 1: Running Model Inference...")
    disease_key, confidence, top_k_predictions = model.predict(image_bytes, top_k=3)
    logger.info(f"[ML PIPELINE] Model predicted: {disease_key} with confidence: {confidence*100:.1f}%")

    logger.info("[ML PIPELINE] Layer 2: Validating Intelligence Engine rules...")
    result = engine.analyze(
        disease_key=disease_key,
        confidence=confidence,
        location=None,
        crop_area_acres=args.crop_area,
        market_price_rs_per_quintal=args.market_price,
        top_k_predictions=top_k_predictions,
    )

    print_intelligence_report(result)
    
    print("\n" + "─"*65)
    print("📦 RAW JSON PAYLOAD (for backend/DB):")
    print("─"*65)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    result["status"] = "success"
    return result


def run_pipeline(args: argparse.Namespace) -> dict:
    logger.info("[BOOT] Loading environment variables from .env")
    _ensure_env_loaded()

    if args.mode == "prompt" or args.input_source == "prompt":
        args = prompt_for_interactive_config(args)

    if args.input_source == "mic":
        logger.info("[INPUT] Recording speech from microphone")
        args.audio = record_microphone_to_wav(
            output_path=args.audio or "input.wav",
            duration_seconds=args.record_seconds,
            sample_rate=args.sample_rate,
        )

    if not args.audio:
        raise ValueError("No audio input provided. Use --audio, --input-source file, or --input-source mic.")

    logger.info("[CHECK] Input audio path=%s", args.audio)
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Input audio file not found: {args.audio}")

    logger.info("[CHECK] Requested mode=%s", args.mode)
    logger.info("[CHECK] Sarvam key present=%s", bool(os.getenv("SARVAM") or os.getenv("SARVAM_API_KEY")))

    result = run_speech_pipeline(
        audio_input_path=args.audio,
        mode=args.mode,
        output_audio_path=args.output_audio,
        custom_response_text=args.response_text,
        enable_tts=not args.stt_only,
        stt_language_code=args.stt_lang,
        stt_model=args.stt_model,
        tts_language_code=args.tts_lang,
        tts_model=args.tts_model,
        tts_speaker=args.speaker,
        whisper_model_size=args.whisper_model,
        whisper_language_hint=args.whisper_language,
    )
    return result


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload.file.read())
        return temp_file.name


def _decode_base64_image(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if not payload:
        raise HTTPException(status_code=400, detail="image_base64 is empty")

    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1].strip()

    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc


def _build_prediction_response(
    disease_key: str,
    confidence: float,
    top_k_predictions: list[dict],
    text_input: str | None,
    location_lat: float | None,
    location_lon: float | None,
    crop_area_acres: float,
    market_price_rs_per_quintal: float,
) -> dict:
    engine = get_intelligence_engine()
    location = None
    if location_lat is not None and location_lon is not None:
        location = (location_lat, location_lon)

    result = engine.analyze(
        disease_key=disease_key,
        confidence=confidence,
        location=location,
        crop_area_acres=crop_area_acres,
        market_price_rs_per_quintal=market_price_rs_per_quintal,
        top_k_predictions=top_k_predictions,
    )

    result["speech_input"] = text_input
    result["frontend_message"] = (
        "Speech text received from frontend." if text_input else "No speech text provided."
    )
    return result


@app.get("/health")
@app.get(f"{API_PREFIX}/health")
def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sarvam_configured": bool(os.getenv("SARVAM") or os.getenv("SARVAM_API_KEY")),
    }


@app.get("/endpoints")
@app.get(f"{API_PREFIX}/endpoints")
def api_endpoints() -> dict:
    return {
        "status": "ok",
        "base_prefix": API_PREFIX,
        "endpoints": [
            {
                "name": "health",
                "method": "GET",
                "path": f"{API_PREFIX}/health",
                "purpose": "Check server status and environment configuration",
            },
            {
                "name": "ml_predict",
                "method": "POST",
                "path": f"{API_PREFIX}/predict",
                "purpose": "Upload leaf image and get disease JSON output",
                "inputs": ["file OR image_base64", "text_input optional", "location optional"],
            },
            {
                "name": "voice_stt",
                "method": "POST",
                "path": f"{API_PREFIX}/voice/stt",
                "purpose": "Speech file to transcript only",
            },
            {
                "name": "voice_tts",
                "method": "POST",
                "path": f"{API_PREFIX}/voice/tts",
                "purpose": "Text to Sarvam TTS audio file",
            },
            {
                "name": "voice_pipeline",
                "method": "POST",
                "path": f"{API_PREFIX}/voice/pipeline",
                "purpose": "Speech to transcript to TTS (no chatbot)",
            },
            {
                "name": "chatbot_reply",
                "method": "POST",
                "path": f"{API_PREFIX}/chatbot/reply",
                "purpose": "Send transcript/context to Gemini chatbot",
            },
            {
                "name": "chatbot_ask_with_disease",
                "method": "POST",
                "path": f"{API_PREFIX}/chatbot/ask-with-disease",
                "purpose": "Disease JSON from /predict (or new image) + text or audio question → Gemini",
            },
            {
                "name": "voice_chatbot_pipeline",
                "method": "POST",
                "path": f"{API_PREFIX}/voice/chatbot-pipeline",
                "purpose": "Speech to STT -> chatbot -> TTS -> downloadable audio",
            },
            {
                "name": "voice_download",
                "method": "GET",
                "path": f"{API_PREFIX}/voice/download/{{filename}}",
                "purpose": "Download generated TTS WAV file",
            },
        ],
    }


@app.post("/chatbot/reply", response_model=ChatbotResponse)
@app.post(f"{API_PREFIX}/chatbot/reply", response_model=ChatbotResponse)
def chatbot_reply_endpoint(payload: ChatbotRequest):
    try:
        from chatbot_engine import chatbot_reply
        result = chatbot_reply(
            session_id=payload.session_id,
            user_input=payload.user_input,
            disease_json=payload.disease_json,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {exc}") from exc


@app.post("/chatbot/ask-with-disease")
@app.post(f"{API_PREFIX}/chatbot/ask-with-disease")
async def chatbot_ask_with_disease_context(
    session_id: str = Form(default="api_session"),
    disease_json: str | None = Form(
        default=None,
        description="JSON string from POST /predict (full response body). Ignored if `image` is sent.",
    ),
    question: str | None = Form(
        default=None,
        description="Farmer question in text. Omit if sending `audio` instead.",
    ),
    audio: UploadFile | None = File(
        default=None,
        description="Speech question; transcribed with same STT stack as /voice/stt.",
    ),
    image: UploadFile | None = File(
        default=None,
        description="Optional leaf image; if set, runs /predict first and uses that dict as context (overrides disease_json).",
    ),
    image_base64: str | None = Form(default=None),
    stt_mode: str = Form(default="auto"),
    stt_language_code: str = Form(default="en-IN"),
    stt_model: str = Form(default="saarika:v2.5"),
    whisper_model: str = Form(default="base"),
    whisper_language: str | None = Form(default=None),
    location_lat: float | None = Form(default=None),
    location_lon: float | None = Form(default=None),
    crop_area_acres: float = Form(default=1.0),
    market_price_rs_per_quintal: float = Form(default=1500.0),
    top_k: int = Form(default=3),
    text_input_for_predict: str | None = Form(
        default=None,
        description="Optional note stored on predict result when using `image` (same as /predict text_input).",
    ),
):
    """
    Flow you described: disease JSON (from earlier image scan) + farmer question → Gemini.

    - Context: send `disease_json` (stringified /predict output), **or** send `image` to run ML here.
    - Question: `question` text and/or `audio` (STT). If both, audio transcript wins.
    Uses `GEMINI_API_KEY` from `.env` (see `chatbot_engine` / `gemini_client`).
    """
    disease_ctx: dict | None = None

    if image is not None:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file is empty")
        try:
            plant_model = get_model()
            disease_key, confidence, top_k_predictions = plant_model.predict(
                image_bytes=image_bytes,
                top_k=top_k,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid image or inference failed: {exc}"
            ) from exc
        disease_ctx = _build_prediction_response(
            disease_key=disease_key,
            confidence=confidence,
            top_k_predictions=top_k_predictions,
            text_input=text_input_for_predict,
            location_lat=location_lat,
            location_lon=location_lon,
            crop_area_acres=crop_area_acres,
            market_price_rs_per_quintal=market_price_rs_per_quintal,
        )
        disease_ctx["input_image_name"] = image.filename or "leaf.jpg"
        disease_ctx["status"] = "success"
    elif disease_json is not None and disease_json.strip():
        try:
            parsed = json.loads(disease_json)
            if not isinstance(parsed, dict):
                raise ValueError("disease_json must be a JSON object")
            disease_ctx = parsed
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid disease_json: {exc}"
            ) from exc
    elif image_base64 is not None and image_base64.strip():
        try:
            image_bytes = _decode_base64_image(image_base64)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid image_base64: {exc}"
            ) from exc
        if not image_bytes:
            raise HTTPException(status_code=400, detail="image_base64 payload is empty")
        try:
            plant_model = get_model()
            disease_key, confidence, top_k_predictions = plant_model.predict(
                image_bytes=image_bytes,
                top_k=top_k,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid image or inference failed: {exc}"
            ) from exc
        disease_ctx = _build_prediction_response(
            disease_key=disease_key,
            confidence=confidence,
            top_k_predictions=top_k_predictions,
            text_input=text_input_for_predict,
            location_lat=location_lat,
            location_lon=location_lon,
            crop_area_acres=crop_area_acres,
            market_price_rs_per_quintal=market_price_rs_per_quintal,
        )
        disease_ctx["input_image_name"] = "leaf.jpg"
        disease_ctx["status"] = "success"
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide disease_json (string), or upload image, or image_base64",
        )

    user_q = (question or "").strip()
    stt_meta: dict | None = None

    if audio is not None:
        temp_audio = _save_upload_to_temp(audio)
        try:
            _ensure_env_loaded()
            sarvam_client = SarvamClient.from_env()
            stt_result = transcribe_audio(
                file_path=temp_audio,
                mode=stt_mode,
                sarvam_client=sarvam_client,
                stt_language_code=stt_language_code,
                stt_model=stt_model,
                whisper_model_size=whisper_model,
                whisper_language_hint=whisper_language,
            )
            stt_meta = {
                "mode": stt_result.get("mode"),
                "provider": stt_result.get("provider"),
                "error": stt_result.get("error"),
            }
            if stt_result.get("error"):
                raise HTTPException(
                    status_code=400,
                    detail=f"STT failed: {stt_result.get('error')}",
                )
            transcript = (stt_result.get("transcript") or "").strip()
            if not transcript:
                raise HTTPException(status_code=400, detail="STT transcript is empty")
            user_q = transcript
        finally:
            try:
                os.remove(temp_audio)
            except OSError:
                pass

    if not user_q:
        raise HTTPException(
            status_code=400,
            detail="Provide question (text) and/or audio with speech",
        )

    try:
        from chatbot_engine import chatbot_reply

        chat = chatbot_reply(session_id, user_q, disease_ctx)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini chatbot failed: {exc}") from exc

    return JSONResponse(
        {
            "status": "success",
            "session_id": chat["session_id"],
            "reply": chat["reply"],
            "language": chat.get("language", "auto"),
            "question_used": user_q,
            "used_audio": audio is not None,
            "stt": stt_meta,
            "had_image_predict": image is not None or bool(
                image_base64 and str(image_base64).strip()
            ),
        }
    )


@app.post("/predict")
@app.post(f"{API_PREFIX}/predict")
async def predict_leaf_disease(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
    text_input: str | None = Form(default=None),
    location_lat: float | None = Form(default=None),
    location_lon: float | None = Form(default=None),
    crop_area_acres: float = Form(default=1.0),
    market_price_rs_per_quintal: float = Form(default=1500.0),
    top_k: int = Form(default=3),
):
    if file is None and not image_base64:
        raise HTTPException(status_code=400, detail="Provide either file upload or image_base64")

    if file is not None:
        image_bytes = await file.read()
        image_name = file.filename or "leaf.jpg"
    else:
        image_bytes = _decode_base64_image(image_base64 or "")
        image_name = "leaf.jpg"

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image payload is empty")

    try:
        plant_model = get_model()
        disease_key, confidence, top_k_predictions = plant_model.predict(
            image_bytes=image_bytes,
            top_k=top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image or inference failed: {exc}") from exc

    result = _build_prediction_response(
        disease_key=disease_key,
        confidence=confidence,
        top_k_predictions=top_k_predictions,
        text_input=text_input,
        location_lat=location_lat,
        location_lon=location_lon,
        crop_area_acres=crop_area_acres,
        market_price_rs_per_quintal=market_price_rs_per_quintal,
    )

    result["input_image_name"] = image_name
    result["status"] = "success"
    return JSONResponse(result)


@app.post("/voice/pipeline")
@app.post(f"{API_PREFIX}/voice/pipeline")
async def voice_pipeline(
    file: UploadFile = File(...),
    mode: str = Form(default="auto"),
    stt_language_code: str = Form(default="en-IN"),
    stt_model: str = Form(default="saarika:v2.5"),
    tts_language_code: str = Form(default="en-IN"),
    tts_model: str = Form(default="bulbul:v2"),
    speaker: str = Form(default="anushka"),
    whisper_model: str = Form(default="base"),
    whisper_language: str | None = Form(default=None),
):
    temp_path = _save_upload_to_temp(file)
    out_dir = _ensure_downloads_dir()
    output_file = out_dir / f"voice_pipeline_{uuid4().hex}.wav"
    try:
        result = run_speech_pipeline(
            audio_input_path=temp_path,
            mode=mode,
            enable_tts=True,
            output_audio_path=str(output_file),
            stt_language_code=stt_language_code,
            stt_model=stt_model,
            tts_language_code=tts_language_code,
            tts_model=tts_model,
            tts_speaker=speaker,
            whisper_model_size=whisper_model,
            whisper_language_hint=whisper_language,
        )
        payload = dict(result)
        final_audio = payload.get("audio_path")
        if final_audio and os.path.exists(final_audio):
            safe_name = Path(final_audio).name
            payload["audio_download_url"] = f"{API_PREFIX}/voice/download/{safe_name}"
            payload["audio_file"] = safe_name
        return JSONResponse(payload)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/voice/chatbot-pipeline")
@app.post(f"{API_PREFIX}/voice/chatbot-pipeline")
async def voice_chatbot_pipeline(
    file: UploadFile = File(...),
    session_id: str = Form(default="api_session"),
    disease_json: str | None = Form(default=None),
    stt_language_code: str = Form(default="en-IN"),
    stt_model: str = Form(default="saarika:v2.5"),
    tts_language_code: str = Form(default="en-IN"),
    tts_model: str = Form(default="bulbul:v2"),
    speaker: str = Form(default="anushka"),
):
    _ensure_env_loaded()
    temp_path = _save_upload_to_temp(file)
    try:
        sarvam_client = SarvamClient.from_env()
        if sarvam_client is None:
            raise HTTPException(status_code=400, detail="Sarvam client is not configured")

        stt_result = transcribe_audio(
            file_path=temp_path,
            mode="online",
            sarvam_client=sarvam_client,
            stt_language_code=stt_language_code,
            stt_model=stt_model,
        )
        if stt_result.get("error"):
            raise HTTPException(status_code=400, detail=stt_result.get("error"))

        transcript = stt_result.get("transcript", "").strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="STT transcript is empty")

        from chatbot_engine import chatbot_reply
        chat = chatbot_reply(session_id, transcript, _parse_disease_json(disease_json))
        reply_text = (chat.get("reply") or "").strip()
        if not reply_text:
            raise HTTPException(status_code=500, detail="Chatbot returned empty response")

        out_dir = _ensure_downloads_dir()
        output_file = out_dir / f"chatbot_tts_{uuid4().hex}.wav"
        tts_result = generate_speech(
            text=reply_text,
            output_path=str(output_file),
            sarvam_client=sarvam_client,
            tts_language_code=tts_language_code,
            tts_model=tts_model,
            tts_speaker=speaker,
        )
        if tts_result.get("error"):
            raise HTTPException(status_code=500, detail=tts_result.get("error"))

        tts_path = tts_result.get("audio_path")
        if not tts_path or not os.path.exists(tts_path):
            raise HTTPException(status_code=500, detail="TTS output file missing")
        tts_name = Path(tts_path).name
        return JSONResponse(
            {
                "status": "success",
                "session_id": session_id,
                "transcript": transcript,
                "chatbot_reply": reply_text,
                "audio_download_url": f"{API_PREFIX}/voice/download/{tts_name}",
                "audio_file": tts_name,
            }
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.get("/voice/download/{filename}")
@app.get(f"{API_PREFIX}/voice/download/{{filename}}")
def download_voice_file(filename: str):
    safe_name = Path(filename).name
    file_path = (_ensure_downloads_dir() / safe_name).resolve()
    downloads_root = _ensure_downloads_dir().resolve()
    if downloads_root not in file_path.parents and file_path != downloads_root:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    guessed, _ = mimetypes.guess_type(safe_name)
    media_type = guessed or "application/octet-stream"
    return FileResponse(
        path=str(file_path), filename=safe_name, media_type=media_type
    )


@app.post("/voice/stt")
@app.post(f"{API_PREFIX}/voice/stt")
async def voice_stt(
    file: UploadFile = File(...),
    mode: str = Form(default="auto"),
    stt_language_code: str = Form(default="en-IN"),
    stt_model: str = Form(default="saarika:v2.5"),
    whisper_model: str = Form(default="base"),
    whisper_language: str | None = Form(default=None),
):
    temp_path = _save_upload_to_temp(file)
    try:
        result = run_speech_pipeline(
            audio_input_path=temp_path,
            mode=mode,
            enable_tts=False,
            stt_language_code=stt_language_code,
            stt_model=stt_model,
            whisper_model_size=whisper_model,
            whisper_language_hint=whisper_language,
        )
        stt_result = result.get("stt", {})
        return JSONResponse(
            {
                "status": result.get("status"),
                "mode": stt_result.get("mode"),
                "provider": stt_result.get("provider"),
                "transcript": result.get("transcript"),
                "error": stt_result.get("error"),
            }
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/voice/tts")
@app.post(f"{API_PREFIX}/voice/tts")
async def voice_tts(
    text: str = Form(...),
    tts_language_code: str = Form(default="en-IN"),
    tts_model: str = Form(default="bulbul:v2"),
    speaker: str = Form(default="anushka"),
):
    from sarvam_client import SarvamClient
    from tts import generate_speech

    _ensure_env_loaded()
    client = SarvamClient.from_env()
    if client is None:
        raise HTTPException(status_code=400, detail="Sarvam client is not configured")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        output_path = temp_file.name

    try:
        result = generate_speech(
            text=text,
            output_path=output_path,
            sarvam_client=client,
            tts_language_code=tts_language_code,
            tts_model=tts_model,
            tts_speaker=speaker,
        )
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return JSONResponse(result)
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass


def print_final_report(result: dict) -> None:
    final_output = {
        "status": result.get("status"),
        "requested_mode": result.get("requested_mode"),
        "stt_mode": result.get("stt", {}).get("mode"),
        "stt_provider": result.get("stt", {}).get("provider"),
        "tts_mode": result.get("tts", {}).get("mode"),
        "tts_provider": result.get("tts", {}).get("provider"),
        "transcript": result.get("transcript"),
        "response_text": result.get("response_text"),
        "audio_path": result.get("audio_path"),
        "runtime_seconds": result.get("runtime_seconds"),
        "errors": {
            "stt_error": result.get("stt", {}).get("error"),
            "tts_error": result.get("tts", {}).get("error"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    print("\n" + "=" * 90)
    print("FINAL PIPELINE REPORT")
    print("=" * 90)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    print("=" * 90 + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)
    args = prompt_for_pipeline_choice(args)

    logger.info("[START] Unified pipeline execution started")
    logger.info("[START] Args=%s", vars(args))

    try:
        if args.pipeline == "ml":
            result = run_ml_pipeline(args)
            return 0 if result.get("status") == "success" else 1
        if args.pipeline in {"speech", "chatbot", "voice"}:
            result = run_chatbot_voice_pipeline(args)
            print("\n" + "=" * 90)
            print("CHATBOT VOICE PIPELINE REPORT")
            print("=" * 90)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 90 + "\n")
            return 0 if result.get("status") == "success" else 1
        else:
            result = run_pipeline(args)
            print_final_report(result)
            return 0 if result.get("status") == "success" else 1
    except Exception:
        logger.exception("[FATAL] Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
