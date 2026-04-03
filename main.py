
import argparse
import json
import logging
import os
import tempfile
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from speech_router import speech_pipeline
from voice_input import record_microphone_to_wav

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logger = logging.getLogger("Main")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logger.info("[BOOT] Logging initialized at level=%s", logging.getLevelName(level))


def _ensure_env_loaded() -> None:
    if not os.getenv("SARVAM") and not os.getenv("SARVAM_API_KEY"):
        load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    _ensure_env_loaded()
    logger.info("[APP] FastAPI app started")
    yield


app = FastAPI(title="Crop Disease Voice API", version="1.0.0", lifespan=lifespan)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech input pipeline for Sarvam STT/TTS (online) and Whisper STT (offline)."
    )
    parser.add_argument("--audio", default=None, help="Input audio file path")
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

    result = speech_pipeline(
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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sarvam_configured": bool(os.getenv("SARVAM") or os.getenv("SARVAM_API_KEY")),
    }


@app.post("/voice/pipeline")
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
    try:
        result = speech_pipeline(
            audio_input_path=temp_path,
            mode=mode,
            enable_tts=True,
            stt_language_code=stt_language_code,
            stt_model=stt_model,
            tts_language_code=tts_language_code,
            tts_model=tts_model,
            tts_speaker=speaker,
            whisper_model_size=whisper_model,
            whisper_language_hint=whisper_language,
        )
        return JSONResponse(result)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/voice/stt")
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
        result = speech_pipeline(
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

    logger.info("[START] Full voice pipeline execution started")
    logger.info("[START] Args=%s", vars(args))

    try:
        result = run_pipeline(args)
        print_final_report(result)
        return 0 if result.get("status") == "success" else 1
    except Exception:
        logger.exception("[FATAL] Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
