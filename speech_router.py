import logging
import os
import time
from typing import Any

from sarvam_client import SarvamClient
from stt import transcribe_audio
from tts import generate_speech

logger = logging.getLogger("Router")


def _build_response_text(transcript: str, custom_response_text: str | None = None) -> str:
    if custom_response_text and custom_response_text.strip():
        return custom_response_text.strip()
    transcript_clean = transcript.strip() or "No speech detected from input audio"
    return f"Detected speech: {transcript_clean}. Recommendation: Please verify crop symptoms with image scan."


def speech_pipeline(
    audio_input_path: str,
    mode: str = "auto",
    output_audio_path: str = "response_output.wav",
    custom_response_text: str | None = None,
    stt_language_code: str = "en-IN",
    stt_model: str = "saarika:v2",
    tts_language_code: str = "en-IN",
    tts_model: str = "bulbul:v2",
    tts_speaker: str = "anushka",
    whisper_model_size: str = "base",
    whisper_language_hint: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end speech pipeline.

    Modes:
    - online: Sarvam STT and Sarvam TTS only
    - offline: Whisper STT only, TTS skipped
    - auto: Sarvam STT first, Whisper fallback; TTS only if Sarvam is available
    """
    started = time.time()
    requested_mode = (mode or "auto").lower()
    logger.info("[PIPELINE] Starting speech pipeline with mode=%s", requested_mode)
    logger.info("[PIPELINE] Input audio path: %s", audio_input_path)

    sarvam_client = SarvamClient.from_env()
    if sarvam_client is None:
        logger.warning("[PIPELINE] Sarvam client not configured from env")
    else:
        logger.info("[PIPELINE] Sarvam client configured")

    stt_result = transcribe_audio(
        file_path=audio_input_path,
        mode=requested_mode,
        sarvam_client=sarvam_client,
        stt_language_code=stt_language_code,
        stt_model=stt_model,
        whisper_model_size=whisper_model_size,
        whisper_language_hint=whisper_language_hint,
    )

    transcript = stt_result.get("transcript", "")
    response_text = _build_response_text(transcript, custom_response_text)
    logger.info("[PIPELINE] Response text generated (chars=%s)", len(response_text))

    should_run_tts = requested_mode != "offline" and custom_response_text is not None
    if should_run_tts:
        tts_result = generate_speech(
            text=response_text,
            output_path=output_audio_path,
            sarvam_client=sarvam_client,
            tts_language_code=tts_language_code,
            tts_model=tts_model,
            tts_speaker=tts_speaker,
        )
    else:
        logger.info("[PIPELINE] Offline mode requested, skipping TTS stage")
        if requested_mode != "offline":
            logger.info("[PIPELINE] TTS skipped because this flow is STT-only")
        tts_result = {
            "audio_path": None,
            "mode": "skipped",
            "provider": "none",
            "error": "TTS disabled in offline mode",
            "mime_type": None,
        }

    elapsed = round(time.time() - started, 3)
    has_errors = bool(stt_result.get("error") or (should_run_tts and tts_result.get("error")))

    report = {
        "status": "error" if has_errors else "success",
        "requested_mode": requested_mode,
        "stt": stt_result,
        "tts": tts_result,
        "transcript": transcript,
        "response_text": response_text,
        "audio_path": tts_result.get("audio_path"),
        "runtime_seconds": elapsed,
    }
    logger.info("[PIPELINE] Completed with status=%s in %ss", report["status"], elapsed)
    return report
