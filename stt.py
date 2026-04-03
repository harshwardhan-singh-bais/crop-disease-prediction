import logging
import mimetypes
import os
from typing import Any, Optional

import whisper

from sarvam_client import SarvamClient, SarvamClientError

logger = logging.getLogger("STT")

_WHISPER_MODEL = None


def _get_whisper_model(model_size: str = "base"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        logger.info("[STT/OFFLINE] Loading Whisper model: %s", model_size)
        _WHISPER_MODEL = whisper.load_model(model_size)
        logger.info("[STT/OFFLINE] Whisper model loaded")
    return _WHISPER_MODEL


def _guess_mime_type(file_path: str) -> str:
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "audio/wav"


def _transcribe_with_sarvam(
    file_path: str,
    client: SarvamClient,
    language_code: str,
    model: str,
) -> dict[str, Any]:
    filename = os.path.basename(file_path)
    mime_type = _guess_mime_type(file_path)

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    logger.info(
        "[STT/ONLINE] Sending %s bytes to Sarvam (language=%s, model=%s)",
        len(audio_bytes),
        language_code,
        model,
    )
    response = client.transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=mime_type,
        language_code=language_code,
        model=model,
    )
    text = response.get("text", "").strip()
    logger.info("[STT/ONLINE] Transcript length: %s chars", len(text))
    return {
        "transcript": text,
        "mode": "online",
        "provider": "sarvam",
        "error": None,
    }


def _transcribe_with_whisper(
    file_path: str,
    whisper_model_size: str,
    language_hint: Optional[str],
) -> dict[str, Any]:
    logger.info(
        "[STT/OFFLINE] Running Whisper transcription (model=%s, file=%s)",
        whisper_model_size,
        file_path,
    )
    model = _get_whisper_model(whisper_model_size)
    kwargs = {}
    if language_hint:
        kwargs["language"] = language_hint
    result = model.transcribe(file_path, **kwargs)
    text = (result.get("text") or "").strip()
    logger.info("[STT/OFFLINE] Transcript length: %s chars", len(text))
    return {
        "transcript": text,
        "mode": "offline",
        "provider": "whisper",
        "error": None,
    }


def transcribe_audio(
    file_path: str,
    mode: str = "auto",
    sarvam_client: Optional[SarvamClient] = None,
    stt_language_code: str = "en-IN",
    stt_model: str = "saarika:v2.5",
    whisper_model_size: str = "base",
    whisper_language_hint: Optional[str] = None,
) -> dict[str, Any]:
    """
    Transcribe input audio with one of these modes:
    - online: Sarvam only
    - offline: Whisper only
    - auto: Sarvam first, Whisper fallback
    """
    if not os.path.exists(file_path):
        message = f"Audio file not found: {file_path}"
        logger.error("[STT] %s", message)
        return {
            "transcript": "",
            "mode": "error",
            "provider": "none",
            "error": message,
        }

    requested_mode = (mode or "auto").lower()
    logger.info("[STT] Requested mode=%s, file=%s", requested_mode, file_path)

    if requested_mode not in {"online", "offline", "auto"}:
        message = f"Invalid STT mode: {requested_mode}"
        logger.error("[STT] %s", message)
        return {
            "transcript": "",
            "mode": "error",
            "provider": "none",
            "error": message,
        }

    if requested_mode in {"online", "auto"} and sarvam_client is not None:
        try:
            return _transcribe_with_sarvam(
                file_path=file_path,
                client=sarvam_client,
                language_code=stt_language_code,
                model=stt_model,
            )
        except SarvamClientError as exc:
            logger.warning("[STT/ONLINE] Sarvam failed: %s", exc)
            if requested_mode == "online":
                return {
                    "transcript": "",
                    "mode": "error",
                    "provider": "sarvam",
                    "error": str(exc),
                }

    if requested_mode == "online" and sarvam_client is None:
        message = "Sarvam client is missing. Set SARVAM or SARVAM_API_KEY for online STT."
        logger.error("[STT/ONLINE] %s", message)
        return {
            "transcript": "",
            "mode": "error",
            "provider": "sarvam",
            "error": message,
        }

    try:
        return _transcribe_with_whisper(
            file_path=file_path,
            whisper_model_size=whisper_model_size,
            language_hint=whisper_language_hint,
        )
    except Exception as exc:
        logger.exception("[STT/OFFLINE] Whisper transcription failed")
        return {
            "transcript": "",
            "mode": "error",
            "provider": "whisper",
            "error": str(exc),
        }
