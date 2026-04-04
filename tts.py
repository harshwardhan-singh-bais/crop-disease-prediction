import logging
import os
from typing import Any, Optional

from sarvam_client import SarvamClient, SarvamClientError

logger = logging.getLogger("TTS")


def _extension_for_mime(mime: str) -> str:
    """Pick a file suffix so players decode correctly (Sarvam often returns MP3 as audio/mpeg)."""
    main = (mime or "").split(";")[0].strip().lower()
    if main in ("audio/mpeg", "audio/mp3", "audio/x-mpeg"):
        return ".mp3"
    if main in ("audio/wav", "audio/x-wav", "audio/wave"):
        return ".wav"
    if main in ("audio/ogg", "audio/opus"):
        return ".ogg"
    return ".mp3"


def generate_speech(
    text: str,
    output_path: str,
    sarvam_client: Optional[SarvamClient],
    tts_language_code: str = "en-IN",
    tts_model: str = "bulbul:v2",
    tts_speaker: str = "anushka",
) -> dict[str, Any]:
    """Generate speech with Sarvam TTS and write the output audio file."""
    if not text.strip():
        message = "TTS text is empty"
        logger.error("[TTS] %s", message)
        return {
            "audio_path": None,
            "mode": "error",
            "provider": "sarvam",
            "error": message,
            "mime_type": None,
        }

    if sarvam_client is None:
        message = "Sarvam client is missing. Set SARVAM or SARVAM_API_KEY for online TTS."
        logger.warning("[TTS] %s", message)
        return {
            "audio_path": None,
            "mode": "skipped",
            "provider": "none",
            "error": message,
            "mime_type": None,
        }

    logger.info(
        "[TTS/ONLINE] Synthesizing speech (chars=%s, language=%s, model=%s, speaker=%s)",
        len(text),
        tts_language_code,
        tts_model,
        tts_speaker,
    )
    try:
        response = sarvam_client.synthesize(
            text=text,
            language_code=tts_language_code,
            speaker=tts_speaker,
            model=tts_model,
        )
        audio_bytes = response["audio_bytes"]
        mime_type = response.get("mime_type", "audio/mpeg")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        root, _old_ext = os.path.splitext(output_path)
        correct_path = root + _extension_for_mime(mime_type)
        if correct_path != output_path and os.path.exists(output_path):
            try:
                os.replace(output_path, correct_path)
                output_path = correct_path
            except OSError:
                logger.warning("[TTS] Could not rename output to %s", correct_path)

        logger.info(
            "[TTS/ONLINE] Audio generated: %s bytes written to %s",
            len(audio_bytes),
            output_path,
        )
        return {
            "audio_path": output_path,
            "mode": "online",
            "provider": "sarvam",
            "error": None,
            "mime_type": mime_type,
        }
    except SarvamClientError as exc:
        logger.exception("[TTS/ONLINE] Sarvam TTS failed")
        return {
            "audio_path": None,
            "mode": "error",
            "provider": "sarvam",
            "error": str(exc),
            "mime_type": None,
        }
