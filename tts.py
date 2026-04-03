import logging
import os
from typing import Any, Optional

from sarvam_client import SarvamClient, SarvamClientError

logger = logging.getLogger("TTS")


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
