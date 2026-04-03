import logging
import wave
from pathlib import Path

import sounddevice as sd
import numpy as np

logger = logging.getLogger("VoiceInput")


def record_microphone_to_wav(
    output_path: str,
    duration_seconds: int = 5,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """Record microphone audio and save it as a 16-bit WAV file."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[VOICE INPUT] Recording microphone for %ss at %s Hz (%s channel(s))",
        duration_seconds,
        sample_rate,
        channels,
    )
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)

    with wave.open(str(output_file), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    logger.info("[VOICE INPUT] Saved microphone audio to %s", output_file)
    return str(output_file)