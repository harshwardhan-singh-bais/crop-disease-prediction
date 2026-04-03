import argparse
import logging
import os

from dotenv import load_dotenv

from speech_router import speech_pipeline
from voice_input import record_microphone_to_wav

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("VoiceInputTest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record speech from mic and test the STT pipeline.")
    parser.add_argument("--mode", default="auto", choices=["auto", "online", "offline"], help="Pipeline mode")
    parser.add_argument("--seconds", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate")
    parser.add_argument("--audio", default="input.wav", help="Output WAV file path")
    parser.add_argument("--stt-lang", default="en-IN")
    parser.add_argument("--stt-model", default="saarika:v2.5")
    parser.add_argument("--tts-lang", default="en-IN")
    parser.add_argument("--tts-model", default="bulbul:v2")
    parser.add_argument("--speaker", default="anushka")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-language", default=None)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if not (os.getenv("SARVAM") or os.getenv("SARVAM_API_KEY")):
        logger.info("Sarvam key not detected; online TTS/STT will fail unless credentials are added.")

    wav_path = record_microphone_to_wav(
        output_path=args.audio,
        duration_seconds=args.seconds,
        sample_rate=args.sample_rate,
    )

    result = speech_pipeline(
        audio_input_path=wav_path,
        mode=args.mode,
        stt_language_code=args.stt_lang,
        stt_model=args.stt_model,
        tts_language_code=args.tts_lang,
        tts_model=args.tts_model,
        tts_speaker=args.speaker,
        whisper_model_size=args.whisper_model,
        whisper_language_hint=args.whisper_language,
    )

    logger.info("Final transcript: %s", result.get("transcript"))
    logger.info("Result status: %s", result.get("status"))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())