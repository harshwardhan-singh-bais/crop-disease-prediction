import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from sarvam_client import SarvamClient
from tts import generate_speech


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("TTSTerminal")

VALID_SPEAKERS = {
    "anushka", "abhilash", "manisha", "vidya", "arya", "karun", "hitesh",
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan",
    "sumit", "roopa", "kabir", "aayan", "shubh", "ashutosh", "advait",
    "amelia", "sophia", "anand", "tanya", "tarun", "sunny", "mani", "gokul",
    "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
}


def prompt(message: str, default: str) -> str:
    value = input(f"{message} [{default}]: ").strip()
    return value or default


def prompt_speaker(default: str = "anushka") -> str:
    print("Available speakers are: anushka, abhilash, manisha, vidya, arya, karun, hitesh, aditya, ritu, priya, neha, rahul, pooja, rohan, simran, kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir, aayan, shubh, ashutosh, advait, amelia, sophia, anand, tanya, tarun, sunny, mani, gokul, vijay, shruti, suhani, mohit, kavitha, rehan, soham, rupali")
    while True:
        speaker = prompt("Enter speaker name", default).lower()
        if speaker in VALID_SPEAKERS:
            return speaker
        logger.warning("Invalid speaker '%s'. Please choose a valid Sarvam speaker.", speaker)
        retry = input(f"Use default speaker '{default}' instead? (y/n) [y]: ").strip().lower()
        if retry in {"", "y", "yes"}:
            return default


def main() -> int:
    load_dotenv()

    print("\n=== Sarvam Bulbul TTS Terminal ===")
    print("Type your text, choose language, and get speech audio file.\n")

    text = input("Enter text to convert to speech: ").strip()
    if not text:
        logger.error("Text is empty. Nothing to synthesize.")
        return 1

    language_code = prompt("Enter language code (e.g., en-IN, hi-IN, mr-IN)", "en-IN")
    speaker = prompt_speaker("anushka")
    output_name = prompt("Output filename", f"tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")

    sarvam_client = SarvamClient.from_env()
    if sarvam_client is None:
        logger.error("Sarvam API key not found. Set SARVAM or SARVAM_API_KEY in .env.")
        return 1

    logger.info("Starting TTS with model=bulbul:v2, language=%s, speaker=%s", language_code, speaker)
    result = generate_speech(
        text=text,
        output_path=output_name,
        sarvam_client=sarvam_client,
        tts_language_code=language_code,
        tts_model="bulbul:v2",
        tts_speaker=speaker,
    )

    if result.get("error"):
        logger.error("TTS failed: %s", result["error"])
        return 1

    logger.info("Success. Audio saved to: %s", os.path.abspath(result["audio_path"]))
    logger.info("Mime type: %s", result.get("mime_type"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
