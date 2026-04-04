import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env before using chatbot.")

model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=key)


def get_response(messages):
    prompt_text = "\n\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '').strip()}"
        for m in messages
        if m.get("content")
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt_text,
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text