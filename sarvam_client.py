import base64
import os
from typing import Any, Optional

import requests


class SarvamClientError(RuntimeError):
    pass


class SarvamClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sarvam.ai",
        stt_path: str = "/speech-to-text",
        tts_path: str = "/text-to-speech",
        timeout_seconds: int = 45,
    ):
        if not api_key:
            raise SarvamClientError("Sarvam API key is required")

        self.api_key = api_key
        self.stt_url = f"{base_url.rstrip('/')}{stt_path}"
        self.tts_url = f"{base_url.rstrip('/')}{tts_path}"
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> Optional["SarvamClient"]:
        api_key = os.getenv("SARVAM", "") or os.getenv("SARVAM_API_KEY", "")
        if not api_key:
            return None

        base_url = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
        stt_path = os.getenv("SARVAM_STT_PATH", "/speech-to-text")
        tts_path = os.getenv("SARVAM_TTS_PATH", "/text-to-speech")
        timeout = int(os.getenv("SARVAM_TIMEOUT_SECONDS", "45"))
        return cls(
            api_key=api_key,
            base_url=base_url,
            stt_path=stt_path,
            tts_path=tts_path,
            timeout_seconds=timeout,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "api-subscription-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        language_code: str = "en-IN",
        model: str = "saarika:v2.5",
    ) -> dict[str, Any]:
        files = {
            "file": (filename or "audio.wav", audio_bytes, content_type or "audio/wav"),
        }
        data = {
            "language_code": language_code,
            "model": model,
        }

        try:
            response = requests.post(
                self.stt_url,
                headers=self._headers(),
                files=files,
                data=data,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = self._extract_error_detail(exc)
            raise SarvamClientError(f"Sarvam STT request failed: {detail}") from exc

        payload = self._safe_json(response)
        text = self._extract_transcript(payload)
        if not text:
            raise SarvamClientError("Sarvam STT response did not include transcript text")

        return {
            "text": text,
            "raw": payload,
        }

    def synthesize(
        self,
        text: str,
        language_code: str = "en-IN",
        speaker: str = "anushka",
        model: str = "bulbul:v2",
    ) -> dict[str, Any]:
        payload = {
            "text": text,
            "target_language_code": language_code,
            "language_code": language_code,
            "speaker": speaker,
            "model": model,
        }

        headers = self._headers()
        headers["Content-Type"] = "application/json"

        try:
            response = requests.post(
                self.tts_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = self._extract_error_detail(exc)
            raise SarvamClientError(f"Sarvam TTS request failed: {detail}") from exc

        content_type = response.headers.get("content-type", "audio/mpeg")
        if content_type.startswith("audio/"):
            return {
                "audio_bytes": response.content,
                "mime_type": content_type,
            }

        data = self._safe_json(response)
        audio_b64 = self._extract_audio_base64(data)
        if not audio_b64:
            raise SarvamClientError("Sarvam TTS response did not include audio data")

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as exc:
            raise SarvamClientError("Failed to decode Sarvam TTS audio payload") from exc

        mime_type = data.get("mime_type") or data.get("content_type") or "audio/mpeg"
        return {
            "audio_bytes": audio_bytes,
            "mime_type": mime_type,
        }

    def _safe_json(self, response: requests.Response) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            raise SarvamClientError("Sarvam response was not valid JSON") from exc

    def _extract_error_detail(self, exc: requests.RequestException) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return str(exc)

        body_text = (response.text or "").strip()
        if len(body_text) > 800:
            body_text = body_text[:800] + "..."

        if body_text:
            return f"{response.status_code} {response.reason}. Response body: {body_text}"
        return str(exc)

    def _extract_transcript(self, payload: dict[str, Any]) -> str:
        direct_keys = [
            "transcript",
            "text",
            "transcription",
            "recognized_text",
            "output_text",
        ]

        for key in direct_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("data", "result", "results", "output"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                nested_text = self._extract_transcript(nested)
                if nested_text:
                    return nested_text
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        nested_text = self._extract_transcript(item)
                        if nested_text:
                            return nested_text

        return ""

    def _extract_audio_base64(self, payload: dict[str, Any]) -> str:
        direct_keys = [
            "audio",
            "audio_base64",
            "audioContent",
            "output_audio",
            "speech",
        ]
        for key in direct_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("audios", "data", "result", "output"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                    if isinstance(item, dict):
                        nested = self._extract_audio_base64(item)
                        if nested:
                            return nested
            if isinstance(value, dict):
                nested = self._extract_audio_base64(value)
                if nested:
                    return nested

        return ""