from pathlib import Path


class GnaniSpeechToText:
    def __init__(self, language_code: str = "en-IN") -> None:
        try:
            from gnani.stt import GnaniSTTClient
        except ImportError as exc:
            raise RuntimeError(
                "Gnani SDK is not installed. Install requirements.txt first."
            ) from exc
        self._client = GnaniSTTClient()
        self.language_code = language_code

    def transcribe(self, audio_path: Path, request_id: str | None = None) -> dict:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        result = self._client.transcribe(
            audio_path,
            language_code=self.language_code,
            format="transcribe",
            request_id=request_id,
        )
        transcript = result.get("transcript")
        if not isinstance(transcript, str):
            raise RuntimeError("Gnani STT returned no transcript")
        return result
