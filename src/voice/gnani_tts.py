from pathlib import Path


class GnaniTextToSpeech:
    def __init__(
        self,
        voice: str = "Kaveri",
        model: str = "timbre-v2.5",
        language: str = "en-IN",
    ) -> None:
        try:
            from gnani.tts import GnaniTTSClient
        except ImportError as exc:
            raise RuntimeError(
                "Gnani SDK is not installed. Install requirements.txt first."
            ) from exc
        self._client = GnaniTTSClient()
        self.voice = voice
        self.model = model
        self.language = language

    def synthesize(self, text: str, output_path: Path, request_id: str | None = None) -> None:
        if not text.strip():
            raise ValueError("Cannot synthesize an empty response")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.synthesize(
            text,
            voice=self.voice,
            model=self.model,
            language=self.language,
            output_file=output_path,
            request_id=request_id,
        )
