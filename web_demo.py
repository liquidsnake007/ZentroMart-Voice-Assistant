#!/usr/bin/env python3
"""Serve the local ZentroMart voice assistant demo."""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import uuid

from src.agent.graph import build_graph
from src.config import settings
from src.rag.embeddings import GeminiEmbeddings
from src.rag.qdrant_store import LocalQdrantStore
from src.rag.retriever import RerankingRetriever
from src.voice.gnani_stt import GnaniSpeechToText
from src.voice.gnani_tts import GnaniTextToSpeech


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEMO_AUDIO_DIR = ROOT / "runtime" / "web_demo"
MAX_AUDIO_BYTES = 15 * 1024 * 1024


def _json_safe(value):
    """Convert library numeric scalars into values supported by JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    return value


def build_services():
    store = LocalQdrantStore(
        settings.qdrant_path,
        GeminiEmbeddings(settings.embedding_model),
    )
    if not store.client.collection_exists("zentromart_kb"):
        raise RuntimeError(
            f"Qdrant collection 'zentromart_kb' was not found at "
            f"{settings.qdrant_path}. Include the persisted vector store."
        )
    return (
        build_graph(RerankingRetriever(store)),
        GnaniSpeechToText(language_code=settings.gnani_stt_language),
        GnaniTextToSpeech(
            voice=settings.gnani_tts_voice,
            model=settings.gnani_tts_model,
            language=settings.gnani_tts_language,
        ),
    )


GRAPH, STT, TTS = build_services()


class DemoHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if self.path == "/styles.css":
            self._serve_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if self.path == "/app.js":
            self._serve_file(WEB_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/process":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_AUDIO_BYTES * 2:
                raise ValueError("Audio request is empty or too large.")
            payload = json.loads(self.rfile.read(length))
            audio = base64.b64decode(payload["audio"], validate=True)
            if len(audio) > MAX_AUDIO_BYTES:
                raise ValueError("Audio file exceeds the 15 MB demo limit.")
            suffix = Path(payload.get("filename", "recording.wav")).suffix.lower()
            if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}:
                suffix = ".wav"
            DEMO_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = DEMO_AUDIO_DIR / f"{uuid.uuid4().hex}{suffix}"
            audio_path.write_bytes(audio)

            stt_result = STT.transcribe(audio_path, request_id=f"web-stt-{audio_path.stem}")
            transcript = stt_result["transcript"].strip()
            state = {
                "transcript": transcript,
                "messages": [],
                "history": payload.get("history", []),
                "order_verified": bool(payload.get("order_verified", False)),
                "verification_attempts": int(payload.get("verification_attempts", 0)),
                "tool_calls": [],
            }
            if transcript:
                result = GRAPH.invoke(state)
                response = result["response_text"]
            else:
                result = {
                    "response_text": "Sorry, I didn't catch that. Could you say that again?",
                    "source_ids": [],
                    "retrieved_context": [],
                    "tool_calls": [],
                    "decision": "clarify",
                }
                response = result["response_text"]

            response_path = DEMO_AUDIO_DIR / f"{audio_path.stem}-response.wav"
            TTS.synthesize(
                response,
                response_path,
                request_id=f"web-tts-{audio_path.stem}",
            )
            reply = {
                "transcript": transcript,
                "response": response,
                "decision": result.get("decision"),
                "sources": result.get("source_ids", []),
                "retrieved_context": result.get("retrieved_context", []),
                "tool_calls": result.get("tool_calls", []),
                "history": result.get("history", []),
                "order_verified": result.get("order_verified", False),
                "verification_attempts": result.get("verification_attempts", 0),
                "audio": base64.b64encode(response_path.read_bytes()).decode("ascii"),
            }
            self._send(
                200,
                json.dumps(_json_safe(reply)).encode("utf-8"),
                "application/json; charset=utf-8",
            )
        except Exception as exc:
            self._send(
                400,
                json.dumps({"error": str(exc)}).encode("utf-8"),
                "application/json; charset=utf-8",
            )

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        self._send(200, path.read_bytes(), content_type)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[web-demo] {format % args}\n")


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    print(f"ZentroMart demo running at http://{host}:{port}")
    ThreadingHTTPServer((host, port), DemoHandler).serve_forever()


if __name__ == "__main__":
    main()
