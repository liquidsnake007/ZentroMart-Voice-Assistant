from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _groq_model() -> str:
    configured = os.getenv("GROQ_MODEL", "gpt-oss-120b")
    if configured == "gpt-oss-120b":
        return "openai/gpt-oss-120b"
    return configured


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = ROOT / "src" / "data"
    kb_dir: Path = ROOT / "src" / "kb"
    practice_dir: Path = ROOT / "src" / "practice"
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", str(ROOT / "runtime" / "qdrant")))
    groq_model: str = _groq_model()
    embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    )
    gnani_stt_language: str = os.getenv("GNANI_STT_LANGUAGE", "en-IN")
    gnani_tts_language: str = os.getenv("GNANI_TTS_LANGUAGE", "en-IN")
    gnani_tts_voice: str = os.getenv("GNANI_TTS_VOICE", "Kaveri")
    gnani_tts_model: str = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")


settings = Settings()
