#!/usr/bin/env python3
"""Run the ZentroMart voice pipeline over an input audio directory."""
from argparse import ArgumentParser
from pathlib import Path
import sys

from src.config import settings
from src.output import write_json, write_text
from src.agent.graph import build_graph
from src.rag.embeddings import GeminiEmbeddings
from src.rag.qdrant_store import LocalQdrantStore
from src.rag.retriever import RerankingRetriever
from src.voice.gnani_stt import GnaniSpeechToText
from src.voice.gnani_tts import GnaniTextToSpeech


SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--input_dir", required=True, type=Path)
    parser.add_argument("--output-dir", "--output_dir", required=True, type=Path)
    parser.add_argument("--language", default=settings.gnani_stt_language)
    parser.add_argument("--voice", default=settings.gnani_tts_voice)
    parser.add_argument("--tts-model", default=settings.gnani_tts_model)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    audio_files = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO
    )
    if not audio_files:
        print(f"No supported audio files found in {args.input_dir}", file=sys.stderr)
        return 2

    store = LocalQdrantStore(
        settings.qdrant_path,
        GeminiEmbeddings(settings.embedding_model),
    )
    if not store.client.collection_exists("zentromart_kb"):
        print(
            f"Qdrant collection 'zentromart_kb' was not found at "
            f"{settings.qdrant_path}. Include the persisted vector store.",
            file=sys.stderr,
        )
        return 2
    graph = build_graph(RerankingRetriever(store))
    stt = GnaniSpeechToText(language_code=args.language)
    tts = GnaniTextToSpeech(
        voice=args.voice,
        model=args.tts_model,
        language=settings.gnani_tts_language,
    )
    summary = {"processed": [], "failed": []}
    for audio_path in audio_files:
        item_dir = args.output_dir / audio_path.stem
        try:
            stt_result = stt.transcribe(audio_path, request_id=f"stt-{audio_path.stem}")
            transcript = stt_result["transcript"].strip()
            write_text(item_dir / "transcript.txt", transcript)
            state = (
                graph.invoke(
                    {
                        "transcript": transcript,
                        "messages": [],
                        "order_verified": False,
                        "verification_attempts": 0,
                        "tool_calls": [],
                    }
                )
                if transcript
                else {
                    "response_text": (
                        "Sorry, I didn't catch that. Could you say that again?"
                    ),
                    "source_ids": [],
                    "retrieved_context": [],
                    "tool_calls": [],
                    "decision": "clarify",
                }
            )
            response = (
                "Sorry, I didn't catch that. Could you say that again?"
                if not transcript
                else state["response_text"]
            )
            write_text(item_dir / "response.txt", response)
            tts.synthesize(
                response,
                item_dir / "response.wav",
                request_id=f"tts-{audio_path.stem}",
            )
            write_json(
                item_dir / "sources.json",
                {
                    "sources": state.get("source_ids", []),
                    "retrieved_context": state.get("retrieved_context", []),
                },
            )
            write_json(
                item_dir / "metadata.json",
                {
                    "input_file": audio_path.name,
                    "language": args.language,
                    "stt_provider": "gnani",
                    "tts_provider": "gnani",
                    "tts_model": args.tts_model,
                    "decision": state.get("decision"),
                    "source_ids": state.get("source_ids", []),
                    "tool_calls": state.get("tool_calls", []),
                },
            )
            summary["processed"].append(audio_path.name)
        except Exception as exc:
            summary["failed"].append({"input_file": audio_path.name, "error": str(exc)})
            print(f"Failed {audio_path.name}: {exc}", file=sys.stderr)
    write_json(args.output_dir / "run_summary.json", summary)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
