#!/usr/bin/env python3
"""Evaluate the text agent on the supplied practice questions and conversations."""
from argparse import ArgumentParser
import csv
import os
import time
from pathlib import Path

from src.agent.graph import build_graph
from src.config import settings
from src.evaluation.scoring import score_rows
from src.output import write_json
from src.rag.documents import load_knowledge_base
from src.rag.embeddings import GeminiEmbeddings
from src.rag.qdrant_store import COLLECTION, LocalQdrantStore
from src.rag.retriever import RerankingRetriever


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _agent(rebuild_index: bool):
    store = LocalQdrantStore(
        settings.qdrant_path,
        GeminiEmbeddings(settings.embedding_model),
    )
    if rebuild_index or not store.client.collection_exists(COLLECTION):
        store.rebuild(load_knowledge_base(settings.kb_dir))
    return build_graph(RerankingRetriever(store))


def _initial_state() -> dict:
    return {
        "messages": [],
        "order_verified": False,
        "verification_attempts": 0,
        "tool_calls": [],
        "history": [],
    }


def _paced_invoke(agent, state: dict) -> dict:
    delay = float(os.getenv("GROQ_MIN_REQUEST_INTERVAL", "2.1"))
    last_call = getattr(_paced_invoke, "_last_call", 0.0)
    wait = delay - (time.monotonic() - last_call)
    if wait > 0:
        time.sleep(wait)
    result = agent.invoke(state)
    _paced_invoke._last_call = time.monotonic()
    return result


def _evaluate_questions(
    agent,
    use_asr: bool,
    limit: int | None = None,
    question_ids: set[str] | None = None,
) -> dict:
    rows = _rows(settings.practice_dir / "questions.csv")
    if question_ids:
        rows = [row for row in rows if row["id"] in question_ids]
    if limit is not None:
        rows = rows[:limit]
    outputs = [
        _paced_invoke(
            agent,
            {**_initial_state(), "transcript": row["asr_text" if use_asr else "question"]},
        )
        for row in rows
    ]
    return score_rows(rows, outputs).as_dict()


def _evaluate_conversations(
    agent,
    conversation_limit: int | None = None,
    conversation_ids: set[str] | None = None,
) -> dict:
    rows = _rows(settings.practice_dir / "conversations.csv")
    if conversation_ids:
        rows = [row for row in rows if row["conversation_id"] in conversation_ids]
    if conversation_limit is not None:
        conversation_ids = sorted({row["conversation_id"] for row in rows})
        rows = [
            row for row in rows
            if row["conversation_id"] in conversation_ids[:conversation_limit]
        ]
    outputs: list[dict] = []
    scored_rows: list[dict] = []
    current_id = None
    state = _initial_state()
    for row in sorted(rows, key=lambda item: (item["conversation_id"], int(item["turn"]))):
        if row["conversation_id"] != current_id:
            current_id = row["conversation_id"]
            state = _initial_state()
        state = _paced_invoke(agent, {**state, "transcript": row["customer_message"]})
        outputs.append(state)
        scored_rows.append(row)
    return score_rows(scored_rows, outputs).as_dict()


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--asr", action="store_true", help="Evaluate asr_text instead of question")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N question rows")
    parser.add_argument(
        "--conversation-limit",
        type=int,
        help="Evaluate only the first N conversation IDs",
    )
    parser.add_argument(
        "--question-ids",
        help="Comma-separated question IDs for a curated evaluation",
    )
    parser.add_argument(
        "--conversation-ids",
        help="Comma-separated conversation IDs for a curated evaluation",
    )
    args = parser.parse_args()

    agent = _agent(args.rebuild_index)
    result = {
        "questions": _evaluate_questions(
            agent,
            args.asr,
            args.limit,
            set(args.question_ids.split(",")) if args.question_ids else None,
        ),
        "conversations": _evaluate_conversations(
            agent,
            args.conversation_limit,
            set(args.conversation_ids.split(",")) if args.conversation_ids else None,
        ),
        "model": settings.groq_model,
        "embedding_model": settings.embedding_model,
    }
    write_json(args.output, result)
    print(f"Wrote evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
