from dataclasses import dataclass
from pathlib import Path
import re


_META_RE = re.compile(r"^([A-Za-z_]+):\s*(.*?)\s*$")
_SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*?)\s*$")


@dataclass(frozen=True)
class KnowledgeChunk:
    text: str
    section_id: str
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_from: str
    source_file: str
    priority: int

    @property
    def metadata(self) -> dict:
        return {
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "status": self.status,
            "effective_from": self.effective_from,
            "source_file": self.source_file,
            "priority": self.priority,
        }


def _priority(doc_type: str, status: str) -> int:
    if status == "superseded":
        return 6
    return {
        "policy": 1,
        "agent_guide": 2,
        "help": 3,
        "faq": 4,
        "marketing": 5,
        "meta": 1,
    }.get(doc_type, 5)


def parse_document(path: Path) -> list[KnowledgeChunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    for line in lines:
        if line.startswith("## "):
            break
        match = _META_RE.match(line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()

    chunks: list[KnowledgeChunk] = []
    current_id: str | None = None
    current_title = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            return
        chunks.append(
            KnowledgeChunk(
                text=text,
                section_id=current_id,
                doc_id=metadata.get("doc_id", path.stem),
                title=metadata.get("title", path.stem),
                doc_type=metadata.get("doc_type", "unknown"),
                status=metadata.get("status", "current"),
                effective_from=metadata.get("effective_from", ""),
                source_file=path.name,
                priority=_priority(
                    metadata.get("doc_type", "unknown"),
                    metadata.get("status", "current"),
                ),
            )
        )

    for line in lines:
        match = _SECTION_RE.match(line)
        if match:
            flush()
            current_id = match.group(1)
            current_title = match.group(2)
            current_lines = [f"## [{current_id}] {current_title}"]
        elif current_id is not None:
            current_lines.append(line)
    flush()
    return chunks


def load_knowledge_base(kb_dir: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(parse_document(path))
    if not chunks:
        raise ValueError(f"No knowledge-base sections found in {kb_dir}")
    section_ids = [chunk.section_id for chunk in chunks]
    duplicates = {sid for sid in section_ids if section_ids.count(sid) > 1}
    if duplicates:
        raise ValueError(f"Duplicate knowledge-base section IDs: {sorted(duplicates)}")
    return chunks
