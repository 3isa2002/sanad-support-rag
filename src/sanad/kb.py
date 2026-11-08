"""Load Markdown knowledge-base documents and split them into heading-aware chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .textnorm import detect_language

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Document:
    id: str
    title: str
    lang: str
    text: str
    url: str = ""
    path: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    lang: str
    text: str
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def embed_text(self) -> str:
        """Text used for embeddings / BM25: title + section give useful context."""
        return f"{self.title} — {self.section}\n{self.text}"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    title = meta.get("title")
    if not title:
        heading = next((m.group(2) for m in map(_HEADING.match, body.splitlines()) if m), None)
        title = heading or path.stem
    return Document(
        id=meta.get("id", path.stem),
        title=title,
        lang=meta.get("lang") or detect_language(body),
        text=body,
        url=meta.get("url", ""),
        path=str(path),
    )


def load_documents(kb_dir: Path) -> list[Document]:
    docs = [load_document(p) for p in sorted(Path(kb_dir).rglob("*.md"))]
    ids = [d.id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate document ids in KB: {sorted(dupes)}")
    return docs


def chunk_document(doc: Document, max_chars: int = 1200) -> list[Chunk]:
    """Split on level-2+ headings; each section becomes one chunk (split further
    on paragraph boundaries if it exceeds ``max_chars``)."""
    sections: list[tuple[str, list[str]]] = []
    current_title, current_lines = doc.title, []
    for line in doc.text.splitlines():
        m = _HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1:
                continue  # document title, already in metadata
            if any(s.strip() for s in current_lines):
                sections.append((current_title, current_lines))
            current_title, current_lines = heading, []
        else:
            current_lines.append(line)
    if any(s.strip() for s in current_lines):
        sections.append((current_title, current_lines))

    chunks: list[Chunk] = []
    for section, lines in sections:
        for piece in _split_long("\n".join(lines).strip(), max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.id}#{len(chunks)}",
                    doc_id=doc.id,
                    title=doc.title,
                    section=section,
                    lang=doc.lang,
                    text=piece,
                    url=doc.url,
                )
            )
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        if buf and len(buf) + len(para) + 2 > max_chars:
            pieces.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def build_chunks(kb_dir: Path) -> list[Chunk]:
    return [c for doc in load_documents(kb_dir) for c in chunk_document(doc)]
