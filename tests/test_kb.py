from pathlib import Path

import pytest

from sanad.kb import Document, chunk_document, load_documents, parse_front_matter


def test_front_matter_parsed():
    meta, body = parse_front_matter("---\nid: x\ntitle: T: sub\n---\n# Hello\n")
    assert meta == {"id": "x", "title": "T: sub"}
    assert body.startswith("# Hello")


def test_chunks_follow_headings():
    doc = Document(id="d", title="Doc", lang="en", text="# Doc\nintro\n## A\none\n## B\ntwo\n")
    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["Doc", "A", "B"]
    assert chunks[1].text == "one" and chunks[1].chunk_id == "d#1"


def test_long_sections_are_split():
    para = "word " * 100
    doc = Document(id="d", title="Doc", lang="en", text=f"## S\n{para}\n\n{para}\n\n{para}")
    chunks = chunk_document(doc, max_chars=700)
    assert len(chunks) >= 2
    assert all(c.section == "S" for c in chunks)


def test_sample_kb_loads(chunks):
    langs = {c.lang for c in chunks}
    assert langs == {"ar", "en"}
    assert all(c.url.startswith("https://") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_duplicate_ids_rejected(tmp_path: Path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text("---\nid: same\n---\n## X\ntext\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_documents(tmp_path)
