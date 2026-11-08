from pathlib import Path

import pytest

from sanad.embeddings import HashEmbedder
from sanad.index import HybridIndex
from sanad.kb import build_chunks

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def chunks():
    return build_chunks(KB_DIR)


@pytest.fixture(scope="session")
def index(chunks, embedder):
    return HybridIndex.build(chunks, embedder)
