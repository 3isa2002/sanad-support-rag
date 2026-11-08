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


@pytest.fixture
def handoffs():
    from sanad.handoff import HandoffStore

    return HandoffStore(":memory:")


@pytest.fixture
def make_assistant(index, embedder, handoffs):
    from sanad.assistant import Assistant

    def _make(llm, tools=None, min_score=0.25):
        return Assistant(index, embedder, llm, handoffs, top_k=4, min_score=min_score, tools=tools)

    return _make
