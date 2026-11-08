# Sanad (سَنَد) — Project Plan

## Overview
Sanad is a bilingual (Arabic / English) customer-support assistant for an online
marketplace. It answers customer questions from the marketplace's help docs and
policies using retrieval-augmented generation (RAG), always shows the source it
used, and hands the conversation to a human agent instead of guessing when it is
not confident. It can also call the marketplace backend API to look up the live
status of an order or a listing. Everything runs locally: the LLM and the
embedding model are served by Ollama, and no data leaves the machine.

## Goals
- Answer policy / how-to questions in the language the customer wrote in (Arabic,
  including Egyptian dialect, or English), grounded only in the knowledge base.
- Every grounded answer carries at least one citation (document + section).
- Low-confidence or out-of-scope questions produce a handoff ticket, not an
  invented answer.
- Order and listing status questions are answered from the backend API, not
  from the LLM's imagination.
- Runs fully offline with Ollama; `pytest` passes without Ollama installed.
- A small evaluation set reports retrieval hit-rate and handoff accuracy.

## Non-goals
- No real marketplace integration — a mock backend API stands in for it.
- No authentication / customer identity verification (the MVP trusts the order
  ID the customer types; see Risks).
- No live agent console — handoff tickets are stored and listed via the API.
- No fine-tuning, no cloud LLM providers.
- No multi-turn memory beyond the current message (each `/chat` call is
  stateless apart from an optional `conversation_id` used on tickets).

## Assumptions
- **Models:** `qwen2.5:7b` for chat (solid Arabic, runs on a laptop GPU / 16 GB
  RAM) and `bge-m3` for embeddings (multilingual, strong on Arabic). Both are
  configurable via environment variables.
- **Hybrid retrieval:** dense (Ollama embeddings) + lexical BM25 fused with
  Reciprocal Rank Fusion. BM25 on normalized Arabic text is robust for short
  dialect queries and keeps working when the embedding model is missing.
- **Offline fallback embedder:** a deterministic hashed character n-gram
  embedder is used in tests and when `SANAD_EMBEDDER=hash`, so the project is
  usable and testable without Ollama.
- **Vector store:** a small in-process NumPy index persisted to disk. A support
  knowledge base is hundreds of chunks, not millions; no vector DB needed.
- **Tool routing is deterministic:** order / listing IDs are extracted with
  regexes (including Arabic-Indic digits) rather than relying on a 7B model's
  function-calling, which is unreliable at that size. The LLM only phrases the
  answer from the returned data, and a template fallback exists.
- **Confidence gating** happens at two levels: (1) retrieval — if the best
  chunk's relevance is below a threshold, hand off before calling the LLM;
  (2) generation — the model is told to output a `HANDOFF` marker when the
  context does not contain the answer, and answers without a valid citation are
  treated as handoffs. The retrieval threshold defaults per embedder
  (`bge-m3`: 0.45, hash: 0.25) because cosine ranges differ; override with
  `SANAD_MIN_SCORE`.
- **ID formats:** orders look like `ORD-10023`, listings like `LST-5001`; bare
  numbers next to "order/اوردر/طلب" or "listing/إعلان" are also accepted.
- **Sample knowledge base:** realistic but fictional policies for a fictional
  marketplace ("Souqna"), written in both Arabic and English.
- **Ollama is not reachable in the build environment**, so the Ollama clients
  are tested against mocked HTTP responses; end-to-end runs against a real
  Ollama are documented but not verified here.

## Features
### MVP
- **KB ingestion** — Markdown docs with front matter → heading-aware chunks →
  embeddings + BM25 index, persisted to `data/index/`.
- **Arabic normalization** — strip diacritics/tatweel, unify alef/yaa/taa
  marbuta forms, Arabic-Indic → Latin digits; used for lexical search and ID
  extraction.
- **Hybrid retrieval** — dense + BM25 with RRF, top-k chunks with scores.
- **Grounded answers with citations** — LLM answers only from retrieved
  context in the user's language, citing `[1]`, `[2]`; response includes
  structured sources.
- **Human handoff** — low retrieval confidence, model `HANDOFF`, missing
  citations, LLM or backend errors, or an explicit "talk to a human" request create a
  ticket in SQLite and return a polite bilingual message with the ticket ID.
- **Backend tools** — order status and listing status lookups through an HTTP
  client against a configurable backend; includes a mock backend service.
- **HTTP API** — FastAPI: `POST /chat`, `GET /handoffs`, `POST /admin/reindex`,
  `GET /health`.
- **CLI** — `sanad ingest | search | ask | serve | eval`.
- **Minimal web chat page** — RTL-aware single HTML page served at `/`.
- **Evaluation** — golden question set; reports retrieval hit@k and handoff
  precision/recall.

### Future work
- Multi-turn conversation memory and follow-up question rewriting.
- Customer authentication so order lookups are scoped to the logged-in user.
- Agent console for handoff tickets (assign, reply, close) and feeding resolved
  tickets back into the KB.
- Re-ranking with a cross-encoder (e.g. `bge-reranker-v2-m3`).
- Streaming responses.
- LLM-based tool calling for more actions (cancel order, open return request).
- PDF / HTML ingestion and incremental re-indexing.
- Dialect-aware query rewriting (Egyptian → MSA) before retrieval.

## Tech Stack
| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11+ | Standard for RAG tooling, simple to run locally |
| LLM runtime | Ollama (`/api/chat`, `/api/embed`) | Local, free, one binary, easy model swaps |
| Chat model | `qwen2.5:7b` (configurable) | Good Arabic + English at laptop size |
| Embeddings | `bge-m3` via Ollama (hash fallback) | Multilingual, strong Arabic retrieval |
| Lexical search | Custom BM25 on normalized tokens | No extra deps, Arabic-aware |
| Vector store | NumPy matrix + JSON on disk | KB is small; zero-ops |
| API | FastAPI + Uvicorn | Typed, fast, auto OpenAPI docs |
| HTTP client | httpx | Sync client with easy mocking (`MockTransport`) |
| Tickets | SQLite (stdlib) | Zero-ops persistence |
| Tests | pytest | — |
| Packaging | `pyproject.toml`, Docker Compose | `pip install -e .` or `docker compose up` |

## Architecture
```
              ┌──────────────────────────── Sanad service (FastAPI) ───────────────────────────┐
 customer ──▶ │ /chat ─▶ language detect ─▶ router                                             │
              │                              ├─ human requested? ──────────────▶ Handoff store │──▶ SQLite
              │                              ├─ order/listing ID? ─▶ Backend client ─────────┐ │
              │                              │                         (httpx) ──▶ Backend API│ │
              │                              └─ else ─▶ Retriever (dense + BM25, RRF)        │ │
              │                                            │ low score? ──▶ Handoff store    │ │
              │                                            ▼                                 │ │
              │                                        Generator ◀───────────────────────────┘ │
              │                                     (Ollama chat, grounded prompt)             │
              │                                            │ HANDOFF / no citation ─▶ Handoff  │
              │                                            ▼                                   │
              │                                answer + sources + tool_calls                   │
              └────────────────────────────────────────────────────────────────────────────────┘
   data/kb/*.md ──▶ ingest (chunk + embed + BM25) ──▶ data/index/{vectors.npy, chunks.json}
```
- **Ollama** runs on `OLLAMA_BASE_URL` (default `http://localhost:11434`).
- **Backend API** at `SANAD_BACKEND_URL` (default the bundled mock on `:8001`).
- **Index** in `data/index/`, **tickets** in `data/sanad.db`.

## Data Model
- **Document** (Markdown front matter): `id`, `title`, `lang` (`ar`/`en`),
  `url` (optional, for the citation link).
- **Chunk**: `chunk_id`, `doc_id`, `title`, `section`, `lang`, `text`, `url`.
- **Source** (in responses): `doc_id`, `title`, `section`, `url`, `score`, `snippet`.
- **HandoffTicket** (SQLite `handoffs`): `id` (e.g. `HT-000012`), `created_at`,
  `conversation_id`, `language`, `question`, `reason`
  (`human_requested` | `low_confidence` | `model_uncertain` | `no_citation` |
  `llm_error` | `backend_error`). An unknown order/listing ID is *not* a handoff:
  the customer is asked to re-check the number, `context` (JSON), `status` (`open`).
- **Order** (backend): `id`, `status`, `items`, `total`, `currency`,
  `created_at`, `estimated_delivery`, `tracking_number`, `last_update`.
- **Listing** (backend): `id`, `title`, `status`
  (`active`/`under_review`/`rejected`/`expired`/`sold`), `reason`, `updated_at`.

## Project Structure
```
sanad-support-rag/
├── PLAN.md  README.md  LICENSE  pyproject.toml  .env.example
├── docker-compose.yml  Dockerfile
├── data/
│   ├── kb/                 # sample policy docs (ar + en)
│   ├── mock_backend/       # seed orders & listings JSON
│   └── eval/questions.jsonl
├── src/sanad/
│   ├── config.py           # settings from env
│   ├── textnorm.py         # Arabic normalization, tokenization, lang detect
│   ├── kb.py               # load docs, chunking
│   ├── embeddings.py       # Ollama + hash embedders
│   ├── bm25.py
│   ├── index.py            # build/save/load index, hybrid search
│   ├── llm.py              # Ollama chat client + fake
│   ├── prompts.py
│   ├── handoff.py          # SQLite ticket store
│   ├── backend.py          # backend API client
│   ├── tools.py            # ID extraction + tool results
│   ├── assistant.py        # orchestration
│   ├── messages.py         # bilingual canned messages
│   ├── api.py              # FastAPI app
│   ├── static/index.html   # chat UI
│   ├── eval.py
│   └── cli.py
├── mock_backend/app.py     # mock marketplace API
└── tests/
```

## Phases
### Phase 1 — Scaffolding, KB and hybrid retrieval
- [x] `pyproject.toml`, package layout, `.env.example`, config
- [x] Arabic text normalization, tokenizer and language detection
- [x] Sample KB (Arabic + English policies) and heading-aware chunker
- [x] Embedders: Ollama `/api/embed` client and offline hash embedder
- [x] BM25 + dense index with RRF fusion, persisted to disk
- [x] CLI: `sanad ingest`, `sanad search`
- [x] Unit tests for normalization, chunking, BM25, retrieval
**Done when:** `pytest` passes and `sanad search "ازاي ارجع منتج"` (hash embedder)
returns a returns-policy chunk first.

### Phase 2 — Grounded answers, citations and handoff
- [x] Ollama chat client (+ fake LLM for tests)
- [x] Grounded bilingual prompt with numbered context and `HANDOFF` rule
- [x] Citation parsing → structured sources; uncited answers → handoff
- [x] Retrieval confidence threshold → handoff before generation
- [x] Explicit human-request detection (ar/en)
- [x] SQLite handoff ticket store
- [x] CLI: `sanad ask`
**Done when:** tests cover answered, low-confidence, model-uncertain, no-citation
and human-requested paths; `sanad ask` with the fake LLM returns an answer with
sources.

### Phase 3 — Backend tools (order & listing status)
- [ ] Mock backend FastAPI app with seed data
- [ ] Backend HTTP client with timeouts and error mapping
- [ ] Order / listing ID extraction (Latin + Arabic-Indic digits, ar/en keywords)
- [ ] Status answers phrased in the user's language (LLM phrasing + template fallback)
- [ ] Not-found and backend-down paths (not found → ask to check ID; down → handoff)
**Done when:** tests pass against the mock backend via `httpx` transport, and
"فين الاوردر ORD-10023؟" returns the real status from the mock.

### Phase 4 — HTTP API, web chat and Docker
- [ ] FastAPI app: `/chat`, `/handoffs`, `/admin/reindex`, `/health`
- [ ] Minimal RTL-aware chat page at `/`
- [ ] `sanad serve` command
- [ ] Dockerfile + `docker-compose.yml` (ollama, sanad, mock backend)
**Done when:** API tests pass with `TestClient`; server boots locally and `/chat`
answers with the hash embedder + fake LLM. Docker Compose is written but not
verified here (no Docker / Ollama in the build environment).

### Phase 5 — Evaluation and polish
- [ ] Golden question set (ar + en, answerable + should-handoff)
- [ ] `sanad eval`: retrieval hit@k and handoff precision/recall
- [ ] README with setup, usage, examples, architecture
**Done when:** `sanad eval` runs offline and prints metrics; README complete.

## Testing Strategy
- Unit tests for pure logic: normalization, tokenization, chunking, BM25,
  RRF, ID extraction, citation parsing, handoff store.
- Ollama clients tested with `httpx.MockTransport` (request shape + parsing).
- Assistant flows tested with a scripted fake LLM and the hash embedder, so
  the whole pipeline runs deterministically without Ollama.
- Backend tools tested against the real mock-backend FastAPI app mounted via
  an in-process transport.
- API tested with FastAPI `TestClient`.
- `sanad eval` gives a quality signal beyond unit tests (run with real Ollama
  for meaningful generation numbers).

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Small local model hallucinates | Retrieval threshold, strict prompt, `HANDOFF` marker, citation required |
| Model answers in the wrong language | Language detected in code and forced in the prompt; canned messages per language |
| Egyptian dialect queries miss MSA docs | Arabic normalization, BM25 + multilingual dense retrieval, dialect phrasing in KB |
| Anyone can look up any order ID | Documented non-goal; future work adds auth. Mock data only |
| Ollama not running / model not pulled | Clear errors in `/health`; hash embedder fallback for retrieval; LLM failure → handoff |
| Prompt injection inside KB or user text | Context is quoted as data, system prompt forbids following instructions inside it; KB is trusted/curated |
| Threshold tuning differs per embedder | Threshold configurable; eval script reports handoff precision/recall to tune it |

## How to Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b && ollama pull bge-m3
sanad ingest
python -m mock_backend          # terminal 1 — mock marketplace API on :8001
sanad serve                     # terminal 2 — Sanad on :8000, open http://localhost:8000
```
Offline (no Ollama): `SANAD_EMBEDDER=hash SANAD_LLM=fake sanad serve`.
See README for details.
