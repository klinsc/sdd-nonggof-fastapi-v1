# 🤖 น้องกอฟ — PEA Substation AI Assistant (Backend)

> AI-powered Chatbot สำหรับสนับสนุนงานมาตรฐานและการออกแบบสถานีไฟฟ้าแรงสูง 115kV ของ กฟภ.
> พัฒนาด้วยเทคโนโลยี **OCR + RAG + LLM** เพื่อช่วยค้นหา อ้างอิง และสรุปข้อมูลจากเอกสารมาตรฐานได้อย่างรวดเร็วและแม่นยำ

---

> ⚠️ **Breaking change for existing clients.** `GET /qa/stream?token=jessica&...` is removed. The new shape is `POST /qa/stream` with an `X-API-Key` header and a JSON body. **See [MIGRATION.md](MIGRATION.md)** for drop-in cURL/browser replacements and the new status-code contract.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [Building the Vector Index](#building-the-vector-index)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment Profiles](#deployment-profiles)
- [Contact](#contact)

---

## Overview

**น้องกอฟ** is the backend service for PEA's Substation Design & Standards AI Assistant. It exposes a streaming question-answering API that retrieves relevant passages from PEA's 115kV substation standard documents and generates Thai-language responses using a RAG (Retrieval-Augmented Generation) pipeline.

The service is structured around a **hexagonal (ports & adapters) architecture**, which lets us swap the LLM, embeddings, vector store, or OCR provider without touching the application core. Two LLM profiles are supported:

- **`cloud`** (default) — GPT-4o-mini via the OpenAI API.
- **`local`** (planned) — local GGUF (e.g. Typhoon 2.1 Gemma3 4B) via `llama-cpp-python` for air-gapped intranet deployments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│              Web UI  /  Line Chatbot  /  API Client             │
└─────────────────────┬───────────────────────────────────────────┘
                      │  POST /qa/stream  (X-API-Key header)
                      │  Server-Sent Events response
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI app                               │
│  routers/chat.py  →  application/qa_service.py  (use case)      │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              LangGraph Agent (in QAService)             │   │
│   │   query_or_respond  →  retrieve  →  generate            │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │              │                  │                      │
│        ▼              ▼                  ▼                      │
│   LLMProvider   VectorStoreRepo    QueryLogRepository           │
│   (port)        (port)             (port)                       │
│        │              │                  │                      │
│   ┌────┴───┐    ┌─────┴────┐       ┌─────┴────────┐             │
│   │OpenAI  │    │Chroma    │       │SQLite        │             │
│   │ adapter│    │ adapter  │       │ adapter      │             │
│   └────────┘    └──────────┘       └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘

         ┌────────────────────────────────────────┐
         │  Offline ingestion (separate process)  │
         │  python -m app.ingestion.build_index   │
         │     PDFs → OCR → chunks → Chroma       │
         └────────────────────────────────────────┘
```

The API process **only reads** the vector store. Re-indexing is an offline job and does not require a redeploy.

---

## Project Structure

```
app/
├── application/
│   ├── ports.py             # Protocols: LLMProvider, EmbeddingsProvider,
│   │                        # VectorStoreRepository, QueryLogRepository
│   └── qa_service.py        # QAService — owns the LangGraph; ports injected
├── core/
│   ├── config.py            # pydantic Settings (env-driven, fail-fast)
│   ├── lifespan.py          # builds adapters and wires QAService onto app.state
│   └── logging.py           # JSON logger + request_id ContextVar
├── infrastructure/
│   ├── llm/
│   │   ├── openai_chat.py        # OpenAIChatProvider (cloud profile)
│   │   └── llamacpp_local.py     # placeholder for the local GGUF profile
│   ├── embeddings/hf_bge_m3.py   # HuggingFace BAAI/bge-m3 provider
│   ├── vectorstore/chroma_repo.py
│   ├── ocr/ollama_client.py        # Ollama Vision OCR adapter (local)
│   └── persistence/query_log_repo.py  # SQLite-backed user query log
├── ingestion/
│   └── build_index.py       # CLI: PDF → OCR → chunk → embed → Chroma
├── routers/chat.py          # transport-only; calls QAService, BackgroundTasks
├── services/llm_service.py  # legacy LlamaIndex reference (to be ported)
├── dependencies.py          # X-API-Key auth via require_api_key
└── main.py                  # app factory: lifespan, CORS, /healthz, request-id

tests/
├── conftest.py
├── fakes.py                 # FakeListChatModel, FakeVectorStoreRepository
├── test_health_and_auth.py
├── test_qa_service.py
└── test_query_log.py

scripts/
└── run_local_ocr.py         # offline OCR alternative (Ollama + PyMuPDF)

data/      # PDFs and OCR output (gitignored)
storage/   # Chroma index + SQLite query log (gitignored)
```

---

## Tech Stack

| Category        | Technology                                               |
| --------------- | -------------------------------------------------------- |
| Framework       | FastAPI 0.115 + Uvicorn                                  |
| Settings        | pydantic + pydantic-settings                             |
| LLM (cloud)     | GPT-4o-mini via OpenAI API                               |
| LLM (local)     | Typhoon 2.1 Gemma3 4B GGUF via llama-cpp-python *(planned)* |
| OCR             | Ollama + Vision Model (local, via PyMuPDF + `ollama`)    |
| PDF rendering   | PyMuPDF (fitz) — renders pages to PNG for OCR            |
| Embeddings      | BAAI/bge-m3 (HuggingFace, CUDA-accelerated)              |
| RAG orchestrator| LangChain 0.3 + LangGraph                                |
| Vector DB       | ChromaDB 1.0 (persistent local)                          |
| Query log       | SQLite (writes off the request hot path)                 |
| Tests           | pytest + pytest-asyncio                                  |
| Language        | Python 3.11                                              |

---

## Prerequisites

- **Python 3.11** (recommended via Conda)
- **NVIDIA GPU** with CUDA 12.8+ recommended for embedding inference; CPU works but is slower
- **Git**

---

## Installation

### 1. Conda environment

```bash
conda create -n llama-gpu python=3.11 -y
conda activate llama-gpu
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
pip install pydantic-settings
```

The Settings class uses `pydantic-settings`; install it explicitly if it is not already pinned in `requirements.txt`.

### 3. (Optional) Local LLM for the air-gapped profile

Build `llama-cpp-python` with CUDA and drop a GGUF model into `models/`. See [app/services/llm_service.py](app/services/llm_service.py) for the legacy LlamaIndex reference that the `LlamaCppLocalProvider` port will replace.

---

## Configuration

Create a `.env` file in the project root:

```env
# --- Auth & CORS ---
API_KEY="generate-a-secret-at-least-16-chars"
CORS_ORIGINS=["http://localhost:3000","https://sdd.chatbordin.com"]

# --- Runtime profile ---
LLM_PROFILE=cloud           # one of: cloud | local
DEBUG=false

# --- Cloud-profile secrets ---
OPENAI_API_KEY="sk-proj-..."

# --- OCR ingestion (only needed when running build_index) ---
# OLLAMA_HOST="http://localhost:11434"
# OLLAMA_MODEL="qwen2.5-vl"
HF_TOKEN="hf_..."

# --- Optional overrides (defaults shown) ---
# DATASET_NAME=sdd-data
# CHROMA_DIR=storage/chroma_data
# SQLITE_PATH=storage/app.sqlite3
# LLM_MODEL=gpt-4o-mini
# LLM_PROVIDER=openai
# EMBEDDING_MODEL=BAAI/bge-m3
# RETRIEVAL_K=2
# CHUNK_SIZE=3000
# CHUNK_OVERLAP=1000
# MAX_INPUT_CHARS=4000
```

`Settings` validates these at startup and **fails fast** if:
- `API_KEY` is missing or shorter than 16 characters
- `LLM_PROFILE=cloud` and `OPENAI_API_KEY` is unset
- `LLM_PROFILE` is anything other than `cloud` or `local`

Required API keys:

| Key              | Used by                              | Provider                                          |
| ---------------- | ------------------------------------ | ------------------------------------------------- |
| `API_KEY`        | Client auth (`X-API-Key` header)     | You — generate and rotate per environment         |
| `OPENAI_API_KEY` | Cloud LLM profile                    | [platform.openai.com](https://platform.openai.com)|
| `HF_TOKEN`       | HuggingFace embedding model download | [huggingface.co](https://huggingface.co)          |

> **Note:** `TYHOON_API_KEY` is no longer required. OCR is handled locally via Ollama.

---

## Running the Server

The API requires a built vector index. If `storage/chroma_data/` does not exist, run the ingestion CLI first ([Building the Vector Index](#building-the-vector-index)).

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- API docs (Swagger): `http://localhost:8000/docs`
- Health: `http://localhost:8000/healthz` → `{"status": "ok", "ready": true}` once startup completes
- Root: `http://localhost:8000/`

The lifespan hook builds the LangGraph after startup, so cold-start latency is bounded by the embedding-model load and Chroma open — not by OCR.

---

## Building the Vector Index

OCR + chunking + embedding is now a separate offline job. Drop new PDFs into `data/sdd-data/` and run:

```bash
# First-time build (or whenever the corpus changes):
python -m app.ingestion.build_index --rebuild

# Custom directories:
python -m app.ingestion.build_index --source data/standards --ocr-out data/standards_json --rebuild

# Use a specific Ollama model / host:
python -m app.ingestion.build_index --rebuild --model qwen2.5vl:7b --host http://localhost:11434
```

Flags:

| Flag         | Purpose                                                          |
| ------------ | ---------------------------------------------------------------- |
| `--source`   | Directory of source PDFs (default `data/sdd-data`)               |
| `--ocr-out`  | Where to write OCR JSON cache (default `data/sdd-data_json`)     |
| `--rebuild`  | Force re-indexing even if `CHROMA_DIR` already exists            |
| `--model`    | Ollama vision model for OCR (default `qwen2.5-vl`)               |
| `--host`     | Ollama server URL (default `http://localhost:11434`)              |

OCR JSONs are cached on disk; rerunning is idempotent for files that have already been processed.

> **Standalone OCR script:** [scripts/run_local_ocr.py](scripts/run_local_ocr.py) is also available as a lightweight alternative that runs without FastAPI dependencies.

---

## API Reference

All API endpoints (except `/`, `/healthz`, and `/docs`) require:

```
X-API-Key: <your API_KEY>
```

The legacy `?token=jessica` query parameter is **removed**.

### Health

```http
GET /healthz
```

```json
{ "status": "ok", "ready": true }
```

### Chat — Streaming QA

```http
POST /qa/stream
Content-Type: application/json
X-API-Key: <your API_KEY>

{
  "input_message": "มาตรฐานการออกแบบสถานีไฟฟ้า 115kV คืออะไร",
  "session_id": "optional-session-id"
}
```

**Response:** `text/event-stream` (Server-Sent Events).

**Validation:**
- `input_message` must be non-empty and ≤ `MAX_INPUT_CHARS` characters (default 4000). Oversize requests get `413 Request Entity Too Large`.
- The endpoint returns `503` if the vector index has not been built yet.

**Event types:**

| Event                | Data schema                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `message_chunk`      | `{ "type": "ai_chunk", "content": "...", "metadata": {} }`             |
| `tool_message`       | `{ "type": "tool_message", "content": "...", "name": "retrieve", "metadata": {} }` |
| `end_of_ai_response` | `{ "metadata": {} }`                                                   |
| `error`              | `{ "type": "error", "message": "internal_error" }`                     |
| `stream_end`         | `{}`                                                                   |

**cURL:**

```bash
curl -N -X POST http://localhost:8000/qa/stream \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_message":"มาตรฐาน สถานีไฟฟ้า 115kV"}'
```

**Browser (`fetch` + `ReadableStream`):**

Native `EventSource` is GET-only and cannot send headers, so the frontend must use `fetch`:

```javascript
const res = await fetch("http://localhost:8000/qa/stream", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": import.meta.env.VITE_API_KEY,
  },
  body: JSON.stringify({ input_message: "มาตรฐาน สถานีไฟฟ้า 115kV" }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  // SSE frames: split on "\n\n", parse "event:" / "data:" lines as needed.
  console.log(decoder.decode(value));
}
```

Client disconnects are honored — the server cancels the LangGraph run on the next chunk boundary.

---

## Testing

```bash
pytest -q
```

The suite uses `FakeListChatModel` and an in-memory `FakeVectorStoreRepository` (see [tests/fakes.py](tests/fakes.py)), so it runs **without** network, GPU, or a built Chroma index.

What's covered today:
- Health and API-key auth ([tests/test_health_and_auth.py](tests/test_health_and_auth.py))
- `QAService.astream` end-to-end with fakes ([tests/test_qa_service.py](tests/test_qa_service.py))
- SQLite query log writes and indexing ([tests/test_query_log.py](tests/test_query_log.py))

---

## Deployment Profiles

| Target                              | LLM Profile | Notes                                                               |
| ----------------------------------- | ----------- | ------------------------------------------------------------------- |
| 🌐 Intranet web app *(active)*      | `cloud`     | OpenAI-fronted, behind PEA's reverse proxy                          |
| 🏢 Air-gapped intranet *(planned)*  | `local`     | Wire `LlamaCppLocalProvider`; port logic from [app/services/llm_service.py](app/services/llm_service.py) |
| 💬 LINE chatbot *(planned)*         | `cloud`     | Same backend, separate LINE-webhook adapter on the frontend         |

---

## Contact

**PEA SSD AI Development Team**

📧 Email: [chatbordin.kli@pea.co.th](mailto:chatbordin.kli@pea.co.th)

---

## License

This project is for internal use by the Provincial Electricity Authority (PEA) of Thailand.
