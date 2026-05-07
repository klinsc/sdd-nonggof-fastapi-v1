# 🤖 น้องกอฟ — PEA Substation AI Assistant (Backend)

> AI-powered Chatbot สำหรับสนับสนุนงานมาตรฐานและการออกแบบสถานีไฟฟ้าแรงสูง 115kV ของ กฟภ.
> พัฒนาด้วยเทคโนโลยี **OCR + RAG + LLM** เพื่อช่วยค้นหา อ้างอิง และสรุปข้อมูลจากเอกสารมาตรฐานได้อย่างรวดเร็วและแม่นยำ

---

## 📑 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Core Services](#core-services)
- [Data Pipeline](#data-pipeline)
- [Deployment Targets](#deployment-targets)
- [Key Features](#key-features)
- [Benefits](#benefits)
- [Contact](#contact)

---

## Overview

**น้องกอฟ** is the backend service for PEA's Substation Design & Standards AI Assistant. It provides an intelligent question-answering system that retrieves relevant information from PEA's 115kV substation standard documents and generates accurate, Thai-language responses using a RAG (Retrieval-Augmented Generation) pipeline.

This repository contains the **FastAPI backend** server, which exposes streaming chat endpoints powered by:

- **Ollama (Local Vision Model)** — for offline OCR extraction from PDF documents (`scripts/run_local_ocr.py`)
- **LangChain + LangGraph** — for orchestrating the RAG pipeline
- **ChromaDB** — as the vector database for semantic search
- **GPT-4o-mini** — as the LLM for response generation
- **BAAI/bge-m3** — as the embedding model for document vectorization

> **Decoupled Architecture:** Document ingestion (OCR) runs **offline** via a standalone script.
> The FastAPI server only loads pre-processed JSON files — no cloud OCR API calls at runtime.

---

## Objectives

- ⏱️ **ลดเวลาในการค้นหาเอกสารมาตรฐาน** — Reduce time spent searching standard documents
- ✅ **เพิ่มความถูกต้องในการอ้างอิงข้อมูล** — Improve accuracy in document referencing
- 👷 **สนับสนุนบุคลากรด้านวิศวกรรมสถานีไฟฟ้า** — Support substation engineering personnel
- 📚 **สร้าง Knowledge Base กลางสำหรับองค์กร** — Build a centralized organizational Knowledge Base

---

## System Architecture

```
 ╔══════════════════════════════════════════════════════════════════╗
 ║              OFFLINE — Document Ingestion Pipeline              ║
 ║                  (scripts/run_local_ocr.py)                     ║
 ║                                                                 ║
 ║  PDF Files ──▶ PyMuPDF (render) ──▶ Ollama Vision Model ──▶ JSON║
 ║  (data/sdd-data/)                           (data/sdd-data_json/)║
 ╚══════════════════════════════════════════════════════════════════╝
                              │
                    Pre-processed JSON files
                              │
                              ▼
 ╔══════════════════════════════════════════════════════════════════╗
 ║              ONLINE — FastAPI Server (this repo)                ║
 ║                                                                 ║
 ║  ┌─────────────┐    ┌────────────────────────────────────────┐  ║
 ║  │  /qa/stream  │───▶│        LangGraph Agent Pipeline        │  ║
 ║  │   (Router)   │    │                                        │  ║
 ║  └─────────────┘    │  query_or_respond → retrieve → generate│  ║
 ║                      └────────────────────────────────────────┘  ║
 ║                                                                 ║
 ║  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   ║
 ║  │   ChromaDB   │    │   SQLite DB   │    │  JSON Loader     │   ║
 ║  │ (Vector Store)│    │ (Query Log)  │    │ (No cloud API)   │   ║
 ║  └──────────────┘    └──────────────┘    └──────────────────┘   ║
 ╚══════════════════════════════════════════════════════════════════╝
                              ▲
                   SSE (Server-Sent Events)
                              │
 ┌──────────────────────────────────────────────────────────────────┐
 │                        Client Layer                              │
 │              Web UI  /  Line Chatbot  /  API Client              │
 └──────────────────────────────────────────────────────────────────┘
```

### Pipeline Workflow

```
[OFFLINE]  PDF Documents
             → PyMuPDF (render pages to base64 PNG)
             → Ollama Vision Model (extract text + tables as Markdown)
             → JSON files saved to data/sdd-data_json/

[ONLINE]   JSON files loaded at server startup
             → Text Chunking (RecursiveCharacterTextSplitter, 3000 tokens, 1000 overlap)
             → Embedding (BAAI/bge-m3 via HuggingFace)
             → ChromaDB Vector Store
             → User Query (via /qa/stream SSE endpoint)
             → LangGraph Agent (query_or_respond → retrieve → generate)
             → Streamed AI Response (Thai language)
```

---

## Tech Stack

| Category          | Technology                                              |
| ----------------- | ------------------------------------------------------- |
| **Framework**     | FastAPI 0.115 + Uvicorn                                 |
| **LLM**          | GPT-4o-mini (via OpenAI API)                            |
| **OCR Engine**   | Ollama + Vision Model (local, offline via `scripts/run_local_ocr.py`) |
| **PDF Renderer** | PyMuPDF (fitz) — renders PDF pages to PNG for OCR      |
| **Embeddings**   | BAAI/bge-m3 (HuggingFace, CUDA-accelerated)            |
| **RAG Pipeline** | LangChain 0.3 + LangGraph                              |
| **Vector DB**    | ChromaDB 1.0 (with persistent local storage)            |
| **Database**     | SQLite (for query logging)                              |
| **GPU Support**  | PyTorch 2.7 + CUDA 12.8                                |
| **Language**     | Python 3.11                                             |

---

## Project Structure

```
sdd-nonggof-fastapi-v1/
├── scripts/                      # Offline tooling
│   └── run_local_ocr.py          # 🔑 Offline OCR ingestion (Ollama + PyMuPDF)
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point, CORS config, router registration
│   ├── config.py                 # Environment variable configuration (VarSettings)
│   ├── dependencies.py           # FastAPI dependency injection (auth tokens)
│   ├── routers/                  # API route handlers
│   │   ├── __init__.py
│   │   ├── chat.py               # /qa/stream — SSE streaming chat endpoint
│   │   ├── items.py              # (placeholder) Item CRUD routes
│   │   └── users.py              # (placeholder) User routes
│   ├── services/                 # Core business logic
│   │   ├── __init__.py
│   │   ├── langgraph_service.py  # 🔑 Online RAG pipeline (JSON → Embed → Retrieve → Generate)
│   │   └── llm_service.py        # (deprecated) LlamaIndex-based pipeline
│   └── internal/                 # Internal admin modules
│       ├── __init__.py
│       └── admin.py              # Admin endpoint (placeholder)
├── data/                         # Document storage (gitignored)
│   ├── manuals/                  # Manual documents
│   ├── sdd-data/                 # Source PDF files (input for offline OCR)
│   ├── sdd-data_json/            # Pre-processed JSON files (output of offline OCR)
│   ├── standards/                # Standard documents
│   └── standards_json/           # Pre-processed JSON for standards
├── models/                       # LLM model files (gitignored)
├── storage/                      # Persistent storage (gitignored)
│   ├── app.sqlite3               # SQLite database for user query logging
│   └── chroma_data/              # ChromaDB vector store data
├── .env                          # Environment variables (gitignored)
├── .gitignore
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Prerequisites

- **Python 3.11** (recommended via Conda)
- **NVIDIA GPU** with CUDA support (recommended for embedding inference)
- **CUDA Toolkit 12.8+** and **cuDNN**
- **Ollama** installed and running locally (for offline OCR ingestion)
- **Git**

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n llama-gpu python=3.11 -y
conda activate llama-gpu
```

### 2. Install System Dependencies

```bash
conda install -c conda-forge cmake ninja git -y
conda install -c nvidia cuda-toolkit -y
conda install -c nvidia cudnn -y
```

### 3. Install llama-cpp-python with CUDA Support

```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
set LLAMA_CUBLAS=1
pip install . --verbose --force-reinstall --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu129
cd ..
```

### 4. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install Triton (Windows)

```bash
pip install -U "triton-windows<3.4"
```

### 6. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install all at once:

```bash
pip install "fastapi[standard]" uvicorn python-dotenv openai typhoon-ocr pypdf \
  langchain-core langchain sentence-transformers torch huggingface_hub \
  transformers bitsandbytes accelerate langchain-community chromadb \
  langchain-openai langchain_huggingface langgraph langchain_chroma
```

---

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Debug mode
DEBUG=true

# HuggingFace token (for model downloads)
HF_TOKEN="hf_your_token_here"

# OpenAI API key (for GPT-4o-mini)
OPENAI_API_KEY="sk-proj-your-openai-key-here"
```

### Required API Keys

| Key              | Purpose                                  | Provider                                          |
| ---------------- | ---------------------------------------- | ------------------------------------------------- |
| `HF_TOKEN`       | Download embedding models from HuggingFace | [huggingface.co](https://huggingface.co)          |
| `OPENAI_API_KEY` | GPT-4o-mini for response generation       | [platform.openai.com](https://platform.openai.com) |

> **Note:** Typhoon OCR API key (`TYHOON_API_KEY`) is no longer required. OCR is now handled offline via Ollama.

---

## Running the Server

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or using FastAPI CLI:

```bash
fastapi dev app/main.py
```

The server will start at: **http://localhost:8000**

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

> **Note**: All endpoints require a `token` query parameter (e.g., `?token=jessica`).

---

## API Reference

### Health Check

```http
GET /?token=jessica
```

**Response:**
```json
{
  "message": "Hello Bigger Applications!"
}
```

---

### Chat — Streaming QA

```http
GET /qa/stream?input_message={question}&token=jessica
```

**Parameters:**

| Parameter       | Type   | Required | Description                      |
| --------------- | ------ | -------- | -------------------------------- |
| `input_message` | string | ✅       | The question to ask the AI       |
| `token`         | string | ✅       | Authentication token (`jessica`) |

**Response:** `text/event-stream` (Server-Sent Events)

**Event Types:**

| Event             | Description                           | Data Schema                                              |
| ----------------- | ------------------------------------- | -------------------------------------------------------- |
| `message_chunk`   | Streamed AI response token            | `{ "type": "ai_chunk", "content": "...", "metadata": {} }` |
| `tool_message`    | Tool execution result (retrieval)     | `{ "type": "tool_message", "content": "...", "name": "retrieve", "metadata": {} }` |
| `stream_end`      | End of stream marker                  | `{}`                                                     |

**Example (cURL):**

```bash
curl -N "http://localhost:8000/qa/stream?input_message=มาตรฐานการออกแบบสถานีไฟฟ้า+115kV+คืออะไร&token=jessica"
```

**Example (JavaScript EventSource):**

```javascript
const eventSource = new EventSource(
  'http://localhost:8000/qa/stream?input_message=มาตรฐาน+สถานีไฟฟ้า&token=jessica'
);

eventSource.addEventListener('message_chunk', (event) => {
  const data = JSON.parse(event.data);
  console.log(data.content); // AI response tokens
});

eventSource.addEventListener('stream_end', () => {
  eventSource.close();
});
```

---

## Core Services

### `langgraph_service.py` — Online RAG Pipeline (Active)

The primary service powering the AI chatbot. Loads pre-processed JSON files and runs the RAG pipeline:

| Component              | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| **Document Loading**   | Reads pre-processed JSON files from `data/sdd-data_json/` (no cloud API)   |
| **Text Splitting**     | `RecursiveCharacterTextSplitter` with 3000-token chunks and 1000 overlap    |
| **Embedding**          | `BAAI/bge-m3` model (GPU-accelerated via HuggingFace)                      |
| **Vector Store**       | ChromaDB with persistent storage at `storage/chroma_data/`                  |
| **Retrieval**          | Similarity search returning top 2 relevant document chunks                  |
| **LLM Generation**     | GPT-4o-mini via OpenAI API, responses in Thai language                      |
| **Query Logging**      | All user queries saved to SQLite at `storage/app.sqlite3`                   |

### `scripts/run_local_ocr.py` — Offline OCR Ingestion

Standalone script that processes PDFs into structured JSON files using a local Ollama vision model:

```bash
# Process all PDFs with default model (qwen2.5-vl)
python scripts/run_local_ocr.py

# Use a specific model
python scripts/run_local_ocr.py --model llama3-typhoon-vision

# Re-process already-converted files
python scripts/run_local_ocr.py --force

# Process a different directory
python scripts/run_local_ocr.py --pdf-dir data/standards --json-dir data/standards_json
```

### `llm_service.py` — Local LLM Pipeline (Deprecated)

An alternative pipeline using **LlamaIndex** with a local GGUF model. Currently **commented out** and not in active use.

---

## Data Pipeline

### Decoupled Architecture

The system uses a **two-phase** approach:

#### Phase 1 — Offline Ingestion (`scripts/run_local_ocr.py`)

1. **Source PDFs** are placed in `data/sdd-data/` (or `data/standards/`)
2. **OCR Processing** (runs locally via Ollama — no cloud API):
   - Each PDF page is rendered to a base64 PNG image using PyMuPDF
   - The image is sent to a local Ollama vision model (e.g., `qwen2.5-vl`)
   - The model extracts text and tables as Markdown
   - Results are saved as JSON in `data/sdd-data_json/`

#### Phase 2 — Online Serving (FastAPI server)

3. **JSON Loading**: Pre-processed JSON files are loaded at server startup
4. **Chunking**: Text is split using TikToken-based chunking (3000 tokens, 1000 overlap)
5. **Embedding**: Chunks are embedded using `BAAI/bge-m3` model
6. **Storage**: Vectors are stored in ChromaDB at `storage/chroma_data/`
7. **On first query**: If no existing vector store is found, embeddings are generated automatically

### Known Problematic Files

Some PDF files are skipped during OCR processing due to formatting issues:

- `สถานีไฟฟ้าลำลูกกา 3 (คพจ.2).pdf`
- `สฟ.ปากท่อ 2 (อนุมัติ + แบบ).pdf`
- `อนุมัติ อุบล 5.pdf`
- `สำเนาอนุมัติแบบและค่าใช้จ่ายสฟ.กันทรวิชัย.pdf`

---

## CORS Configuration

The following origins are allowed:

| Origin                                    | Description          |
| ----------------------------------------- | -------------------- |
| `http://localhost:3000`                    | Local frontend dev   |
| `https://sdd.chatbordin.com`              | Production frontend  |
| `http://sdd.chatbordin.com`               | Production (HTTP)    |
| `https://ssd-web-beta.vercel.app`         | Beta frontend        |
| `https://sdd-nonggof-reverse.chatbordin.com` | Reverse proxy     |

---

## Deployment Targets

| Target                         | Status       |
| ------------------------------ | ------------ |
| 🌐 Intranet Web Application    | ✅ Active    |
| 💬 Line Chatbot                | 🔧 Planned  |
| 🏢 Internal Knowledge Assistant | 🔧 Planned  |

---

## Key Features

- 🔍 **Semantic Search** — Find relevant information from standard documents using vector similarity
- 🤖 **AI-powered QA** — Natural language answers in Thai, grounded in actual documents
- 📄 **Document-based Responses** — All answers are backed by retrieved document context
- 🌊 **Streaming Responses** — Real-time token streaming via Server-Sent Events (SSE)
- 📝 **Query Logging** — All user queries are persisted for analytics and improvement
- 🔄 **Automatic Embedding** — Vector store is built automatically on first run
- 🎯 **GPU Acceleration** — CUDA-powered embedding inference for fast processing

---

## Benefits

- ⏱️ ลดเวลาค้นหาเอกสาร — Reduce document search time
- 🛡️ ลด Human Error — Minimize human errors in referencing
- 📈 เพิ่ม Productivity — Increase engineering productivity
- 🔄 สนับสนุน Digital Transformation — Support organizational digital transformation
- 🏗️ พัฒนาเป็น Enterprise Knowledge Platform ได้ — Scalable to enterprise knowledge platform

---

## Contact

**PEA SSD AI Development Team**

📧 Email: [chatbordin.kli@pea.co.th](mailto:chatbordin.kli@pea.co.th)

---

## License

This project is for internal use by the Provincial Electricity Authority (PEA) of Thailand.
