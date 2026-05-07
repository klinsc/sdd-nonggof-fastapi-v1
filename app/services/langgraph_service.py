"""
LangGraph RAG Service — น้องกอฟ (PEA Substation AI Assistant)
=============================================================

Online serving layer that loads pre-processed JSON documents (produced by
the offline OCR ingestion script), chunks them, embeds them into ChromaDB,
and exposes a LangGraph agent for question-answering.

NOTE: This module no longer performs any OCR or calls any cloud API.
      All document processing is done offline via `scripts/run_local_ocr.py`.
"""

import getpass
import json
import os
import sqlite3
from datetime import datetime
from enum import Enum

import torch
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class DatasetName(str, Enum):
    STANDARDS = "standards"
    SDD_DATA = "sdd-data"


# Active dataset — change this to switch between document collections
dataset_name: DatasetName = DatasetName.SDD_DATA

# Directories for pre-processed JSON files (produced by scripts/run_local_ocr.py)
JSON_DIRS = {
    DatasetName.SDD_DATA: "data/sdd-data_json",
    DatasetName.STANDARDS: "data/standards_json",
}

CHROMA_PERSIST_DIR = "storage/chroma_data"


# ---------------------------------------------------------------------------
# SQLite — user query logging
# ---------------------------------------------------------------------------

DB_DIR = os.path.join("storage")
DB_PATH = os.path.join(DB_DIR, "app.sqlite3")
os.makedirs(DB_DIR, exist_ok=True)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_id TEXT,
                content TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        conn.commit()


def _coerce_message_content_to_text(content) -> str:
    """Handles LangChain message.content being str or list (for multimodal)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def save_user_query(
    content: str, session_id: str | None = None, metadata: dict | None = None
):
    meta_str = None
    if metadata is not None:
        try:
            meta_str = json.dumps(metadata, ensure_ascii=False)
        except Exception:
            meta_str = str(metadata)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO user_queries (created_at, session_id, content, metadata) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), session_id, content, meta_str),
        )
        conn.commit()


# Initialize DB on import
init_db()


# ---------------------------------------------------------------------------
# Document loading — reads pre-processed JSON files (no OCR, no cloud API)
# ---------------------------------------------------------------------------

def _clean_document(doc: Document) -> Document:
    """Clean up the page_content of a Document object."""
    cleaned = doc.page_content.replace("\\n", "\n")
    return Document(page_content=cleaned, metadata=doc.metadata)


def load_documents_from_json(json_dir: str | None = None) -> list[Document]:
    """
    Load pre-processed JSON files from the given directory and return
    a list of LangChain Document objects.

    Each JSON file is expected to have a ``natural_text`` key (produced by
    ``scripts/run_local_ocr.py``) or a ``content_markdown`` key.
    """
    if json_dir is None:
        json_dir = JSON_DIRS.get(dataset_name, JSON_DIRS[DatasetName.SDD_DATA])

    if not os.path.isdir(json_dir):
        print(f"Warning: JSON directory does not exist: {json_dir}")
        return []

    docs: list[Document] = []
    problematic_files: list[str] = []

    for fname in sorted(os.listdir(json_dir)):
        if not fname.lower().endswith(".json"):
            continue

        fpath = os.path.join(json_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Support both output formats:
            #   1. Offline script → "natural_text" (combined markdown of all pages)
            #   2. Legacy / per-page → "content_markdown"
            text = json_data.get("natural_text") or json_data.get("content_markdown")

            if not text:
                raise ValueError(
                    f"Neither 'natural_text' nor 'content_markdown' found in {fpath}"
                )

            doc = Document(
                page_content=text,
                metadata={
                    "source": json_data.get("original_file_path", fpath),
                    "json_source": fpath,
                },
            )
            docs.append(doc)
            print(f"  ✅ Loaded: {fname}")

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error in {fname}: {e}")
            problematic_files.append(fpath)
        except ValueError as e:
            print(f"  ❌ Missing content in {fname}: {e}")
            problematic_files.append(fpath)
        except Exception as e:
            print(f"  ❌ Failed to load {fname}: {e}")
            problematic_files.append(fpath)

    # Clean documents
    docs = [_clean_document(doc) for doc in docs]

    print(f"\nTotal documents loaded: {len(docs)}")
    if problematic_files:
        print(f"Problematic files ({len(problematic_files)}):")
        for pf in problematic_files:
            print(f"  - {pf}")

    return docs


# ---------------------------------------------------------------------------
# Document chunking + embedding
# ---------------------------------------------------------------------------

print("Loading pre-processed documents from JSON ...")
docs = load_documents_from_json()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=3000, chunk_overlap=1000
)
texts = text_splitter.split_documents(docs)
print(f"Total text chunks: {len(texts)}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
)


def embed_text(texts: list[Document]) -> Chroma | None:
    """Create a new ChromaDB vector store from document chunks."""
    try:
        vectorstore = Chroma.from_documents(
            documents=texts,
            collection_name=dataset_name.value,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        return vectorstore
    except Exception as e:
        print(f"Error during embedding or vector store creation: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


# ---------------------------------------------------------------------------
# LangGraph — RAG agent
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""

    # Check if the vector store already exists
    if os.path.exists(CHROMA_PERSIST_DIR):
        vector_store = Chroma(
            collection_name=dataset_name.value,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    else:
        vector_store = embed_text(texts)

    if vector_store is None:
        raise ValueError("Vector store could not be created or loaded.")

    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


graph_builder = StateGraph(MessagesState)


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    # Persist the latest human query (if any)
    try:
        latest_human = next(
            (m for m in reversed(state["messages"]) if m.type == "human"), None
        )
        if latest_human is not None:
            content_text = _coerce_message_content_to_text(
                getattr(latest_human, "content", "")
            )
            # session_id is optional; pass it via state when invoking the graph if you have one
            session_id = state.get("session_id") if isinstance(state, dict) else None
            save_user_query(content_text, session_id=session_id, metadata=None)
    except Exception as e:
        # Non-fatal: log and continue
        print(f"Warning: failed to save user query to SQLite: {e}")

    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use ten sentences maximum."
        "Respond **only in Thai language**."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_graph():
    from langgraph.graph import END
    from langgraph.prebuilt import tools_condition

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()
