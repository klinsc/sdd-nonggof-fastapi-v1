"""Offline ingestion pipeline.

Run with:  python -m app.ingestion.build_index --source data/sdd-data \
                                               --ocr-out data/sdd-data_json

Splits OCR + chunking + embedding from the API serving process. The API
process only ever *reads* the resulting Chroma index.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.embeddings.hf_bge_m3 import HuggingFaceEmbeddingsProvider
from app.infrastructure.ocr.typhoon_client import TyphoonOCRClient
from app.infrastructure.vectorstore.chroma_repo import ChromaVectorStoreRepository

logger = logging.getLogger(__name__)

DEFAULT_PROBLEMATIC_FILES: tuple[str, ...] = (
    "สถานีไฟฟ้าลำลูกกา 3 (คพจ.2).pdf",
    "สฟ.ปากท่อ 2 (อนุมัติ + แบบ).pdf",
    "อนุมัติ อุบล 5.pdf",
    "สำเนาอนุมัติแบบและค่าใช้จ่ายสฟ.กันทรวิชัย.pdf",
)


def _clean(doc: Document) -> Document:
    return Document(page_content=doc.page_content.replace("\\n", "\n"), metadata=doc.metadata)


def ocr_corpus(
    source_dir: str,
    ocr_output_dir: str,
    ocr: TyphoonOCRClient,
    skip: Iterable[str] = DEFAULT_PROBLEMATIC_FILES,
) -> list[Document]:
    os.makedirs(ocr_output_dir, exist_ok=True)
    skip_set = set(skip)
    docs: list[Document] = []

    for root, _dirs, files in os.walk(source_dir):
        for filename in files:
            if not filename.endswith(".pdf"):
                continue
            if filename in skip_set:
                logger.info("Skipping problematic file: %s", filename)
                continue

            original_path = os.path.join(root, filename)
            ocr_path = os.path.join(
                ocr_output_dir, os.path.splitext(filename)[0] + "_ocr.json"
            )

            try:
                if not os.path.exists(ocr_path):
                    logger.info("OCR -> %s", ocr_path)
                    ocr.extract_to_json(original_path, ocr_path)
                else:
                    logger.info("OCR cache hit: %s", ocr_path)
            except Exception as exc:
                logger.exception("OCR failed for %s: %s", original_path, exc)
                continue

            try:
                with open(ocr_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                natural_text = data.get("natural_text")
                if not natural_text:
                    logger.warning("Empty natural_text in %s; skipping.", ocr_path)
                    continue
                docs.append(
                    Document(
                        page_content=natural_text,
                        metadata={"source": original_path, "ocr_json_source": ocr_path},
                    )
                )
            except Exception as exc:
                logger.exception("Failed to load OCR JSON %s: %s", ocr_path, exc)

    logger.info("Loaded %d documents from %s", len(docs), source_dir)
    return [_clean(d) for d in docs]


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)

    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--source", default="data/sdd-data", help="PDF source directory")
    parser.add_argument(
        "--ocr-out", default="data/sdd-data_json", help="OCR JSON output directory"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild even if a Chroma index already exists"
    )
    args = parser.parse_args(argv)

    embeddings = HuggingFaceEmbeddingsProvider(settings).build()
    store_repo = ChromaVectorStoreRepository(settings, embeddings)

    if store_repo.exists() and not args.rebuild:
        logger.info(
            "Chroma index already exists at %s. Use --rebuild to re-index.",
            settings.CHROMA_DIR,
        )
        return 0

    ocr = TyphoonOCRClient(settings.TYHOON_API_KEY)
    docs = ocr_corpus(args.source, args.ocr_out, ocr)
    if not docs:
        logger.error("No documents loaded; aborting index build.")
        return 1

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info("Indexing %d chunks", len(chunks))
    store_repo.build(chunks)
    logger.info("Index build complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
