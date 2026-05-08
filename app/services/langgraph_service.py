# This module has been removed.
#
# The active RAG pipeline now lives in app/application/qa_service.py
# using the hexagonal architecture (ports + adapters).
#
# OCR ingestion is handled by:
#   - app/infrastructure/ocr/ollama_client.py (adapter)
#   - app/ingestion/build_index.py (CLI)
#   - scripts/run_local_ocr.py (standalone convenience script)
#
# Please delete this file from version control.
raise ImportError(
    "langgraph_service has been removed. "
    "Use app.application.qa_service.QAService instead."
)
