# This module has been replaced by ollama_chat.py.
#
# Local LLM inference is now handled via Ollama:
#   app/infrastructure/llm/ollama_chat.py
#
# Please delete this file from version control.
raise ImportError(
    "LlamaCppLocalProvider has been removed. "
    "Use OllamaChatProvider (app.infrastructure.llm.ollama_chat) instead. "
    "Set LLM_PROFILE=local in .env to use Ollama for inference."
)
