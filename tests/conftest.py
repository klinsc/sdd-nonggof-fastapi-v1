from __future__ import annotations

import os
from typing import Iterator

import pytest

# Ensure required env is in place before Settings is imported anywhere.
os.environ.setdefault("API_KEY", "test-api-key-1234567890")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ.setdefault("LLM_PROFILE", "cloud")
os.environ.setdefault("CHROMA_DIR", "tests/.tmp_chroma")
os.environ.setdefault("SQLITE_PATH", "tests/.tmp_app.sqlite3")


@pytest.fixture
def api_key() -> str:
    return os.environ["API_KEY"]


@pytest.fixture
def auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}


@pytest.fixture
def cleanup_tmp_storage() -> Iterator[None]:
    yield
    for path in ("tests/.tmp_app.sqlite3",):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
