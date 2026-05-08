from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

from app.core.config import Settings

logger = logging.getLogger(__name__)


class SqliteQueryLogRepository:
    """SQLite-backed append log for user queries.

    Synchronous on purpose: writes are dispatched off the request hot path
    via FastAPI's BackgroundTasks, so a brief `sqlite3.connect()` does not
    block the event loop on the response path.
    """

    def __init__(self, settings: Settings) -> None:
        self._path = settings.SQLITE_PATH
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        # sqlite3 context manager only commits — it does not close the
        # connection, so on Windows the file stays locked until GC.
        # Use try/finally to guarantee close.
        conn = sqlite3.connect(self._path)
        try:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_queries_session_created "
                "ON user_queries (session_id, created_at)"
            )
            conn.commit()
        finally:
            conn.close()

    def save(
        self,
        content: str,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        meta_str: str | None = None
        if metadata is not None:
            try:
                meta_str = json.dumps(metadata, ensure_ascii=False)
            except (TypeError, ValueError):
                meta_str = str(metadata)
        try:
            conn = sqlite3.connect(self._path)
            try:
                conn.execute(
                    "INSERT INTO user_queries (created_at, session_id, content, metadata) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        session_id,
                        content,
                        meta_str,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("failed to save user query: %s", exc)
