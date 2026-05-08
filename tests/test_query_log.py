from __future__ import annotations

import os
import sqlite3

from app.core.config import get_settings
from app.infrastructure.persistence.query_log_repo import SqliteQueryLogRepository


def test_save_writes_row(cleanup_tmp_storage):
    get_settings.cache_clear()
    settings = get_settings()
    repo = SqliteQueryLogRepository(settings)
    repo.save("hello", session_id="s1", metadata={"k": "v"})

    conn = sqlite3.connect(settings.SQLITE_PATH)
    try:
        rows = conn.execute(
            "SELECT content, session_id FROM user_queries WHERE session_id = ?", ("s1",)
        ).fetchall()
    finally:
        conn.close()
    assert any(r[0] == "hello" and r[1] == "s1" for r in rows)
    assert os.path.exists(settings.SQLITE_PATH)
