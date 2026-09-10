from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import fingerprint
from .numeric import require_finite


class DecompositionCache:
    def __init__(self, path: str):
        self.path = Path(path)

    @staticmethod
    def key(question, model_config, prompt, template_fingerprint=None):
        return fingerprint({"question": question.strip(), "llm": model_config.model_dump(), "prompt": prompt,
                            "template_fingerprint": template_fingerprint})

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS decomposition (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                yield conn
        finally:
            conn.close()

    def get(self, key):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM decomposition WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        require_finite(value)
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO decomposition VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False, allow_nan=False)))
