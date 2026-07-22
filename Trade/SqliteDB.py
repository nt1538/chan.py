import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Any


class TradeDatabase:
    """Optional local persistence for simulated fills; no broker dependency."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS fills (id INTEGER PRIMARY KEY, run_id TEXT, payload TEXT NOT NULL)")

    def _connect(self):
        return sqlite3.connect(self.path)

    def save_fills(self, run_id: str, fills: Iterable[Mapping[str, Any]]) -> None:
        rows = [(run_id, json.dumps(dict(fill), default=str)) for fill in fills]
        with self._connect() as conn:
            conn.executemany("INSERT INTO fills(run_id, payload) VALUES (?, ?)", rows)
