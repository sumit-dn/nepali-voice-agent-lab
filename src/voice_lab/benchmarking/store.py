"""SQLite index of every benchmark run/sample (results/lab.db). Raw outputs live in per-run JSON files."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, kind TEXT, created TEXT, dataset TEXT, dataset_sha TEXT,
  run_dir TEXT, config TEXT, hardware TEXT);
CREATE TABLE IF NOT EXISTS models (
  run_id TEXT, model TEXT, variant TEXT, provider TEXT, checkpoint TEXT, revision TEXT,
  cache_key TEXT, license TEXT, load_seconds REAL, resources TEXT, summary TEXT, error TEXT,
  PRIMARY KEY (run_id, model, variant));
CREATE TABLE IF NOT EXISTS samples (
  run_id TEXT, model TEXT, variant TEXT, cache_key TEXT, dataset_sha TEXT, sample_id TEXT,
  category TEXT, latency REAL, metrics TEXT, output TEXT, error TEXT, reused_from TEXT, created TEXT,
  PRIMARY KEY (run_id, model, variant, sample_id));
CREATE INDEX IF NOT EXISTS samples_cache ON samples (cache_key, dataset_sha, sample_id);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def add_run(
        self,
        run_id: str,
        kind: str,
        dataset: str,
        dataset_sha: str,
        run_dir: Path,
        config: dict[str, Any],
        hardware: dict[str, Any],
    ) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    kind,
                    now(),
                    dataset,
                    dataset_sha,
                    str(run_dir),
                    json.dumps(config, ensure_ascii=False),
                    json.dumps(hardware),
                ),
            )

    def cached(self, cache_key: str, dataset_sha: str, sample_id: str) -> dict[str, Any] | None:
        """Latest successful result for identical model config + dataset + sample, from any run."""
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM samples WHERE cache_key=? AND dataset_sha=? AND sample_id=? AND error IS NULL "
                "ORDER BY created DESC LIMIT 1",
                (cache_key, dataset_sha, sample_id),
            ).fetchone()
        return dict(row) if row else None

    def add_sample(
        self, run_id: str, model: str, variant: str, cache_key: str, dataset_sha: str, sample: dict[str, Any]
    ) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    model,
                    variant,
                    cache_key,
                    dataset_sha,
                    sample["id"],
                    sample.get("category"),
                    sample.get("latency"),
                    json.dumps(sample.get("metrics"), ensure_ascii=False),
                    sample.get("output"),
                    sample.get("error"),
                    sample.get("reused_from"),
                    now(),
                ),
            )

    def add_model(self, run_id: str, record: dict[str, Any]) -> None:
        spec = record["model"]
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    spec["id"],
                    record["variant"],
                    spec["provider"],
                    spec["checkpoint"],
                    spec["revision"],
                    record["cache_key"],
                    json.dumps(spec["license"], ensure_ascii=False),
                    record["load_seconds"],
                    json.dumps(record["resources"]),
                    json.dumps(record["summary"], ensure_ascii=False),
                    record["error"],
                ),
            )
