"""Blind human rating of TTS output: anonymised clip sheet -> filled CSV -> per-model mean ± 95% CI.

Raters see only `clip_###.wav` and the input text; the clip->model key is written to a separate file.
Rater ids are free-form pseudonyms (e.g. R01); never store names. Not "MOS" unless the protocol in
docs/evaluation.md (enough raters, anchors, randomisation) was followed.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from voice_lab.benchmarking.store import Store, now

CRITERIA = ("naturalness", "pronunciation", "clarity", "prosody", "accent", "intelligibility", "mixed_language")
SCHEMA = """CREATE TABLE IF NOT EXISTS human_ratings (
  run_id TEXT, rater TEXT, clip TEXT, model TEXT, voice TEXT, sample_id TEXT, criterion TEXT, score INTEGER,
  created TEXT, PRIMARY KEY (run_id, rater, clip, criterion));"""


def make_sheet(run_dir: Path, seed: int = 0) -> Path:
    clips = []
    for record_path in sorted(run_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for s in record["samples"]:
            if s.get("metrics") and not s.get("error"):
                clips.append(
                    {
                        "model": record["model"]["id"],
                        "voice": record["variant"],
                        "sample_id": s["id"],
                        "audio": s["metrics"]["audio_path"],
                        "text": s["metrics"].get("text", ""),
                    }
                )
    if not clips:
        raise ValueError(f"No synthesized clips found in {run_dir}")
    random.Random(seed).shuffle(clips)
    out = run_dir / "human_eval"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    key = {}
    with (out / "ratings_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "sample_id", "text", *CRITERIA, "comment"])
        for i, clip in enumerate(clips, 1):
            name = f"clip_{i:03d}"
            shutil.copy(clip["audio"], out / "clips" / f"{name}.wav")
            key[name] = {k: clip[k] for k in ("model", "voice", "sample_id")}
            writer.writerow([name, clip["sample_id"], clip["text"], *[""] * len(CRITERIA), ""])
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")  # keep away from raters
    return out


def import_ratings(run_dir: Path, ratings_csv: Path, rater: str, db_path: Path) -> dict[str, Any]:
    key = json.loads((run_dir / "human_eval" / "key.json").read_text(encoding="utf-8"))
    rows = []
    with ratings_csv.open(encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            if row["clip"] not in key:
                raise ValueError(f"{ratings_csv}:{n}: unknown clip {row['clip']!r}")
            for criterion in CRITERIA:
                value = (row.get(criterion) or "").strip()
                if not value:
                    continue  # e.g. mixed_language on a Nepali-only clip
                if value not in {"1", "2", "3", "4", "5"}:
                    raise ValueError(f"{ratings_csv}:{n}: {criterion}={value!r} is not an integer 1-5")
                k = key[row["clip"]]
                rows.append(
                    (
                        run_dir.name,
                        rater,
                        row["clip"],
                        k["model"],
                        k["voice"],
                        k["sample_id"],
                        criterion,
                        int(value),
                        now(),
                    )
                )
    store = Store(db_path)
    with store.db:
        store.db.execute(SCHEMA)
        store.db.executemany("INSERT OR REPLACE INTO human_ratings VALUES (?,?,?,?,?,?,?,?,?)", rows)
    summary = summarize(store, run_dir.name)
    (run_dir / "human_eval" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize(store: Store, run_id: str) -> dict[str, Any]:
    scores: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    raters = set()
    for model, voice, criterion, score, rater in store.db.execute(
        "SELECT model, voice, criterion, score, rater FROM human_ratings WHERE run_id=?", (run_id,)
    ):
        scores[(model, voice, criterion)].append(score)
        raters.add(rater)
    out: dict[str, Any] = {"raters": len(raters), "results": {}}
    for (model, voice, criterion), values in sorted(scores.items()):
        ci = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
        out["results"].setdefault(f"{model} [{voice}]", {})[criterion] = {
            "mean": round(statistics.fmean(values), 2),
            "ci95": round(ci, 2) if ci is not None else None,
            "n": len(values),
        }
    return out
