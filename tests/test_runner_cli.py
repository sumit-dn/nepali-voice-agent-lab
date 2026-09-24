"""End-to-end over fake providers: benchmark -> JSON/CSV/SQLite -> cache reuse -> reports -> CLI."""

import json
import sqlite3

from voice_lab.cli import main


def run_dirs(lab, kind):
    return sorted((lab / "results" / kind).glob("*T*Z-*"))


def test_benchmark_asr_writes_results_and_reuses_cache(lab, capsys):
    assert main(["benchmark", "asr"]) == 0
    (run,) = run_dirs(lab, "asr")
    record = json.loads((run / "fake-asr__original.json").read_text())
    s = record["summary"]
    assert s["n"] == 2 and s["failed"] == 0
    assert s["entity_accuracy"]["phone_numbers"]["acc"] == 1.0
    assert s["wer"] > 0  # second sample is transcribed wrongly by the fake model
    assert record["model"]["license"]["id"] == "MIT" and record["hardware"]["cpu"]
    assert (run / "summary.csv").exists() and (run / "samples.csv").exists()
    # 8 kHz file was upsampled for the 16 kHz model and that conversion is recorded, not silent
    s2 = next(x for x in record["samples"] if x["id"] == "s2")
    assert s2["metrics"]["input_conversion"] == ["resample 8000->16000 Hz (model input rate)"]

    assert main(["benchmark", "asr"]) == 0
    second = json.loads((run_dirs(lab, "asr")[-1] / "fake-asr__original.json").read_text())
    assert all(x["reused_from"] for x in second["samples"])
    assert main(["benchmark", "asr", "--force"]) == 0
    forced = json.loads((run_dirs(lab, "asr")[-1] / "fake-asr__original.json").read_text())
    assert not any(x["reused_from"] for x in forced["samples"])

    db = sqlite3.connect(lab / "results" / "lab.db")
    assert db.execute("select count(*) from runs").fetchone()[0] == 3
    assert db.execute("select count(*) from samples").fetchone()[0] == 4  # cached rows not duplicated


def test_telephone_llm_tts_pipeline_and_reports(lab, capsys):
    assert main(["benchmark", "telephone"]) == 0
    (run,) = run_dirs(lab, "telephone")
    assert {p.name for p in run.glob("*.json")} == {"fake-asr__original.json", "fake-asr__clean_8k.json"}
    assert main(["benchmark", "llm"]) == 0
    llm = json.loads(next(run_dirs(lab, "llm")[-1].glob("*.json")).read_text())
    assert llm["summary"]["pass_rate"] == 1.0 and llm["summary"]["human_eval_pending"] == 1
    assert main(["benchmark", "tts", "--asr-judge", "fake-asr"]) == 0
    tts = json.loads(next(run_dirs(lab, "tts")[-1].glob("*.json")).read_text())
    assert tts["summary"]["sample_rate"] == 16000 and "asr_judge_cer" in tts["summary"]
    assert main(["human-eval", "sheet"]) == 0
    sheet = next((lab / "results" / "tts").glob("*/human_eval/ratings_template.csv"))
    filled = sheet.read_text().splitlines()
    filled[1] = filled[1].replace(",,,,,,,", ",4,5,4,3,4,5,")  # clip,sample_id,text,<7 criteria>,comment
    (lab / "filled.csv").write_text("\n".join(filled))
    assert main(["human-eval", "import", "--ratings", str(lab / "filled.csv"), "--rater", "R01"]) == 0
    summary = json.loads((sheet.parent / "summary.json").read_text())
    assert summary["raters"] == 1 and summary["results"]["fake-tts [default]"]["pronunciation"]["mean"] == 5
    (lab / "bad.csv").write_text("\n".join([filled[0], filled[1].replace(",4,5,", ",9,5,", 1)]))
    assert main(["human-eval", "import", "--ratings", str(lab / "bad.csv"), "--rater", "R02"]) == 2
    assert main(["conversation", "--audio", str(lab / "data/a.wav"), "--out", str(lab / "conv")]) == 0
    result = json.loads((lab / "conv" / "result.json").read_text())
    assert result["transcript"] and result["response"] and (lab / "conv" / "response.wav").exists()
    assert set(result["timings"]) >= {"vad", "asr", "llm", "llm_ttft", "tts", "total"}
    assert main(["benchmark", "pipeline", "--limit", "1"]) == 0
    assert main(["reports"]) == 0
    reports = {p.name for p in (lab / "reports").glob("*.md")}
    assert {
        "asr-comparison.md",
        "llm-comparison.md",
        "tts-comparison.md",
        "licensing.md",
        "critical-entity-accuracy.md",
        "hardware-profile.md",
        "asr-error-analysis.md",
    } <= reports
    text = (lab / "reports" / "telephone-comparison.md").read_text()
    assert "fake-asr" in text and "clean_8k" in text and "does not rank models" in text
    assert "No results yet" in (lab / "reports" / "asr-comparison.md").read_text()  # only telephone ran here


def test_cli_misc_commands(lab, capsys):
    assert main(["models", "--json"]) == 0
    assert "fake-asr" in capsys.readouterr().out
    assert main(["system-info", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "GEMINI_API_KEY" in info["optional_credentials"]
    assert main(["dataset", "validate", str(lab / "data/manifests/asr.jsonl"), "--kind", "asr"]) == 0
    assert main(["asr", "transcribe", str(lab / "data/a.wav"), "--model", "fake-asr", "--profile", "clean_8k"]) == 0
    assert main(["llm", "chat", "--model", "fake-llm", "--text", "नमस्ते"]) == 0
    assert main(["vad", "detect", str(lab / "data/a.wav"), "--model", "fake-vad"]) == 0
    assert main(["tts", "synthesize", "--model", "fake-tts", "--text", "नमस्ते", "--out", str(lab / "x.wav")]) == 0
    assert (lab / "x.wav").exists() and (lab / "x.orig.wav").exists()
    # errors are clean exit codes, not tracebacks
    assert main(["asr", "transcribe", str(lab / "data/a.wav"), "--model", "fake-cloud-asr"]) == 2
    assert "Nothing was sent" in capsys.readouterr().err
    assert main(["asr", "transcribe", str(lab / "data/a.wav"), "--model", "nope"]) == 2
    assert main(["benchmark", "asr", "--dataset", str(lab / "missing.jsonl")]) == 2
