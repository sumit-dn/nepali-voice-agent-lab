"""UI handlers over fake providers (no browser, no real models). Skipped if the `ui` extra isn't installed."""

import html
import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from voice_lab import ui  # noqa: E402
from voice_lab.cli import main  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_model_cache():
    ui.unload()
    yield
    ui.unload()


def test_app_builds_and_interactive_handlers_work(lab):
    assert isinstance(ui.build_app(), gr.Blocks)
    audio = str(lab / "data" / "a.wav")

    text, stats = ui.transcribe(audio, "fake-asr", "clean_8k", "auto", False)
    assert "९८४१२३४५६७" in text and "resample 8000->16000" in html.unescape(stats)  # shown, not hidden
    rows, _ = ui.detect(audio, "fake-vad", "auto", False)
    assert rows and rows[0][0] == 0.1
    wav, _ = ui.speak("नमस्ते", "fake-tts", "", None, "auto", False)
    assert Path(wav).exists() and str(lab / "results" / "ui") in wav

    first, *_, last = ui.chat("नमस्ते", None, "fake-llm", False, "sys", "auto", False)
    assert first[0][-1]["metadata"]["status"] == "pending"  # the user sees their message + a working indicator at once
    display, state, cleared, _ = last
    assert cleared == "" and display[-1] == {"role": "assistant", "content": "नमस्ते, हजुर।"}
    *_, (display, state, _, _) = ui.chat("फेरि", state, "fake-llm", False, "sys", "auto", False)
    assert len(state["turns"]) == 4 and len(display) == 4
    with pytest.raises(gr.Error, match="reference voice only works with XTTS"):
        ui.speak("नमस्ते", "fake-tts", "", audio, "auto", False)

    *partials, (heard, reply, reply_wav, details) = ui.converse_ui(
        audio, "fake-vad", "fake-asr", "fake-llm", "fake-tts", False, "auto", False
    )
    assert heard and reply and Path(reply_wav).exists() and "Total" in details
    assert any(h == heard and not r and "Generating reply" in d for h, r, _, d in partials)  # transcript before reply
    *_, (heard, _, _, details) = ui.converse_ui(audio, "", "fake-asr", "fake-llm", "fake-tts", False, "auto", False)
    assert heard and "VAD" not in details  # "" = None (skip VAD)
    with pytest.raises(gr.Error, match="Choose a LLM model"):  # errors in the worker thread reach the UI
        list(ui.converse_ui(audio, "", "fake-asr", "", "fake-tts", False, "auto", False))


def test_errors_are_shown_not_raised_raw(lab):
    audio = str(lab / "data" / "a.wav")
    with pytest.raises(gr.Error, match="Nothing was sent"):  # cloud guard still applies in the UI
        ui.transcribe(audio, "fake-cloud-asr", "original", "auto", False)
    with pytest.raises(gr.Error, match="Record or upload"):
        ui.transcribe(None, "fake-asr", "original", "auto", False)
    import numpy as np
    import soundfile as sf

    sf.write(lab / "silent.wav", np.zeros(16000), 16000)  # what an unplugged/muted mic records
    with pytest.raises(gr.Error, match="silent"):
        ui.transcribe(str(lab / "silent.wav"), "fake-asr", "original", "auto", False)
    with pytest.raises(gr.Error, match="confirm"):
        next(ui.do_download("fake-asr", False))


def test_benchmark_kind_leaves_judge_unset(lab):
    *_, judge = ui.on_benchmark_kind("tts")
    assert judge["visible"] is True and judge["value"] is None  # not silently the first ASR model


def test_unload_frees_ollama_models(lab, monkeypatch):
    from voice_lab.config import ModelSpec
    from voice_lab.llm.openai_compat import OpenAICompatLLM
    from voice_lab.providers.base import ProviderError
    from voice_lab.utils import http

    calls = []

    def fake_open(url, **kwargs):
        calls.append((url, kwargs["body"]))
        raise ProviderError("server down")  # ignored: nothing is loaded there then

    monkeypatch.setattr(http, "open_url", fake_open)
    params = {"default_base_url": "http://ollama:11434", "server": "ollama"}
    ui._providers["k"] = OpenAICompatLLM(ModelSpec(id="o", provider="openai-compat", checkpoint="qwen3", params=params))
    assert "No models" in ui.unload()
    assert calls == [("http://ollama:11434/api/generate", {"model": "qwen3", "keep_alive": 0})]


def test_reports_from_picked_runs(lab):
    from voice_lab.benchmarking.report import generate

    assert main(["benchmark", "asr"]) == 0 and main(["benchmark", "asr", "--force"]) == 0
    (label, newest), (_, older) = ui.run_choices("asr")
    assert label.endswith("1 model · 2 samples") and older < newest  # newest first
    generate(runs={"asr": lab / "results" / "asr" / older})
    text = (lab / "reports" / "asr-comparison.md").read_text()
    assert older in text and newest not in text
    _, view = ui.make_reports(None, None, None)
    assert "No results yet" in (lab / "reports" / "asr-comparison.md").read_text() and view


def test_human_rating_flow(lab):
    assert main(["benchmark", "tts"]) == 0
    ((label, run),) = ui.tts_runs()
    assert label.endswith("1 system · 1 clip") and "fake-tts" not in label  # blind: no model names
    with pytest.raises(gr.Error, match="rater id"):
        ui.rate_start(run, "Ram Bahadur")  # real names are rejected
    state, clip, text, progress, *rest = ui.rate_start(run, "R01")
    assert Path(clip).exists() and "नमस्ते" in text and "wrap around" in progress
    with pytest.raises(gr.Error, match="Rate at least one criterion"):
        ui.rate_save(state, *[None] * 7, " ")
    saved = ui.rate_save(state, "4", "5", "4", "3", "4", "5", None, "ok")
    assert len(saved) == 4 + len(rest) and not any("fake-tts" in str(v) for v in saved[1:])  # results stay hidden
    summary = json.loads((lab / "results" / "tts" / run / "human_eval" / "summary.json").read_text())
    assert summary["raters"] == 1 and summary["results"]["fake-tts [default]"]["pronunciation"]["mean"] == 5
    assert "mixed_language" not in summary["results"]["fake-tts [default]"]  # left empty = not applicable


def test_intended_text_falls_back_to_the_manifest(lab):
    from voice_lab.evaluation.human import sample_texts

    assert main(["benchmark", "tts"]) == 0
    run = next((lab / "results" / "tts").glob("*T*Z-*"))
    record_path = next(run.glob("*.json"))
    record = json.loads(record_path.read_text())
    for s in record["samples"]:
        del s["metrics"]["text"]  # runs from before metrics.text existed
    record["dataset"] = "data/manifests/tts.jsonl"  # relative to the project root
    record_path.write_text(json.dumps(record, ensure_ascii=False))
    assert sample_texts(run) == {"t1": "नमस्ते"}
    _, _, text, *_ = ui.rate_start(run.name, "R01")
    assert "नमस्ते" in text
