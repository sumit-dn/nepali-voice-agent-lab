"""UI handlers over fake providers (no browser, no real models). Skipped if the `ui` extra isn't installed."""

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
    assert "९८४१२३४५६७" in text and "resample 8000->16000" in stats  # conversion is shown, not hidden
    rows, _ = ui.detect(audio, "fake-vad", "auto", False)
    assert rows and rows[0][0] == 0.1
    wav, _ = ui.speak("नमस्ते", "fake-tts", "", None, "auto", False)
    assert Path(wav).exists() and str(lab / "results" / "ui") in wav

    display, state, cleared, _ = ui.chat("नमस्ते", None, "fake-llm", False, "sys", "auto", False)
    assert cleared == "" and display[-1] == {"role": "assistant", "content": "नमस्ते, हजुर।"}
    display, state, _, _ = ui.chat("फेरि", state, "fake-llm", False, "sys", "auto", False)
    assert len(state["turns"]) == 4 and len(display) == 4

    heard, reply, reply_wav, details = ui.converse_ui(
        audio, "fake-vad", "fake-asr", "fake-llm", "fake-tts", False, "auto", False
    )
    assert heard and reply and Path(reply_wav).exists() and "Total" in details


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


def test_human_rating_flow(lab):
    assert main(["benchmark", "tts"]) == 0
    (run,) = ui.tts_runs()
    with pytest.raises(gr.Error, match="rater id"):
        ui.rate_start(run, "Ram Bahadur")  # real names are rejected
    state, clip, text, progress, *rest = ui.rate_start(run, "R01")
    assert Path(clip).exists() and "नमस्ते" in text
    out = ui.rate_save(state, "4", "5", "4", "3", "4", "5", None, "ok")
    summary = out[-1]
    assert summary["raters"] == 1 and summary["results"]["fake-tts [default]"]["pronunciation"]["mean"] == 5
    assert "mixed_language" not in summary["results"]["fake-tts [default]"]  # left empty = not applicable
