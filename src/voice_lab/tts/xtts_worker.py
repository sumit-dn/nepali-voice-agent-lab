"""XTTS worker. Runs inside .venv-xtts (coqui-tts 0.27 needs transformers<5 and, to avoid torchcodec/FFmpeg,
torch<2.9), so it must not import voice_lab. The lab talks to it over JSON lines (see tts/xtts.py):

  {"cmd": "load", "dir": ".../epoch-20", "device": "cpu"}  -> {"ok": true, "sample_rate": 24000, "speakers": [...]}
  {"cmd": "synth", "text": "...", "voice": "<built-in speaker | reference .wav path>", "language": "hi",
   "out": "/tmp/x.wav", "options": {"temperature": 0.65}}  -> {"ok": true, "sentences": 2}
"""

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

SENTENCE_END = re.compile(r"(?<=[।॥?!.])\s+")


def split_sentences(text: str) -> list[str]:
    """The model card advises one sentence per call so the autoregressive GPT does not drift into noise."""
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def load(folder: str, device: str) -> tuple[Any, int]:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    d = Path(folder)
    cfg = XttsConfig()
    cfg.load_json(str(d / "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(
        cfg,
        checkpoint_path=str(d / "model.pth"),
        vocab_path=str(d / "vocab.json"),
        speaker_file_path=str(d / "speakers_xtts.pth"),
        eval=True,
    )
    if device.startswith("cuda"):
        model.cuda()
    return model, int(cfg.model_args.output_sample_rate)


def synth(model: Any, sample_rate: int, req: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    voice, speakers = req["voice"], model.speaker_manager.speakers
    if voice in speakers:
        latent, embedding = speakers[voice]["gpt_cond_latent"], speakers[voice]["speaker_embedding"]
    elif Path(voice).is_file():
        latent, embedding = model.get_conditioning_latents(audio_path=[voice])  # zero-shot voice cloning
    else:
        raise ValueError(
            f"Unknown voice {voice!r}: use a built-in speaker (e.g. {', '.join(list(speakers)[:4])}) "
            "or a path to a reference WAV"
        )
    gap = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
    pieces: list[Any] = []
    for sentence in split_sentences(req["text"]):
        wav = model.inference(sentence, req["language"], latent, embedding, **req.get("options", {}))["wav"]
        pieces += [np.asarray(wav, dtype=np.float32), gap]
    sf.write(
        req["out"], np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32), sample_rate, subtype="FLOAT"
    )
    return {"sentences": len(pieces) // 2}


def main() -> None:
    replies, sys.stdout = sys.stdout, sys.stderr  # coqui prints progress; keep the protocol channel clean
    model = sample_rate = None
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["cmd"] == "load":
                model, sample_rate = load(req["dir"], req.get("device", "cpu"))
                out = {"ok": True, "sample_rate": sample_rate, "speakers": list(model.speaker_manager.speakers)}
            elif model is None or sample_rate is None:
                raise RuntimeError("send a 'load' request first")
            else:
                out = {"ok": True, **synth(model, sample_rate, req)}
        except Exception as e:  # reported to the lab, which raises it as a ProviderError
            traceback.print_exc()
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        replies.write(json.dumps(out, ensure_ascii=False) + "\n")
        replies.flush()


if __name__ == "__main__":
    main()
