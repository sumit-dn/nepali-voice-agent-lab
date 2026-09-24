"""AI4Bharat IndicConformer 600M multilingual (hybrid CTC/RNNT, ONNX via trust_remote_code).

Gated on Hugging Face (auto-approve): accept the terms, set HF_TOKEN, then `voice-lab download`.
The revision is pinned in configs/asr.yaml because remote code executes locally.

The repo's own `IndicASRModel.from_pretrained` ignores `revision` (it snapshot-downloads `main`), which would
silently bypass the pin. So we load the pinned code and the pinned snapshot explicitly instead.
"""

from __future__ import annotations

from typing import Any

from voice_lab.audio.io import Audio
from voice_lab.providers.base import ASRProvider, ProviderError


class IndicConformerASR(ASRProvider):
    def _load(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        ckpt, rev = str(self.spec.checkpoint), self.spec.revision
        try:
            folder = snapshot_download(ckpt, revision=rev)
            config_cls = get_class_from_dynamic_module("model_onnx.IndicASRConfig", ckpt, revision=rev)
            model_cls = get_class_from_dynamic_module("model_onnx.IndicASRModel", ckpt, revision=rev)
        except (OSError, ValueError) as e:
            raise ProviderError(
                f"{ckpt}@{rev} not available locally ({e.__class__.__name__}: {e}). It is gated: accept terms on "
                f"https://huggingface.co/{ckpt}, set HF_TOKEN, then: voice-lab download {self.spec.id}"
            ) from e
        self.model = model_cls(config_cls(ts_folder=folder))

    def _transcribe(self, audio: Audio) -> tuple[str, dict[str, Any]]:
        import torch

        p = self.spec.params
        language, decoding = p.get("language", "ne"), p.get("decoding", "ctc")
        wav = torch.from_numpy(audio.samples).unsqueeze(0)  # (1, samples), mono 16 kHz per model card
        return str(self.model(wav, language, decoding)), {"language": language, "decoding": decoding}
