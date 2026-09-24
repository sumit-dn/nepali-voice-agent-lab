"""AI4Bharat Indic Parler-TTS (~0.9B, 44.1 kHz). GPU recommended.

parler-tts pins transformers==4.46.1, so it lives in its own venv (docs/tts.md) rather than the main env.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from voice_lab.providers.base import ProviderError, TTSProvider


class ParlerTTS(TTSProvider):
    def _load(self) -> None:
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except ImportError as e:
            raise ProviderError(
                "parler-tts is not installed in this env. It needs a separate venv: see docs/tts.md"
            ) from e
        from transformers import AutoTokenizer

        ckpt, rev = str(self.spec.checkpoint), self.spec.revision
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(ckpt, revision=rev).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, revision=rev)
            self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        except OSError as e:
            raise ProviderError(
                f"{ckpt} not available locally. It is gated: accept terms on https://huggingface.co/{ckpt}, "
                f"set HF_TOKEN, then: voice-lab download {self.spec.id}  (size: {self.spec.size})"
            ) from e

    def _synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int, dict[str, Any]]:
        description = self.spec.params["description"].format(voice=voice or self.spec.params.get("voice", ""))
        desc = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
        generation = self.model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        audio = generation.cpu().numpy().squeeze().astype(np.float32)
        return audio, int(self.model.config.sampling_rate), {"description": description}
