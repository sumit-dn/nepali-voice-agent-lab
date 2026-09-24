UV  := env -u VIRTUAL_ENV uv
LAB := .venv/bin/voice-lab

.PHONY: install install-gpu test test-integration lint typecheck check info models smoke \
        bench-asr bench-llm bench-tts bench-telephone reports ui install-xtts

install:            ## CPU torch + local runtimes + web UI + dev tools
	$(UV) sync --python 3.11 --extra local --extra cpu --extra ui --extra dev

install-gpu:        ## CUDA 12.8 torch instead of CPU torch
	$(UV) sync --python 3.11 --extra local --extra gpu --extra ui --extra dev

install-xtts:       ## separate env for Coqui XTTS (coqui-tts needs transformers<5; torch<2.9 avoids torchcodec/FFmpeg)
	$(UV) venv .venv-xtts --python 3.11
	$(UV) pip install --python .venv-xtts/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*" "torchaudio==2.8.*"
	$(UV) pip install --python .venv-xtts/bin/python "coqui-tts==0.27.5" "transformers>=4.57,<5" soundfile

test:
	.venv/bin/pytest -q

test-integration:   ## needs: voice-lab download piper-ne-google-x-low
	.venv/bin/pytest -q -m integration

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

typecheck:
	.venv/bin/mypy

check: lint typecheck test

info:
	$(LAB) system-info

models:
	$(LAB) models

smoke:              ## tiny end-to-end run on SYNTHETIC audio (downloads the 28 MB Piper voice)
	$(LAB) download piper-ne-google-x-low -y
	$(LAB) dataset synth --tts piper-ne-google-x-low --limit 5
	$(LAB) benchmark asr --dataset data/manifests/asr_synthetic_piper-ne-google-x-low.jsonl --limit 5
	$(LAB) benchmark tts --model piper-ne-google-x-low --limit 5
	$(LAB) reports

bench-asr:
	$(LAB) benchmark asr

bench-telephone:
	$(LAB) benchmark telephone

bench-llm:
	$(LAB) benchmark llm

bench-tts:
	$(LAB) benchmark tts

reports:
	$(LAB) reports

ui:                 ## web UI on http://127.0.0.1:7860
	$(LAB) ui --open
