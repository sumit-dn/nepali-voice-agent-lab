"""Apply a preprocessing profile from configs/benchmark.yaml to audio files, for listening/inspection.

    python scripts/prepare_audio.py telephone_8k data/raw/spk01/*.wav

Writes data/processed/<profile>/<name>.wav. Benchmarks do NOT read these files; they re-apply the
profile themselves so every result records exactly which steps were used.
"""

import sys
from pathlib import Path

from voice_lab import config
from voice_lab.audio.io import save_wav
from voice_lab.audio.preprocess import load_and_prepare

if len(sys.argv) < 3:
    sys.exit(__doc__)
profile, files = sys.argv[1], sys.argv[2:]
steps = config.load_benchmark_config()["preprocess_profiles"][profile]
out_dir = config.ROOT / "data" / "processed" / profile
for name in files:
    audio = load_and_prepare(name, steps)
    print(save_wav(out_dir / (Path(name).stem + ".wav"), audio.samples, audio.sample_rate))
