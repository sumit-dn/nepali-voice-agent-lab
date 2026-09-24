"""Thin wrapper: same as `voice-lab benchmark asr ...` (extra arguments are passed through)."""

import sys

from voice_lab.cli import main

sys.exit(main(["benchmark", "asr", *sys.argv[1:]]))
