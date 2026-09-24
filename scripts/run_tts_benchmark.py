"""Thin wrapper: same as `voice-lab benchmark tts ...` (extra arguments are passed through)."""

import sys

from voice_lab.cli import main

sys.exit(main(["benchmark", "tts", *sys.argv[1:]]))
