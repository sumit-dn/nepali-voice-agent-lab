"""Thin wrapper: same as `voice-lab benchmark llm ...` (extra arguments are passed through)."""

import sys

from voice_lab.cli import main

sys.exit(main(["benchmark", "llm", *sys.argv[1:]]))
