"""Thin wrapper: same as `voice-lab download ...` (extra arguments are passed through)."""

import sys

from voice_lab.cli import main

sys.exit(main(["download", *sys.argv[1:]]))
