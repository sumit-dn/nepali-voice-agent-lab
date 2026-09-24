"""Thin wrapper: same as `voice-lab reports ...` (extra arguments are passed through)."""

import sys

from voice_lab.cli import main

sys.exit(main(["reports", *sys.argv[1:]]))
