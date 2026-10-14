"""Intonation Analyzer launcher: ``python main.py``.

Kept beside the package so a source checkout starts the way it always has;
it is the same start-up as ``python -m intonation``.
"""
import sys

from intonation.__main__ import main

if __name__ == '__main__':
    sys.exit(main())
