"""Reusable scenario steps, written only against the harness API."""
from __future__ import annotations

from tests.harness import HeadlessApp, Tone

# (sounding note, cents off true pitch) played on an Eb alto. Chosen so each
# note's measured mean rounds to the same whole number on every platform.
KNOWN_TABLE = [("C4", 0), ("E4", -15), ("G4", 8), ("A4", 30), ("C5", -3)]


def play_known_table(app: HeadlessApp, instrument: str = "eb_alto",
                     seconds: float = 1.0, timbre: str = "sax") -> None:
    """Pick ``instrument`` and hold each note of KNOWN_TABLE for ``seconds``."""
    app.pick_instrument(instrument)
    for note, cents in KNOWN_TABLE:
        app.play(Tone(note=note, cents=cents, seconds=seconds, timbre=timbre))
