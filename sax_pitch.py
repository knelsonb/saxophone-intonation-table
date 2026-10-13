"""Pitch maths shared by the window, the table, the exports and the output
sources: the A4 reference, MIDI <-> frequency conversion, note names and the
cents readout format.

Pure Python with no Qt or numpy dependency, so any module can import it
without pulling in the GUI. Each of these used to be copied into several
modules; this is the one copy.
"""
from __future__ import annotations

import math

A4_DEFAULT = 440.0

# MIDI 21..108 (A0..C8): the full catalog, including piccolo, recorder, piano
# and banjo upper registers. The detection range gate in sax_audio_engine
# (AudioEngine._midi_min / _midi_max) must cover the same span.
SAX_MIDI = range(21, 109)

CHROMA = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F',
          'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def freq_to_midi(f: float, a4: float | None = None) -> float:
    """Fractional MIDI number of ``f`` Hz at concert pitch ``a4``
    (A4_DEFAULT when None). Raises ValueError for f <= 0."""
    ref = a4 if a4 is not None else A4_DEFAULT
    return 69.0 + 12.0 * math.log2(f / ref)


def cents_dev(f: float, a4: float | None = None) -> tuple[int, float]:
    """(nearest MIDI note, deviation from it in cents) for ``f`` Hz."""
    mf = freq_to_midi(f, a4)
    mr = round(mf)
    return mr, (mf - mr) * 100.0


def midi_to_freq(midi: float, a4: float = A4_DEFAULT) -> float:
    """Frequency (Hz) of a MIDI note at the given A4 reference: the inverse of
    freq_to_midi. A4 (MIDI 69) maps to ``a4`` exactly, so a non-440 A4 shifts
    every note correctly."""
    return float(a4) * (2.0 ** ((float(midi) - 69.0) / 12.0))


def midi_note_name(m: int) -> str:
    """Display name of a MIDI note; black keys carry both spellings
    ("A#/Bb3")."""
    return f"{CHROMA[m % 12]}{m // 12 - 1}"


# B is B natural, as midi_note_name prints it. H is the German name for the
# same note and is accepted as an alias.
_NOTE_PITCH_CLASSES = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11, 'H': 11,
}


def _pitch_class_offset(spelling: str) -> int | None:
    """Semitone offset from C for a letter plus optional accidental
    ("A", "A#", "Bb", "H"), or None."""
    if not spelling:
        return None
    letter = spelling[0].upper()
    if letter not in _NOTE_PITCH_CLASSES:
        return None
    rest = spelling[1:]
    if rest == '':
        return _NOTE_PITCH_CLASSES[letter]
    if rest == '#':
        return _NOTE_PITCH_CLASSES[letter] + 1
    if rest == 'b':
        return _NOTE_PITCH_CLASSES[letter] - 1
    return None


def note_name_to_midi(text: str) -> int | None:
    """Reverse of midi_note_name: accept "G3", "F#5", "Bb2", "B♭2", "g3",
    "A#/Bb3", "H3" and return the MIDI number, or None if the text does not
    parse or lies outside 0..127. The range editor's note-name field uses it
    so users need not think in MIDI integers."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Normalise unicode accidentals to ASCII.
    s = s.replace('♯', '#').replace('♭', 'b')
    # Split off the octave: everything from the first digit, or from a
    # minus sign that starts a negative octave like "C-1".
    i = 0
    while i < len(s) and not (s[i].isdigit() or s[i] == '-'):
        i += 1
    name, octave_text = s[:i], s[i:]
    if not octave_text:
        return None
    try:
        octave = int(octave_text)
    except ValueError:
        return None
    # midi_note_name prints black keys as both spellings ("A#/Bb3"). Accept
    # that form back as long as both halves name the same pitch class.
    offsets = {_pitch_class_offset(part) for part in name.split('/')}
    if len(offsets) != 1 or None in offsets:
        return None
    midi = (octave + 1) * 12 + offsets.pop()
    if midi < 0 or midi > 127:
        return None
    return midi


def format_cents(value_cents: float, freq_hz: float,
                 sample_rate: int) -> str:
    """Signed whole-cent readout: "+0", "-7", or "–" for no data.

    Always emits a sign so a neutral readout reads "+0" rather than a
    visually jumpy bare "0"; tiny negatives round to "+0". Integer cents
    only: a real saxophone tone vibratos by more than a cent, so sub-cent
    digits would claim resolution the reading cannot hold and only jitter
    the display.

    Non-finite input (a NaN mean after a reset race, a corrupt imported
    frequency) returns the "–" placeholder the other no-data cells use,
    since int(round(NaN)) raises. ``sample_rate`` is kept for call-site
    stability; the precision no longer depends on it.
    """
    if (not math.isfinite(value_cents) or not math.isfinite(freq_hz)
            or freq_hz <= 0):
        return '–'
    snapped = int(round(float(value_cents)))
    sign = '+' if snapped >= 0 else '-'
    return f"{sign}{abs(snapped)}"
