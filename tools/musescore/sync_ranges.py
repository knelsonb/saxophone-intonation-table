"""
Generate the catalogue's two baked range tables from MuseScore's instruments.xml.

Every catalogue instrument has two ranges, both in FINGERED (written) MIDI:

* keyed: the notes the instrument's keywork or normal compass gives. It seeds
  the table's blank guide rows and the matrix span. Source: MuseScore's
  ``aPitchRange``, unless KEYED_EXCEPTIONS records a better-supported value.
* extended: what skilled players reach. It decides whether a played note counts
  as in range (chart greying, the filter-to-range switch, the wrong-instrument
  hint). Source: MuseScore's ``pPitchRange``, widened where needed so it always
  contains the keyed range, unless EXTENDED_EXCEPTIONS records a labelled
  extension.

MuseScore's handbook calls ``aPitchRange`` the range of an "early amateur" (notes
outside it turn olive) and ``pPitchRange`` the professional range (notes outside
it turn red). Neither is a keywork specification, which is why saxophones and
clarinets carry recorded exceptions for the keyed tier.

How the numbers relate:
* ``aPitchRange`` and ``pPitchRange`` are SOUNDING MIDI, written ``lo-hi``.
* The catalogue's transposition is ``sounding - fingered``, so
  ``fingered = sounding - transposition``. The conversion uses the catalogue's
  own transposition, not MuseScore's ``transposeChromatic``: the catalogue
  writes the sousaphone and two recorders differently from MuseScore, and
  converting with its own value keeps the sounding range equal to MuseScore's.

Run ``python tools/musescore/sync_ranges.py`` to print both tables. The contract
tests require the printed blocks to appear verbatim in
intonation/catalog/instruments.py, so this script is the source of those tables:
change the data here, rerun, paste.

The vendored XML comes from
https://raw.githubusercontent.com/musescore/MuseScore/main/share/instruments/instruments.xml
(GPL-3.0).
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

XML = Path(__file__).resolve().parent / 'instruments.xml'
_REPO_ROOT = Path(__file__).resolve().parents[2]

Range = tuple[int, int]


# Catalogue key -> MuseScore instrument id.
MUSESCORE_ID = {
    # Saxes
    'eb_sopranino':       'sopranino-saxophone',
    'bb_soprano':         'soprano-saxophone',
    'eb_alto':            'alto-saxophone',
    'bb_tenor':           'tenor-saxophone',
    'eb_bari':            'baritone-saxophone',
    'bb_bass':            'bass-saxophone',
    'eb_contrabass':      'contrabass-saxophone',
    # Clarinets
    'clar_eb':            'eb-clarinet',
    'clar_d':             'd-clarinet',
    'clar_c':             'c-clarinet',
    'clar_bb':            'bb-clarinet',
    'clar_a':             'a-clarinet',
    'clar_basset_f':      'basset-horn',
    'clar_alto_eb':       'alto-clarinet',
    'clar_bass_bb':       'bb-bass-clarinet',
    'clar_contraalto_eb': 'contra-alto-clarinet',
    'clar_contrabass_bb': 'contrabass-clarinet',
    # Flutes
    'flute_piccolo':      'piccolo',
    'flute_c':            'flute',
    'flute_alto_g':       'alto-flute',
    'flute_bass_c':       'bass-flute',
    # Trumpets
    'trp_piccolo_bb':     'bb-piccolo-trumpet',
    'trp_piccolo_a':      'a-piccolo-trumpet',
    'trp_f':              'f-trumpet',
    'trp_e':              'e-trumpet',
    'trp_eb':             'eb-trumpet',
    'trp_d':              'd-trumpet',
    'trp_c':              'c-trumpet',
    'trp_bb':             'bb-trumpet',
    'trp_a':              'a-trumpet',
    'trp_bass_bb':        'bb-bass-trumpet',
    'cornet_bb':          'bb-cornet',
    'flugel_bb':          'flugelhorn',
    # Horns
    'horn_f':             'horn',
    'horn_bb':            'bb-horn-alto',
    'horn_eb_alto':       'eb-alto-horn',
    'mellophone_f':       'mellophone',
    # Trombones
    'tbn_alto_eb':        'alto-trombone',
    'tbn_tenor':          'trombone',
    'tbn_bass':           'bass-trombone',
    'tbn_contrabass':     'contrabass-trombone',
    # Low brass. 'baritone' in MuseScore is the singing voice; the bass-clef
    # horn is 'baritone-horn', the same horn as the treble-clef entry below.
    'euph_bc':            'euphonium',
    'euph_tc':            'euphonium-treble',
    'baritone_bc':        'baritone-horn',
    'baritone_tc':        'baritone-horn-treble',
    'tuba_f':             'f-tuba',
    'tuba_eb':            'eb-tuba',
    'tuba_cc':            'c-tuba',
    'tuba_bbb':           'bb-tuba',
    'sousaphone_bbb':     'bb-sousaphone',
    # Double reeds
    'oboe':               'oboe',
    'oboe_damore':        'oboe-d\'amore',
    'english_horn':       'english-horn',
    'bassoon':            'bassoon',
    'contrabassoon':      'contrabassoon',
    # Recorders
    'rec_sopranino_f':    'sopranino-recorder',
    'rec_soprano_c':      'soprano-recorder',
    'rec_alto_f':         'alto-recorder',
    'rec_tenor_c':        'tenor-recorder',
    'rec_bass_f':         'bass-recorder',
    # Strings
    'violin':             'violin',
    'viola':              'viola',
    'cello':              'violoncello',
    'double_bass':        'contrabass',
    'mandolin':           'mandolin',
    # Plucked
    'guitar':             'guitar-steel',
    'bass_guitar':        'bass-guitar',
    'ukulele':            'ukulele',
    'banjo':              'banjo',
    'harp':               'harp',
    # Concert / generic
    'voice':              'voice',
    'piano':              'piano',
}


# Catalogue keys with no MuseScore counterpart: one hand-set range used for
# both tiers, with the reason.
HAND_SET: dict[str, tuple[Range, str]] = {
    'c': ((48, 79), 'generic concert-pitch entry with no instrument to look up; '
                    'a two-and-a-half-octave middle range'),
}


_SAX_KEYWORK = (
    'saxophone keywork: written low Bb to high F#, as manufacturer key '
    'specifications list it; MuseScore\'s amateur top is a playing level, not a key')
_CLARINET_COMPASS = (
    'clarinet written compass E3 to C7, as VSL, the Iowa recordings and Buffet '
    'give it; MuseScore\'s amateur top is lower')

# Keyed tier: fingered (lo, hi) that replaces the converted aPitchRange, with
# the reason. A stale entry (equal to the MuseScore value) fails the contract
# tests, so this list only holds real departures.
KEYED_EXCEPTIONS: dict[str, tuple[Range, str]] = {
    # Saxophones
    'eb_sopranino': ((58, 90), _SAX_KEYWORK + '; the sopranino top is not '
                               'confirmed against a maker\'s page'),
    'bb_soprano': ((58, 90), _SAX_KEYWORK),
    'eb_alto': ((58, 90), _SAX_KEYWORK),
    'bb_tenor': ((58, 90), _SAX_KEYWORK),
    'eb_bari': ((57, 90), _SAX_KEYWORK + '; the baritone adds the low A key'),
    'bb_bass': ((58, 90), _SAX_KEYWORK + '; MuseScore\'s amateur top G6 is above '
                          'the high F# key'),
    'eb_contrabass': ((58, 90), _SAX_KEYWORK + '; the contrabass top is not '
                                'confirmed against a maker\'s page'),
    # Clarinets
    'clar_eb': ((52, 96), _CLARINET_COMPASS),
    'clar_d': ((52, 96), _CLARINET_COMPASS),
    'clar_c': ((52, 96), _CLARINET_COMPASS),
    'clar_bb': ((52, 96), _CLARINET_COMPASS),
    'clar_a': ((52, 96), _CLARINET_COMPASS),
    'clar_basset_f': ((48, 96), 'basset extension down to written C3, clarinet '
                                'compass up to C7'),
    'clar_bass_bb': ((48, 96), 'keyed down to written low C (Buffet BC1193); top '
                               'C7 as VSL and the Iowa recordings give it'),
    'clar_contraalto_eb': ((48, 91), 'low C extension reported by a player; '
                                     'MuseScore stops at written low Eb'),
    'clar_contrabass_bb': ((48, 91), 'keyed down to written low C, which '
                                     'MuseScore\'s professional range also reaches'),
}

# Extended tier: fingered (lo, hi) that replaces the default (pPitchRange
# widened to contain the keyed range), with the reason. Every extended range
# must sound inside the detector window, MIDI 21..108.
EXTENDED_EXCEPTIONS: dict[str, tuple[Range, str]] = {
    'bb_soprano': ((58, 101), 'altissimo to written F7, as MuseScore gives the '
                              'alto, tenor and baritone; its soprano stops at A6'),
    'tbn_contrabass': ((21, 72), 'MuseScore\'s professional low end, F#0, sounds '
                                 'below A0, the lowest note the detector reads'),
}


def parse_instruments(path: Path = XML) -> dict[str, dict]:
    """Return {musescore_id: {'transp', 'a_lo', 'a_hi', 'p_lo', 'p_hi'}}.

    The range fields are SOUNDING MIDI; an instrument missing one tier lacks
    those two fields."""
    root = ET.parse(path).getroot()
    out: dict[str, dict] = {}
    for inst in root.iter('Instrument'):
        mid = inst.attrib.get('id')
        if not mid:
            continue
        entry: dict = {}
        for tag, prefix in (('aPitchRange', 'a'), ('pPitchRange', 'p')):
            m = re.match(r'(\d+)-(\d+)', (inst.findtext(tag) or '').strip())
            if m:
                entry[f'{prefix}_lo'], entry[f'{prefix}_hi'] = int(m.group(1)), int(m.group(2))
        if not entry:
            continue
        try:
            entry['transp'] = int((inst.findtext('transposeChromatic') or '0').strip())
        except ValueError:
            entry['transp'] = 0
        out[mid] = entry
    return out


def sounding_tier(entry: dict, tier: str) -> Range:
    """One tier ('a' amateur or 'p' professional) as SOUNDING (lo, hi)."""
    return (entry[f'{tier}_lo'], entry[f'{tier}_hi'])


def to_fingered(sounding: Range, transp: int) -> Range:
    """Sounding (lo, hi) to fingered, with the catalogue's transposition."""
    return (sounding[0] - transp, sounding[1] - transp)


def hull(a: Range, b: Range) -> Range:
    """The smallest range containing both."""
    return (min(a[0], b[0]), max(a[1], b[1]))


def catalogue_families() -> list:
    """The catalogue's baked family table (never runtime customs)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from intonation.catalog import instruments
    return instruments._FAMILIES


def _transpositions(families) -> dict[str, int]:
    return {key: transp for (_fk, _de, _en, instrs) in families
            for (key, transp, _d, _e) in instrs}


def default_keyed(entry: dict, transp: int) -> Range:
    return to_fingered(sounding_tier(entry, 'a'), transp)


def default_extended(entry: dict, transp: int, keyed: Range) -> Range:
    return hull(to_fingered(sounding_tier(entry, 'p'), transp), keyed)


def generate_ranges(musescore: dict[str, dict] | None = None,
                    families=None) -> tuple[dict[str, Range], dict[str, Range], dict[str, str]]:
    """Build (keyed, extended, sources) for every catalogue key, in catalogue
    order. ``sources`` holds a short provenance tag per key for the printed
    table. Raises KeyError for a catalogue key with no decision."""
    if musescore is None:
        musescore = parse_instruments()
    if families is None:
        families = catalogue_families()
    keyed: dict[str, Range] = {}
    extended: dict[str, Range] = {}
    keyed_src: dict[str, str] = {}
    extended_src: dict[str, str] = {}
    for key, transp in _transpositions(families).items():
        if key in HAND_SET:
            keyed[key] = extended[key] = HAND_SET[key][0]
            keyed_src[key] = extended_src[key] = 'hand-set'
            continue
        ms_id = MUSESCORE_ID[key]
        entry = musescore[ms_id]
        if key in KEYED_EXCEPTIONS:
            keyed[key] = KEYED_EXCEPTIONS[key][0]
            keyed_src[key] = f'{ms_id}, keyed exception'
        else:
            keyed[key] = default_keyed(entry, transp)
            keyed_src[key] = ms_id
        if key in EXTENDED_EXCEPTIONS:
            extended[key] = EXTENDED_EXCEPTIONS[key][0]
            extended_src[key] = f'{ms_id}, extension'
        else:
            extended[key] = default_extended(entry, transp, keyed[key])
            professional = to_fingered(sounding_tier(entry, 'p'), transp)
            extended_src[key] = (ms_id if extended[key] == professional
                                 else f'{ms_id}, widened to keyed')
    return keyed, extended, {'keyed': keyed_src, 'extended': extended_src}


def render_table(name: str, table: dict[str, Range], sources: dict[str, str],
                 families=None) -> str:
    """Python source for one baked table, grouped by family, as it appears in
    intonation/catalog/instruments.py."""
    if families is None:
        families = catalogue_families()
    lines = [f'{name}: dict[str, tuple[int, int]] = {{']
    for (_fk, _de, family_en, instrs) in families:
        lines.append(f'    # {family_en}')
        for (key, _t, _d, _e) in instrs:
            lo, hi = table[key]
            lines.append(f"    {repr(key) + ':':25s} ({lo:3d}, {hi:3d}),   # {sources[key]}")
    lines.append('}')
    return '\n'.join(lines)


def render_tables(families=None) -> tuple[str, str]:
    """(keyed source, extended source) for the current XML and data."""
    if families is None:
        families = catalogue_families()
    keyed, extended, sources = generate_ranges(families=families)
    return (render_table('_KEYED_RANGES', keyed, sources['keyed'], families),
            render_table('_EXTENDED_RANGES', extended, sources['extended'], families))


def main() -> int:
    if not XML.exists():
        print(f'ERROR: {XML} missing; run\n'
              '  curl -sSL https://raw.githubusercontent.com/musescore'
              '/MuseScore/main/share/instruments/instruments.xml '
              f'-o {XML}', file=sys.stderr)
        return 1
    keyed_src, extended_src = render_tables()
    print(keyed_src)
    print()
    print(extended_src)
    return 0


if __name__ == '__main__':
    sys.exit(main())
