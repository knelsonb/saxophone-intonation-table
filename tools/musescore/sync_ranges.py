"""
Sync instrument fingered-MIDI ranges from MuseScore's instruments.xml into
the Intonation Analyzer's catalog.

MuseScore is open-source, ~10M users; the ranges in their instruments.xml
are vetted by orchestrators and notation experts and are about as close
as we'll get to a "single source of truth" for instrument ranges. We use
their **amateur** range (`aPitchRange`) as our default — it's the
"competent working player" envelope, broader than student-only and
narrower than virtuoso-only.

How transposition + ranges relate (verified against tenor sax):
* `aPitchRange` and `pPitchRange` are SOUNDING pitch (concert), expressed
  as MIDI numbers in the form ``lo-hi``.
* `transposeChromatic` is ``sounding_midi - written_midi`` (negative when
  the instrument sounds lower than written).
* Our catalog stores FINGERED (written) ranges. Convert via
  ``written = sounding - transposeChromatic``.

Tom Nelson's hard overrides (from real-player feedback) win over
MuseScore where they conflict — saxophones at low A, contras at low C.
The override list is explicit in OVERRIDES.

Output: a freshly generated `_RANGES` dict that we paste into
sax_instruments.py (the script prints it; the caller chooses what to
merge).
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


XML = Path(__file__).resolve().parent / 'instruments.xml'


# Our catalog key → MuseScore instrument id. Keys without a MuseScore
# equivalent are left commented out; they keep whatever range the catalog
# already has.
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
    # mellophone — no MuseScore entry; keeps existing range
    # Trombones
    'tbn_alto_eb':        'alto-trombone',
    'tbn_tenor':          'trombone',
    'tbn_bass':           'bass-trombone',
    'tbn_contrabass':     'contrabass-trombone',
    # Low brass
    'euph_bc':            'euphonium',
    'euph_tc':            'euphonium-treble',
    'baritone_bc':        'baritone',
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
    # 'c': intentionally no mapping — generic concert-pitch fallback
    'piano':              'piano',
}


# Hard overrides from real-player feedback (Tom Nelson). These win over
# whatever MuseScore says.
OVERRIDES: dict[str, tuple[int, int]] = {
    'eb_sopranino':       (57, 78),
    'bb_soprano':         (57, 78),
    'eb_alto':            (57, 78),
    'bb_tenor':           (57, 78),
    'eb_bari':            (57, 78),
    'bb_bass':            (57, 78),
    'eb_contrabass':      (57, 78),
    'clar_contraalto_eb': (48,),   # 1-tuple = override LO only, keep MS HI
    'clar_contrabass_bb': (48,),
}


def parse_instruments() -> dict[str, dict]:
    """Return {musescore_id: {'a_lo','a_hi','p_lo','p_hi','transp'}}."""
    tree = ET.parse(XML)
    root = tree.getroot()
    out: dict[str, dict] = {}
    for inst in root.iter('Instrument'):
        mid = inst.attrib.get('id')
        if not mid:
            continue
        a = (inst.findtext('aPitchRange') or '').strip()
        p = (inst.findtext('pPitchRange') or '').strip()
        tc = (inst.findtext('transposeChromatic') or '0').strip()
        if not a and not p:
            continue
        try:
            transp = int(tc)
        except ValueError:
            transp = 0
        m = re.match(r'(\d+)-(\d+)', a or p)
        if not m:
            continue
        entry = {'transp': transp}
        if a:
            am = re.match(r'(\d+)-(\d+)', a)
            if am:
                entry['a_lo'], entry['a_hi'] = int(am.group(1)), int(am.group(2))
        if p:
            pm = re.match(r'(\d+)-(\d+)', p)
            if pm:
                entry['p_lo'], entry['p_hi'] = int(pm.group(1)), int(pm.group(2))
        out[mid] = entry
    return out


def fingered_range_from_ms(entry: dict) -> tuple[int, int]:
    """Convert MuseScore sounding range to FINGERED range.

    written = sounding - transposeChromatic. We use the amateur range as
    the default; fall back to professional if amateur is missing."""
    transp = entry.get('transp', 0)
    if 'a_lo' in entry:
        lo, hi = entry['a_lo'], entry['a_hi']
    else:
        lo, hi = entry['p_lo'], entry['p_hi']
    return (lo - transp, hi - transp)


def main() -> int:
    if not XML.exists():
        print(f'ERROR: {XML} missing — run\n'
              '  curl -sSL https://raw.githubusercontent.com/musescore'
              '/MuseScore/master/share/instruments/instruments.xml '
              f'-o {XML}', file=sys.stderr)
        return 1

    instruments = parse_instruments()
    missing: list[str] = []
    out_lines: list[str] = []
    out_lines.append('_RANGES: dict[str, tuple[int, int]] = {')

    # Group output by family for readability (matches the existing layout
    # so the diff is small).
    groups = [
        ('Saxophones', [
            'eb_sopranino', 'bb_soprano', 'eb_alto', 'bb_tenor',
            'eb_bari', 'bb_bass', 'eb_contrabass',
        ]),
        ('Clarinets', [
            'clar_eb', 'clar_d', 'clar_c', 'clar_bb', 'clar_a',
            'clar_basset_f', 'clar_alto_eb', 'clar_bass_bb',
            'clar_contraalto_eb', 'clar_contrabass_bb',
        ]),
        ('Flutes', [
            'flute_piccolo', 'flute_c', 'flute_alto_g', 'flute_bass_c',
        ]),
        ('Trumpets', [
            'trp_piccolo_bb', 'trp_piccolo_a', 'trp_f', 'trp_e', 'trp_eb',
            'trp_d', 'trp_c', 'trp_bb', 'trp_a', 'trp_bass_bb',
            'cornet_bb', 'flugel_bb',
        ]),
        ('Horns', [
            'horn_f', 'horn_bb', 'horn_eb_alto', 'mellophone_f',
        ]),
        ('Trombones', [
            'tbn_alto_eb', 'tbn_tenor', 'tbn_bass', 'tbn_contrabass',
        ]),
        ('Low brass', [
            'euph_bc', 'euph_tc', 'baritone_bc', 'baritone_tc',
            'tuba_f', 'tuba_eb', 'tuba_cc', 'tuba_bbb', 'sousaphone_bbb',
        ]),
        ('Double reeds', [
            'oboe', 'oboe_damore', 'english_horn', 'bassoon', 'contrabassoon',
        ]),
        ('Recorders', [
            'rec_sopranino_f', 'rec_soprano_c', 'rec_alto_f', 'rec_tenor_c',
            'rec_bass_f',
        ]),
        ('Strings', [
            'violin', 'viola', 'cello', 'double_bass', 'mandolin',
        ]),
        ('Plucked', [
            'guitar', 'bass_guitar', 'ukulele', 'banjo', 'harp',
        ]),
        ('Concert / generic', ['voice', 'c', 'piano']),
    ]

    for group_name, keys in groups:
        out_lines.append(f'    # {group_name}')
        for key in keys:
            ms_id = MUSESCORE_ID.get(key)
            if ms_id is None:
                missing.append(f'{key}: no MuseScore id (manual range kept)')
                continue
            ent = instruments.get(ms_id)
            if not ent:
                missing.append(f'{key}: MS id {ms_id!r} not found in xml')
                continue
            lo, hi = fingered_range_from_ms(ent)
            ov = OVERRIDES.get(key)
            if ov:
                if len(ov) == 2:
                    lo, hi = ov
                    src = 'override'
                elif len(ov) == 1:
                    lo = ov[0]
                    src = 'override-lo'
                else:
                    src = 'MS'
            else:
                src = 'MS'
            out_lines.append(
                f"    {repr(key) + ':':25s} ({lo:3d}, {hi:3d}),   # {src} via {ms_id}"
            )
    out_lines.append('}')
    print('\n'.join(out_lines))
    if missing:
        print('\n# Notes:', file=sys.stderr)
        for m in missing:
            print(f'#   - {m}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
