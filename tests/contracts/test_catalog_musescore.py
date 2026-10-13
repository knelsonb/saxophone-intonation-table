"""Instrument catalog versus its source of truth, MuseScore's instruments.xml.

The catalog (sax_instruments) stores a transposition and a baked FINGERED range
per instrument; both were derived from tools/musescore/instruments.xml (see
tools/musescore/sync_ranges.py). In that file ``transposeChromatic`` is
``sounding - written`` and ``aPitchRange`` is a SOUNDING range. So for every key
the catalog maps to a MuseScore id the contract is:

  * catalog transposition == transposeChromatic
  * baked fingered range shifted by the transposition == aPitchRange

The catalog departs from MuseScore on purpose in a few places. Each departure is
recorded in KNOWN_DIVERGENCES with the exact values on both sides and the reason,
so an unrecorded change in either direction fails, and so does a stale entry.

Only the baked tables are read here (``baked_fingered_range``, ``_FAMILIES``);
the user's ``~/.intonation_analyzer`` overrides and runtime customs never enter.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import sax_instruments as si

_SYNC_PATH = (Path(__file__).resolve().parents[2]
              / 'tools' / 'musescore' / 'sync_ranges.py')
_spec = importlib.util.spec_from_file_location('musescore_sync_ranges', _SYNC_PATH)
sync_ranges = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_ranges)

MUSESCORE = sync_ranges.parse_instruments()
MUSESCORE_ID: dict[str, str] = sync_ranges.MUSESCORE_ID

# Catalog transposition per key, from the baked family table (never from
# transp_map(), which also folds in runtime-registered customs).
CATALOG_TRANSP: dict[str, int] = {
    key: transp
    for (_fk, _de, _en, instruments) in si._FAMILIES
    for (key, transp, _d, _e) in instruments
}

SOUNDING_WINDOW = (21, 108)   # A0..C8, what the detector can read

# Catalog keys with no MuseScore counterpart; their ranges are hand-set.
UNMAPPED = {
    'mellophone_f': 'no MuseScore entry; range set by hand',
    'c': 'generic concert-pitch fallback entry',
}

_SAX_REASON = (
    'low A: fingered A3 (57) is one semitone below the lowest note MuseScore '
    'lists; top widened to fingered C7 (96) so altissimo work is not clipped')
_CLARINET_TOP_REASON = (
    'top widened to fingered C7 (96) for altissimo; MuseScore amateur range '
    'ends lower')
_CLARINET_LOW_REASON = (
    'low extension: lowest note set to fingered C (48) by hand, below the '
    'MuseScore amateur range; top widened for altissimo')

# key -> {'transp': (catalog, musescore)?, 'range': (catalog_sounding,
# musescore_sounding)?, 'reason': str}. Sounding ranges are (lo, hi) MIDI.
KNOWN_DIVERGENCES: dict[str, dict] = {
    # Saxophones
    'eb_sopranino': {'range': ((60, 99), (61, 87)), 'reason': _SAX_REASON},
    'bb_soprano': {'range': ((55, 94), (56, 87)), 'reason': _SAX_REASON},
    'eb_alto': {'range': ((48, 87), (49, 80)), 'reason': _SAX_REASON},
    'bb_tenor': {'range': ((43, 82), (44, 75)), 'reason': _SAX_REASON},
    'eb_bari': {'range': ((36, 75), (36, 68)), 'reason': _SAX_REASON},
    'bb_bass': {'range': ((31, 70), (32, 65)), 'reason': _SAX_REASON},
    'eb_contrabass': {'range': ((24, 63), (25, 53)), 'reason': _SAX_REASON},
    # Clarinets
    'clar_eb': {'range': ((55, 99), (55, 94)), 'reason': _CLARINET_TOP_REASON},
    'clar_d': {'range': ((54, 98), (54, 93)), 'reason': _CLARINET_TOP_REASON},
    'clar_c': {'range': ((52, 96), (52, 89)), 'reason': _CLARINET_TOP_REASON},
    'clar_bb': {'range': ((50, 94), (50, 89)), 'reason': _CLARINET_TOP_REASON},
    'clar_a': {'range': ((49, 93), (49, 88)), 'reason': _CLARINET_TOP_REASON},
    'clar_basset_f': {'range': ((41, 89), (41, 84)),
                      'reason': _CLARINET_TOP_REASON},
    'clar_alto_eb': {'range': ((43, 82), (43, 80)),
                     'reason': _CLARINET_TOP_REASON},
    'clar_bass_bb': {'range': ((34, 82), (37, 75)),
                     'reason': _CLARINET_LOW_REASON},
    'clar_contraalto_eb': {'range': ((27, 70), (30, 70)),
                           'reason': _CLARINET_LOW_REASON},
    'clar_contrabass_bb': {'range': ((22, 65), (25, 65)),
                           'reason': _CLARINET_LOW_REASON},
    # Transposition conventions. The sounding ranges agree with MuseScore, so
    # only the written spelling of the instrument differs.
    'sousaphone_bbb': {
        'transp': (0, -14),
        'reason': 'catalog stores the sousaphone like the other tubas '
                  '(written = sounding); MuseScore writes it 14 semitones up'},
    'rec_sopranino_f': {
        'transp': (12, 0),
        'reason': 'recorder convention: written an octave below sounding; '
                  'MuseScore carries no transposition tag for recorders, so '
                  'the parser reads 0'},
    'rec_soprano_c': {
        'transp': (12, 0),
        'reason': 'recorder convention: written an octave below sounding; '
                  'MuseScore carries no transposition tag for recorders, so '
                  'the parser reads 0'},
}


def _musescore_sounding_range(entry: dict) -> tuple[int, int]:
    if 'a_lo' in entry:
        return (entry['a_lo'], entry['a_hi'])
    return (entry['p_lo'], entry['p_hi'])


def _measured_divergence(key: str) -> dict:
    ms = MUSESCORE[MUSESCORE_ID[key]]
    transp = CATALOG_TRANSP[key]
    lo, hi = si.baked_fingered_range(key)
    sounding = (lo + transp, hi + transp)
    diff: dict = {}
    if transp != ms['transp']:
        diff['transp'] = (transp, ms['transp'])
    ms_sounding = _musescore_sounding_range(ms)
    if sounding != ms_sounding:
        diff['range'] = (sounding, ms_sounding)
    return diff


# ---------------------------------------------------------------------------
# Catalog versus MuseScore
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('key', sorted(MUSESCORE_ID))
def test_catalog_matches_musescore_or_is_a_known_divergence(key: str) -> None:
    assert key in CATALOG_TRANSP, f'{key} is mapped to MuseScore but not in the catalog'
    expected = {k: v for k, v in KNOWN_DIVERGENCES.get(key, {}).items()
                if k != 'reason'}
    assert _measured_divergence(key) == expected


def test_every_divergence_names_a_mapped_key_and_gives_a_reason() -> None:
    for key, entry in KNOWN_DIVERGENCES.items():
        assert key in MUSESCORE_ID, f'{key}: divergence for an unmapped key'
        assert entry['reason'].strip(), f'{key}: divergence without a reason'
        assert set(entry) <= {'transp', 'range', 'reason'}


def test_every_catalog_key_is_mapped_or_explicitly_unmapped() -> None:
    unaccounted = set(CATALOG_TRANSP) - set(MUSESCORE_ID) - set(UNMAPPED)
    assert not unaccounted, f'catalog keys with no MuseScore decision: {sorted(unaccounted)}'
    assert set(UNMAPPED) <= set(CATALOG_TRANSP)
    assert not set(UNMAPPED) & set(MUSESCORE_ID)


def test_musescore_ids_resolve_and_are_unique() -> None:
    missing = [k for k, mid in MUSESCORE_ID.items() if mid not in MUSESCORE]
    assert not missing, f'ids absent from instruments.xml: {missing}'
    ids = list(MUSESCORE_ID.values())
    assert len(ids) == len(set(ids)), 'two catalog keys share one MuseScore id'


def test_oracle_parses_known_instruments() -> None:
    """Pins the parser against the XML so the comparison cannot pass vacuously."""
    assert len(MUSESCORE) > 100
    tenor = MUSESCORE['tenor-saxophone']
    assert tenor['transp'] == -14
    assert (tenor['a_lo'], tenor['a_hi']) == (44, 75)
    assert sync_ranges.fingered_range_from_ms(tenor) == (58, 89)


def test_sync_script_conversion_round_trips_every_mapped_instrument() -> None:
    for key, ms_id in MUSESCORE_ID.items():
        entry = MUSESCORE[ms_id]
        lo, hi = sync_ranges.fingered_range_from_ms(entry)
        assert (lo + entry['transp'], hi + entry['transp']) == (
            _musescore_sounding_range(entry)), key


# ---------------------------------------------------------------------------
# Baked table properties, for every catalog key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('key', sorted(CATALOG_TRANSP))
def test_baked_sounding_range_is_inside_detector_window(key: str) -> None:
    assert si.has_baked_range(key), f'{key} falls back to the generic range'
    lo, hi = si.baked_fingered_range(key)
    transp = CATALOG_TRANSP[key]
    assert 0 <= lo <= hi <= 127
    assert SOUNDING_WINDOW[0] <= lo + transp, f'{key} sounds below A0'
    assert hi + transp <= SOUNDING_WINDOW[1], f'{key} sounds above C8'
    assert 12 <= hi - lo + 1 <= 88, 'table seeds at least an octave, at most the piano'


@pytest.mark.parametrize('key', sorted(CATALOG_TRANSP))
def test_written_sounding_round_trip(key: str) -> None:
    lo, hi = si.baked_fingered_range(key)
    transp = CATALOG_TRANSP[key]
    for written in range(lo, hi + 1):
        sounding = written + transp
        assert SOUNDING_WINDOW[0] <= sounding <= SOUNDING_WINDOW[1]
        assert sounding - transp == written


def test_catalog_transposition_agrees_with_transp_map_without_customs(monkeypatch) -> None:
    monkeypatch.setattr(si, '_CUSTOMS', [])
    assert si.transp_map() == CATALOG_TRANSP


def test_baked_range_ignores_user_overrides(monkeypatch, tmp_path) -> None:
    """The contract tables read the baked ranges only; a user override must not
    leak into them, though fingered_range() follows it."""
    overrides = tmp_path / 'instrument_ranges.json'
    overrides.write_text('{"version": 1, "ranges": {"eb_alto": [1, 2]}}', encoding='utf-8')
    monkeypatch.setattr(si, '_active_overrides', si.RangeOverrides(overrides))
    assert si.fingered_range('eb_alto') == (1, 2)
    assert si.baked_fingered_range('eb_alto') == (57, 96)
