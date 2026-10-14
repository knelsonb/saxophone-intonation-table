"""The two range tiers and how a user override combines with them.

keyed_range() seeds the guide rows and the matrix span; extended_range() decides
whether a played note is in range. A range-editor override replaces the keyed
tier and widens the extended tier; the file format (one fingered (lo, hi) per
key) is the same as before the tiers existed.

baked_ranges.json pins every baked range of both tiers, with note names, so a
change to the catalogue's ranges is visible in review as a golden diff.
"""
from __future__ import annotations

import json

import pytest

from intonation.catalog import instruments as si
from intonation.core.pitch import midi_note_name
from tests.harness.golden import assert_golden


@pytest.fixture()
def overrides(monkeypatch, tmp_path):
    """Install an override store for one test; returns a writer for it."""
    path = tmp_path / 'instrument_ranges.json'
    monkeypatch.setattr(si, '_active_overrides', si.RangeOverrides(path))

    def write(ranges: dict) -> None:
        path.write_text(json.dumps({'version': 1, 'ranges': ranges}), encoding='utf-8')
        si.range_overrides().load()
    return write


@pytest.fixture()
def custom(monkeypatch):
    monkeypatch.setattr(si, '_CUSTOMS', [])
    si.register_custom('custom_test_horn', -5, 'Testhorn', 'Test Horn')
    return 'custom_test_horn'


def test_without_an_override_each_tier_is_its_baked_value(overrides):
    overrides({})
    assert si.keyed_range('eb_alto') == si.baked_keyed_range('eb_alto') == (58, 90)
    assert si.extended_range('eb_alto') == si.baked_extended_range('eb_alto') == (58, 101)


def test_an_override_replaces_the_keyed_tier(overrides):
    overrides({'eb_alto': [60, 84]})
    assert si.keyed_range('eb_alto') == (60, 84)
    assert si.baked_keyed_range('eb_alto') == (58, 90)


def test_a_narrower_override_leaves_the_extended_tier_as_baked(overrides):
    overrides({'eb_alto': [60, 84]})
    assert si.extended_range('eb_alto') == (58, 101)


def test_an_override_beyond_the_baked_extended_tier_widens_it(overrides):
    overrides({'eb_alto': [57, 103]})                      # low A, altissimo G7
    assert si.extended_range('eb_alto') == (57, 103)
    overrides({'eb_alto': [57, 80]})
    assert si.extended_range('eb_alto') == (57, 101)


def test_an_override_apart_from_the_baked_range_spans_the_gap(overrides):
    overrides({'cello': [100, 105]})
    assert si.extended_range('cello') == (36, 105)


def test_the_legacy_names_follow_the_keyed_tier(overrides):
    overrides({'eb_bari': [45, 77]})
    assert si.fingered_range('eb_bari') == si.keyed_range('eb_bari') == (45, 77)
    assert si.baked_fingered_range('eb_bari') == si.baked_keyed_range('eb_bari')


def test_a_custom_instrument_uses_the_fallback_for_both_tiers(overrides, custom):
    overrides({})
    assert not si.has_baked_range(custom)
    assert si.keyed_range(custom) == si.extended_range(custom) == (48, 79)
    assert si.baked_keyed_range(custom) == si.baked_extended_range(custom) == (48, 79)


def test_a_custom_instrument_override_is_its_whole_range(overrides, custom):
    """The fallback is a placeholder, not data about the instrument, so it does
    not widen the extended tier the way baked data does."""
    overrides({custom: [60, 72]})
    assert si.keyed_range(custom) == si.extended_range(custom) == (60, 72)


def test_saving_an_override_keeps_the_file_format(overrides, tmp_path):
    overrides({})
    si.save_range_override('eb_alto', 57, 96)
    data = json.loads((tmp_path / 'instrument_ranges.json').read_text(encoding='utf-8'))
    assert data == {'version': 1, 'ranges': {'eb_alto': [57, 96]}}
    assert si.keyed_range('eb_alto') == (57, 96)
    assert si.extended_range('eb_alto') == (57, 101)


def _describe(rng: tuple[int, int]) -> str:
    lo, hi = rng
    return f'{midi_note_name(lo)}..{midi_note_name(hi)} ({lo}..{hi})'


def test_baked_ranges_match_golden(monkeypatch):
    monkeypatch.setattr(si, '_CUSTOMS', [])
    table = {
        key: {
            'transposition': transp,
            'keyed': _describe(si.baked_keyed_range(key)),
            'extended': _describe(si.baked_extended_range(key)),
        }
        for key, transp in si.transp_map().items()
    }
    assert_golden(table, 'expected/baked_ranges.json')
