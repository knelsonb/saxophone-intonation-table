"""
Instrument catalog: families, sub-instruments, transposition values, and
display names in DE + EN.

Transposition convention: `transp = sounding_midi - fingered_midi`, including
the octave, so the sounding name reads in the right register (Eb alto -9,
Bb tenor -14, Eb baritone -21; double bass and bass guitar -12; recorders +12).
Values follow MuseScore's `transposeChromatic` for each instrument.

v0.4.2 corrected the saxophone values (alto +3 -> -9, tenor -2 -> -14,
baritone -9 -> -21). Logs and CSVs written before that carry `midi_fingered`
values an octave off from current ones; the sounding pitch is unaffected.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from intonation.persistence.atomic import atomic_write_json
from intonation.persistence.paths import AppPaths

# (family_key, family_name_de, family_name_en,
#   [(instrument_key, transp, name_de, name_en), ...])
_FAMILIES = [
    ('saxophone', 'Saxophon', 'Saxophone', [
        # Real acoustic transpositions — sounding pitch is not octave-
        # collapsed. Tenor sax really sounds M9 below written (octave +
        # major 2nd); baritone an octave + M6 below; bass two octaves +
        # M2 below; contrabass two octaves + M6 below.
        ('eb_sopranino',    +3, 'Eb-Sax · Sopranino', 'Eb Sax · Sopranino'),
        ('bb_soprano',      -2, 'Bb-Sax · Sopran',    'Bb Sax · Soprano'),
        ('eb_alto',         -9, 'Eb-Sax · Alt',       'Eb Sax · Alto'),
        ('bb_tenor',       -14, 'Bb-Sax · Tenor',     'Bb Sax · Tenor'),
        ('eb_bari',        -21, 'Eb-Sax · Bariton',   'Eb Sax · Baritone'),
        ('bb_bass',        -26, 'Bb-Sax · Bass',      'Bb Sax · Bass'),
        ('eb_contrabass',  -33, 'Eb-Sax · Kontrabass', 'Eb Sax · Contrabass'),
    ]),
    ('clarinet', 'Klarinette', 'Clarinet', [
        ('clar_eb',           +3, 'Eb-Klarinette',          'Eb Clarinet'),
        ('clar_d',            +2, 'D-Klarinette',           'D Clarinet'),
        ('clar_c',             0, 'C-Klarinette',           'C Clarinet'),
        ('clar_bb',           -2, 'Bb-Klarinette',          'Bb Clarinet'),
        ('clar_a',            -3, 'A-Klarinette',           'A Clarinet'),
        ('clar_basset_f',     -7, 'Bassetthorn (F)',        'Basset Horn (F)'),
        ('clar_alto_eb',      -9, 'Eb-Altklarinette',       'Eb Alto Clarinet'),
        ('clar_bass_bb',     -14, 'Bb-Bassklarinette',      'Bb Bass Clarinet'),
        ('clar_contraalto_eb', -21,
         'Eb-Kontra-Altklarinette', 'Eb Contra-Alto Clarinet'),
        ('clar_contrabass_bb', -26,
         'Bb-Kontrabassklarinette', 'Bb Contrabass Clarinet'),
    ]),
    ('flute', 'Flöte', 'Flute', [
        ('flute_piccolo',   +12, 'Piccolo (C)',         'Piccolo (C)'),
        ('flute_c',           0, 'Querflöte (C)',       'Concert Flute (C)'),
        ('flute_alto_g',     -5, 'Altflöte (G)',        'Alto Flute (G)'),
        ('flute_bass_c',    -12, 'Bassflöte (C)',       'Bass Flute (C)'),
    ]),
    ('trumpet', 'Trompete', 'Trumpet', [
        ('trp_piccolo_bb',  +10, 'Piccolo-Trompete Bb', 'Piccolo Trumpet Bb'),
        ('trp_piccolo_a',    +9, 'Piccolo-Trompete A',  'Piccolo Trumpet A'),
        ('trp_f',            +5, 'F-Trompete',          'F Trumpet'),
        ('trp_e',            +4, 'E-Trompete',          'E Trumpet'),
        ('trp_eb',           +3, 'Eb-Trompete',         'Eb Trumpet'),
        ('trp_d',            +2, 'D-Trompete',          'D Trumpet'),
        ('trp_c',             0, 'C-Trompete',          'C Trumpet'),
        ('trp_bb',           -2, 'Bb-Trompete',         'Bb Trumpet'),
        ('trp_a',            -3, 'A-Trompete',          'A Trumpet'),
        ('trp_bass_bb',     -14, 'Bb-Basstrompete',     'Bb Bass Trumpet'),
        ('cornet_bb',        -2, 'Bb-Kornett',          'Bb Cornet'),
        ('flugel_bb',        -2, 'Bb-Flügelhorn',       'Bb Flugelhorn'),
    ]),
    ('horn', 'Horn', 'Horn', [
        ('horn_f',           -7, 'F-Horn',              'F Horn'),
        ('horn_bb',          -2, 'Bb-Horn',             'Bb Horn'),
        ('horn_eb_alto',     -9, 'Eb-Althorn',          'Eb Alto Horn'),
        ('mellophone_f',     -7, 'Mellophon F',         'Mellophone F'),
    ]),
    ('trombone', 'Posaune', 'Trombone', [
        ('tbn_alto_eb',       0, 'Altposaune Eb',       'Alto Trombone Eb'),
        ('tbn_tenor',         0, 'Tenorposaune',        'Tenor Trombone'),
        ('tbn_bass',          0, 'Bassposaune',         'Bass Trombone'),
        ('tbn_contrabass',    0, 'Kontrabassposaune',   'Contrabass Trombone'),
    ]),
    ('low_brass', 'Tiefes Blech', 'Low Brass', [
        ('euph_bc',           0, 'Euphonium (Bassschl.)',  'Euphonium (Bass Clef)'),
        ('euph_tc',         -14, 'Euphonium (Violinschl.)', 'Euphonium (Treble Clef)'),
        ('baritone_bc',       0, 'Baritonhorn (Bassschl.)', 'Baritone Horn (BC)'),
        ('baritone_tc',     -14, 'Baritonhorn (Violinschl.)', 'Baritone Horn (TC)'),
        ('tuba_f',            0, 'F-Tuba',              'F Tuba'),
        ('tuba_eb',           0, 'Eb-Tuba',             'Eb Tuba'),
        ('tuba_cc',           0, 'CC-Tuba',             'CC Tuba'),
        ('tuba_bbb',          0, 'BBb-Tuba',            'BBb Tuba'),
        ('sousaphone_bbb',    0, 'Sousaphon BBb',       'Sousaphone BBb'),
    ]),
    ('double_reed', 'Doppelrohrblatt', 'Double Reed', [
        ('oboe',              0, 'Oboe',                'Oboe'),
        ('oboe_damore',      -3, 'Oboe d’amore',        'Oboe d’amore'),
        ('english_horn',     -7, 'Englischhorn (F)',    'English Horn (F)'),
        ('bassoon',           0, 'Fagott',              'Bassoon'),
        ('contrabassoon',   -12, 'Kontrafagott',        'Contrabassoon'),
    ]),
    ('recorder', 'Blockflöte', 'Recorder', [
        ('rec_sopranino_f', +12, 'Sopranino-Blockflöte (F)', 'Sopranino Recorder (F)'),
        ('rec_soprano_c',   +12, 'Sopran-Blockflöte (C)',    'Soprano Recorder (C)'),
        ('rec_alto_f',        0, 'Alt-Blockflöte (F)',       'Alto Recorder (F)'),
        ('rec_tenor_c',       0, 'Tenor-Blockflöte (C)',     'Tenor Recorder (C)'),
        ('rec_bass_f',        0, 'Bass-Blockflöte (F)',      'Bass Recorder (F)'),
    ]),
    ('strings', 'Streicher', 'Strings', [
        ('violin',            0, 'Violine',             'Violin'),
        ('viola',             0, 'Viola',               'Viola'),
        ('cello',             0, 'Violoncello',         'Cello'),
        ('double_bass',     -12, 'Kontrabass',          'Double Bass'),
        ('mandolin',          0, 'Mandoline',           'Mandolin'),
    ]),
    ('plucked', 'Zupfinstrumente', 'Plucked', [
        ('guitar',            0, 'Gitarre',             'Guitar'),
        ('bass_guitar',     -12, 'Bassgitarre',         'Bass Guitar'),
        ('ukulele',           0, 'Ukulele',             'Ukulele'),
        ('banjo',             0, 'Banjo',               'Banjo'),
        ('harp',              0, 'Harfe',               'Harp'),
    ]),
    ('voice_other', 'Stimme / Konzertstimmung', 'Voice / Concert', [
        ('voice',             0, 'Stimme',              'Voice'),
        ('c',                 0, 'C-Instrument',        'C Instrument'),
        ('piano',             0, 'Klavier',             'Piano'),
    ]),
]


# Runtime-registered user instruments. Loaded once at startup from
# the ConfigStore's custom instruments file and refreshed when the user adds a Custom… entry.
_CUSTOMS: list[tuple[str, int, str, str]] = []   # (key, transp, name_de, name_en)


def register_custom(key: str, transp: int,
                    name_de: str, name_en: str) -> None:
    """Register or update a custom instrument. Duplicate keys replace."""
    global _CUSTOMS
    _CUSTOMS = [c for c in _CUSTOMS if c[0] != key]
    _CUSTOMS.append((key, transp, name_de, name_en))


_CUSTOM_FAMILY_KEY = 'custom'


def families() -> list[tuple[str, str, str]]:
    """Return [(family_key, name_de, name_en)] in display order.

    A 'custom' family is appended automatically when any user instrument
    is registered."""
    out = [(k, de, en) for (k, de, en, _) in _FAMILIES]
    if _CUSTOMS:
        out.append((_CUSTOM_FAMILY_KEY, 'Eigene', 'Custom'))
    return out


def instruments_in(family_key: str) -> list[tuple[str, str, str]]:
    """Return [(instrument_key, name_de, name_en)] for the given family."""
    if family_key == _CUSTOM_FAMILY_KEY:
        return [(k, de, en) for (k, _t, de, en) in _CUSTOMS]
    for (k, _de, _en, instrs) in _FAMILIES:
        if k == family_key:
            return [(ik, de, en) for (ik, _t, de, en) in instrs]
    return []


# transp_map() is read on every detected note, so it is built once per
# custom-instrument list. register_custom rebinds _CUSTOMS to a new list, which
# makes the identity of that list the cache key: registering (or a test
# swapping the list) invalidates the cache with no separate bookkeeping.
_transp_cache: tuple[list, Mapping[str, int]] | None = None


def transp_map() -> Mapping[str, int]:
    """Read-only {instrument_key: transp_semitones}, including customs.

    Always reflects the instruments registered so far; callers read it at use
    time rather than keeping a copy, so a custom instrument added mid-session
    is seen everywhere."""
    global _transp_cache
    cached = _transp_cache
    if cached is not None and cached[0] is _CUSTOMS:
        return cached[1]
    m: dict[str, int] = {}
    for (_, _de, _en, instrs) in _FAMILIES:
        for (key, transp, _d, _e) in instrs:
            m[key] = transp
    for (key, transp, _d, _e) in _CUSTOMS:
        m[key] = transp
    view = MappingProxyType(m)
    _transp_cache = (_CUSTOMS, view)
    return view


def display_name(instrument_key: str, lang: str = 'en') -> str:
    """Localised display name. Falls back to the instrument key if unknown."""
    for (key, _transp, de, en) in _CUSTOMS:
        if key == instrument_key:
            return de if lang == 'de' else en
    for (_fk, _fde, _fen, instrs) in _FAMILIES:
        for (key, _transp, de, en) in instrs:
            if key == instrument_key:
                return de if lang == 'de' else en
    return instrument_key


# Two baked ranges per instrument, both FINGERED MIDI (lo, hi) inclusive.
#
# _KEYED_RANGES: the notes the keywork or normal compass gives. It seeds the
# table's blank guide rows when an instrument is picked, and the matrix span.
# _EXTENDED_RANGES: what skilled players reach (altissimo, pedal and high
# registers). It decides whether a played note counts as in range: chart
# greying, the filter-to-range switch and the wrong-instrument hint. It always
# contains the keyed range.
#
# Both come from MuseScore's instruments.xml: keyed from `aPitchRange` (the
# handbook calls it an "early amateur" range), extended from `pPitchRange` (the
# professional range). Saxophones and clarinets take their keyed range from
# keywork and published compasses instead, and a few extended ranges carry
# labelled extensions. Every departure and its reason is recorded as data in
# tools/musescore/sync_ranges.py, which prints these two tables; a contract test
# requires them to match its output exactly, so edit the data there and paste.
_KEYED_RANGES: dict[str, tuple[int, int]] = {
    # Saxophone
    'eb_sopranino':           ( 58,  90),   # sopranino-saxophone, keyed exception
    'bb_soprano':             ( 58,  90),   # soprano-saxophone, keyed exception
    'eb_alto':                ( 58,  90),   # alto-saxophone, keyed exception
    'bb_tenor':               ( 58,  90),   # tenor-saxophone, keyed exception
    'eb_bari':                ( 57,  90),   # baritone-saxophone, keyed exception
    'bb_bass':                ( 58,  90),   # bass-saxophone, keyed exception
    'eb_contrabass':          ( 58,  90),   # contrabass-saxophone, keyed exception
    # Clarinet
    'clar_eb':                ( 52,  96),   # eb-clarinet, keyed exception
    'clar_d':                 ( 52,  96),   # d-clarinet, keyed exception
    'clar_c':                 ( 52,  96),   # c-clarinet, keyed exception
    'clar_bb':                ( 52,  96),   # bb-clarinet, keyed exception
    'clar_a':                 ( 52,  96),   # a-clarinet, keyed exception
    'clar_basset_f':          ( 48,  96),   # basset-horn, keyed exception
    'clar_alto_eb':           ( 52,  89),   # alto-clarinet
    'clar_bass_bb':           ( 48,  96),   # bb-bass-clarinet, keyed exception
    'clar_contraalto_eb':     ( 48,  91),   # contra-alto-clarinet, keyed exception
    'clar_contrabass_bb':     ( 48,  91),   # contrabass-clarinet, keyed exception
    # Flute
    'flute_piccolo':          ( 62,  93),   # piccolo
    'flute_c':                ( 60,  93),   # flute
    'flute_alto_g':           ( 60,  93),   # alto-flute
    'flute_bass_c':           ( 60,  89),   # bass-flute
    # Trumpet
    'trp_piccolo_bb':         ( 49,  76),   # bb-piccolo-trumpet
    'trp_piccolo_a':          ( 49,  76),   # a-piccolo-trumpet
    'trp_f':                  ( 60,  77),   # f-trumpet
    'trp_e':                  ( 54,  81),   # e-trumpet
    'trp_eb':                 ( 54,  81),   # eb-trumpet
    'trp_d':                  ( 54,  81),   # d-trumpet
    'trp_c':                  ( 54,  82),   # c-trumpet
    'trp_bb':                 ( 54,  82),   # bb-trumpet
    'trp_a':                  ( 54,  82),   # a-trumpet
    'trp_bass_bb':            ( 54,  81),   # bb-bass-trumpet
    'cornet_bb':              ( 54,  81),   # bb-cornet
    'flugel_bb':              ( 54,  81),   # flugelhorn
    # Horn
    'horn_f':                 ( 48,  76),   # horn
    'horn_bb':                ( 46,  79),   # bb-horn-alto
    'horn_eb_alto':           ( 54,  81),   # eb-alto-horn
    'mellophone_f':           ( 54,  81),   # mellophone
    # Trombone
    'tbn_alto_eb':            ( 45,  74),   # alto-trombone
    'tbn_tenor':              ( 40,  71),   # trombone
    'tbn_bass':               ( 32,  65),   # bass-trombone
    'tbn_contrabass':         ( 28,  62),   # contrabass-trombone
    # Low Brass
    'euph_bc':                ( 40,  70),   # euphonium
    'euph_tc':                ( 54,  84),   # euphonium-treble
    'baritone_bc':            ( 40,  67),   # baritone-horn
    'baritone_tc':            ( 54,  81),   # baritone-horn-treble
    'tuba_f':                 ( 26,  64),   # f-tuba
    'tuba_eb':                ( 26,  64),   # eb-tuba
    'tuba_cc':                ( 26,  60),   # c-tuba
    'tuba_bbb':               ( 28,  58),   # bb-tuba
    'sousaphone_bbb':         ( 30,  60),   # bb-sousaphone
    # Double Reed
    'oboe':                   ( 58,  87),   # oboe
    'oboe_damore':            ( 59,  87),   # oboe-d'amore
    'english_horn':           ( 59,  88),   # english-horn
    'bassoon':                ( 34,  69),   # bassoon
    'contrabassoon':          ( 34,  69),   # contrabassoon
    # Recorder
    'rec_sopranino_f':        ( 65,  88),   # sopranino-recorder
    'rec_soprano_c':          ( 60,  81),   # soprano-recorder
    'rec_alto_f':             ( 65,  88),   # alto-recorder
    'rec_tenor_c':            ( 60,  81),   # tenor-recorder
    'rec_bass_f':             ( 53,  74),   # bass-recorder
    # Strings
    'violin':                 ( 55,  88),   # violin
    'viola':                  ( 48,  79),   # viola
    'cello':                  ( 36,  67),   # violoncello
    'double_bass':            ( 40,  74),   # contrabass
    'mandolin':               ( 55,  85),   # mandolin
    # Plucked
    'guitar':                 ( 40,  83),   # guitar-steel
    'bass_guitar':            ( 40,  77),   # bass-guitar
    'ukulele':                ( 60,  81),   # ukulele
    'banjo':                  ( 48,  87),   # banjo
    'harp':                   ( 23, 104),   # harp
    # Voice / Concert
    'voice':                  ( 41,  79),   # voice
    'c':                      ( 48,  79),   # hand-set
    'piano':                  ( 21, 108),   # piano
}

_EXTENDED_RANGES: dict[str, tuple[int, int]] = {
    # Saxophone
    'eb_sopranino':           ( 58,  90),   # sopranino-saxophone, widened to keyed
    'bb_soprano':             ( 58, 101),   # soprano-saxophone, extension
    'eb_alto':                ( 58, 101),   # alto-saxophone
    'bb_tenor':               ( 58, 101),   # tenor-saxophone
    'eb_bari':                ( 57, 101),   # baritone-saxophone
    'bb_bass':                ( 58,  96),   # bass-saxophone
    'eb_contrabass':          ( 58,  90),   # contrabass-saxophone, widened to keyed
    # Clarinet
    'clar_eb':                ( 52,  96),   # eb-clarinet
    'clar_d':                 ( 52,  96),   # d-clarinet
    'clar_c':                 ( 52,  96),   # c-clarinet, widened to keyed
    'clar_bb':                ( 52,  96),   # bb-clarinet
    'clar_a':                 ( 52,  96),   # a-clarinet
    'clar_basset_f':          ( 48,  96),   # basset-horn, widened to keyed
    'clar_alto_eb':           ( 52,  93),   # alto-clarinet
    'clar_bass_bb':           ( 48,  96),   # bb-bass-clarinet
    'clar_contraalto_eb':     ( 48,  96),   # contra-alto-clarinet, widened to keyed
    'clar_contrabass_bb':     ( 48,  96),   # contrabass-clarinet
    # Flute
    'flute_piccolo':          ( 62,  96),   # piccolo
    'flute_c':                ( 59,  98),   # flute
    'flute_alto_g':           ( 60,  96),   # alto-flute
    'flute_bass_c':           ( 60,  93),   # bass-flute
    # Trumpet
    'trp_piccolo_bb':         ( 49,  79),   # bb-piccolo-trumpet
    'trp_piccolo_a':          ( 49,  79),   # a-piccolo-trumpet
    'trp_f':                  ( 54,  86),   # f-trumpet
    'trp_e':                  ( 54,  84),   # e-trumpet
    'trp_eb':                 ( 54,  84),   # eb-trumpet
    'trp_d':                  ( 54,  84),   # d-trumpet
    'trp_c':                  ( 54,  85),   # c-trumpet
    'trp_bb':                 ( 54,  87),   # bb-trumpet
    'trp_a':                  ( 54,  87),   # a-trumpet
    'trp_bass_bb':            ( 54,  84),   # bb-bass-trumpet
    'cornet_bb':              ( 54,  84),   # bb-cornet
    'flugel_bb':              ( 54,  84),   # flugelhorn
    # Horn
    'horn_f':                 ( 38,  84),   # horn
    'horn_bb':                ( 42,  84),   # bb-horn-alto
    'horn_eb_alto':           ( 54,  84),   # eb-alto-horn
    'mellophone_f':           ( 54,  84),   # mellophone
    # Trombone
    'tbn_alto_eb':            ( 45,  77),   # alto-trombone
    'tbn_tenor':              ( 36,  74),   # trombone
    'tbn_bass':               ( 21,  77),   # bass-trombone
    'tbn_contrabass':         ( 21,  72),   # contrabass-trombone, extension
    # Low Brass
    'euph_bc':                ( 40,  74),   # euphonium
    'euph_tc':                ( 54,  88),   # euphonium-treble
    'baritone_bc':            ( 40,  70),   # baritone-horn
    'baritone_tc':            ( 54,  84),   # baritone-horn-treble
    'tuba_f':                 ( 24,  72),   # f-tuba
    'tuba_eb':                ( 24,  72),   # eb-tuba
    'tuba_cc':                ( 24,  72),   # c-tuba
    'tuba_bbb':               ( 22,  72),   # bb-tuba
    'sousaphone_bbb':         ( 24,  74),   # bb-sousaphone
    # Double Reed
    'oboe':                   ( 58,  96),   # oboe
    'oboe_damore':            ( 59,  90),   # oboe-d'amore
    'english_horn':           ( 59,  90),   # english-horn
    'bassoon':                ( 34,  76),   # bassoon
    'contrabassoon':          ( 34,  72),   # contrabassoon
    # Recorder
    'rec_sopranino_f':        ( 65,  91),   # sopranino-recorder
    'rec_soprano_c':          ( 60,  91),   # soprano-recorder
    'rec_alto_f':             ( 64,  96),   # alto-recorder
    'rec_tenor_c':            ( 60,  91),   # tenor-recorder
    'rec_bass_f':             ( 53,  77),   # bass-recorder
    # Strings
    'violin':                 ( 55, 103),   # violin
    'viola':                  ( 48,  93),   # viola
    'cello':                  ( 36,  90),   # violoncello
    'double_bass':            ( 36,  79),   # contrabass
    'mandolin':               ( 55, 100),   # mandolin
    # Plucked
    'guitar':                 ( 40,  84),   # guitar-steel
    'bass_guitar':            ( 40,  79),   # bass-guitar
    'ukulele':                ( 60,  81),   # ukulele
    'banjo':                  ( 48,  87),   # banjo
    'harp':                   ( 23, 104),   # harp
    # Voice / Concert
    'voice':                  ( 38,  84),   # voice
    'c':                      ( 48,  79),   # hand-set
    'piano':                  ( 21, 108),   # piano
}

# A generic two-and-a-half-octave middle range for keys without baked data
# (user-defined custom instruments).
_FALLBACK_RANGE: tuple[int, int] = (48, 79)


def keyed_range(instrument_key: str) -> tuple[int, int]:
    """The KEYED tier (lo, hi), fingered MIDI: the guide rows and matrix span.

    A user override from the range editor replaces it; otherwise the baked
    value, or the generic fallback for keys without baked data."""
    ov = range_overrides().get(instrument_key)
    if ov is not None:
        return ov
    return baked_keyed_range(instrument_key)


def extended_range(instrument_key: str) -> tuple[int, int]:
    """The EXTENDED tier (lo, hi), fingered MIDI: whether a played note is in
    range.

    A user override widens it: the result spans both the override and the
    baked extended range, so editing the guide rows never makes notes a
    skilled player reaches count as wrong. Keys without baked data (custom
    instruments) have nothing to widen; there the override, or the fallback,
    is the whole range."""
    ov = range_overrides().get(instrument_key)
    baked = _EXTENDED_RANGES.get(instrument_key)
    if ov is None:
        return baked if baked is not None else _FALLBACK_RANGE
    if baked is None:
        return ov
    # One span from the lower start to the higher end, including any gap
    # between an override and a baked range that do not overlap.
    return (min(ov[0], baked[0]), max(ov[1], baked[1]))


def baked_keyed_range(instrument_key: str) -> tuple[int, int]:
    """The baked keyed range, ignoring user overrides. The range editor's
    Restore Default button restores this."""
    return _KEYED_RANGES.get(instrument_key, _FALLBACK_RANGE)


def baked_extended_range(instrument_key: str) -> tuple[int, int]:
    """The baked extended range, ignoring user overrides."""
    return _EXTENDED_RANGES.get(instrument_key, _FALLBACK_RANGE)


def fingered_range(instrument_key: str) -> tuple[int, int]:
    """The KEYED tier; the name predates the two tiers. New code should call
    keyed_range() or extended_range() so the choice of tier is visible."""
    return keyed_range(instrument_key)


def baked_fingered_range(instrument_key: str) -> tuple[int, int]:
    """The baked KEYED tier; the name predates the two tiers. New code should
    call baked_keyed_range()."""
    return baked_keyed_range(instrument_key)


def has_baked_range(instrument_key: str) -> bool:
    """Whether the instrument has baked ranges. Custom instruments without
    them get the generic fallback; the editor uses this to label the Restore
    button accordingly."""
    return instrument_key in _KEYED_RANGES


# ---------------------------------------------------------------------------
# Per-user range overrides
# ---------------------------------------------------------------------------
# Stored in instrument_ranges.json in the data directory. Read on first use
# (never at import, so importing the catalog touches no file) and refreshed
# in-process whenever save/clear is called, so the next keyed_range() or
# extended_range() consult reflects the change without restart.
_RANGES_SCHEMA_VERSION = 1


def _coerce_range_entry(v) -> tuple[int, int] | None:
    """Validate a (lo, hi) entry. Rejects anything not parseable, out of
    0..127, or with lo > hi. Returns None on any defect so the caller can
    skip silently."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    try:
        lo = int(v[0])
        hi = int(v[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0 <= lo <= 127 and 0 <= hi <= 127 and lo <= hi):
        return None
    return (lo, hi)


class RangeOverridesUnreadable(Exception):
    """instrument_ranges.json exists but could not be read or understood.
    Raised by the strict load so a caller never rewrites the file from an
    empty result and wipes every other instrument's override."""


class RangeOverrides:
    """The user's per-instrument range overrides in one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: dict[str, tuple[int, int]] | None = None

    def get(self, key: str) -> tuple[int, int] | None:
        if self._cache is None:
            self.load()
        return self._cache.get(key)

    def load(self, strict: bool = False) -> dict[str, tuple[int, int]]:
        """Read the file into a fresh dict and refresh the cache.

        Tolerant by default: missing file, malformed JSON, wrong shape and
        bad entries all degrade to {} or skip the offending key, because the
        first read happens while the window is being built and must never
        stop the app from starting.

        ``strict=True`` is for read-modify-write callers. A file that exists
        but cannot be read (OSError) or parsed (ValueError, wrong shape)
        raises RangeOverridesUnreadable and leaves the cache alone.
        Individual bad entries are still skipped; they carry nothing worth
        keeping."""
        out: dict[str, tuple[int, int]] = {}
        if not self.path.exists():
            self._cache = out
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            # ValueError covers JSONDecodeError and undecodable bytes.
            if strict:
                raise RangeOverridesUnreadable(str(exc)) from exc
            self._cache = out
            return {}
        ranges = data.get("ranges", {}) if isinstance(data, dict) else None
        if not isinstance(ranges, dict):
            if strict:
                raise RangeOverridesUnreadable(
                    f"{self.path} does not hold a ranges object")
            self._cache = out
            return {}
        for key, val in ranges.items():
            if not isinstance(key, str):
                continue
            coerced = _coerce_range_entry(val)
            if coerced is None:
                continue
            out[key] = coerced
        self._cache = out
        # A copy: read-modify-write callers edit it before the write, and
        # the cache must keep matching the file if that write fails.
        return dict(out)

    def _write(self, current: dict[str, tuple[int, int]]) -> None:
        """Write the full override table atomically (temp file + rename, so
        a crashed write never leaves a half-baked file) and make it the
        cache. Raises OSError when the write fails; the old file is left
        intact."""
        payload = {
            "version": _RANGES_SCHEMA_VERSION,
            "ranges": {k: list(v) for k, v in current.items()},
        }
        if not atomic_write_json(
                self.path, payload, tmp_prefix=".instrument_ranges."):
            raise OSError(f"could not write {self.path}")
        self._cache = dict(current)

    def save(self, key: str, lo: int, hi: int) -> None:
        """Persist one override entry, preserving every other key in the
        file.

        Raises ValueError for a range outside 0..127 or with lo > hi,
        RangeOverridesUnreadable when the existing file cannot be read (it
        is left untouched), and OSError when the write fails. On success the
        cache matches the file."""
        coerced = _coerce_range_entry((lo, hi))
        if coerced is None:
            raise ValueError(f"invalid range {lo}..{hi} for {key}")
        # Re-read from disk so a parallel process's edits are kept.
        current = self.load(strict=True)
        current[key] = coerced
        self._write(current)

    def clear(self, key: str) -> None:
        """Remove one override entry, if present. Same error contract as
        save()."""
        current = self.load(strict=True)
        if key not in current:
            return
        del current[key]
        self._write(current)


# The session's overrides. The window installs one for its data directory at
# start-up; until then the first use resolves the current HOME.
_active_overrides: RangeOverrides | None = None


def range_overrides() -> RangeOverrides:
    global _active_overrides
    if _active_overrides is None:
        _active_overrides = RangeOverrides(AppPaths.from_environment().ranges)
    return _active_overrides


def use_range_overrides(store: RangeOverrides | None) -> None:
    global _active_overrides
    _active_overrides = store


def load_range_overrides(strict: bool = False) -> dict[str, tuple[int, int]]:
    return range_overrides().load(strict)


def save_range_override(key: str, lo: int, hi: int) -> None:
    range_overrides().save(key, lo, hi)


def clear_range_override(key: str) -> None:
    range_overrides().clear(key)


def __getattr__(name: str):
    # The overrides file used to be a module constant resolved at import;
    # it now follows the active store.
    if name == "_RANGES_FILE":
        return range_overrides().path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def family_of(instrument_key: str) -> str | None:
    """Which family contains this instrument key, or None if unknown."""
    for (key, _t, _d, _e) in _CUSTOMS:
        if key == instrument_key:
            return _CUSTOM_FAMILY_KEY
    for (fk, _de, _en, instrs) in _FAMILIES:
        for (key, _t, _d, _e) in instrs:
            if key == instrument_key:
                return fk
    return None
