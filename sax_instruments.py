"""
Instrument catalog: families, sub-instruments, transposition values, and
display names in DE + EN.

Transposition convention matches the original app: `transp = sounding_midi -
fingered_midi`. For most instruments this is the smallest interval that
fits — for instance Bb instruments are -2 (major 2nd below) regardless of
octave register. Bass-register instruments get the octave added so the
displayed sounding pitch reads in the right octave.

The original six saxophone/C-instrument keys (`eb_alto`, `eb_bari`,
`bb_tenor`, `bb_soprano`, `bb_bass`, `c`) are preserved with identical
transposition values so old CSV exports round-trip cleanly.
"""

from __future__ import annotations

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
        ('horn_eb_alto',     +3, 'Eb-Althorn',          'Eb Alto Horn'),
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
        ('rec_sopranino_f',  +5, 'Sopranino-Blockflöte (F)', 'Sopranino Recorder (F)'),
        ('rec_soprano_c',   +12, 'Sopran-Blockflöte (C)',    'Soprano Recorder (C)'),
        ('rec_alto_f',        0, 'Alt-Blockflöte (F)',       'Alto Recorder (F)'),
        ('rec_tenor_c',       0, 'Tenor-Blockflöte (C)',     'Tenor Recorder (C)'),
        ('rec_bass_f',        0, 'Bass-Blockflöte (F)',      'Bass Recorder (F)'),
    ]),
    ('strings', 'Streicher', 'Strings', [
        ('violin',            0, 'Violine',             'Violin'),
        ('viola',             0, 'Viola',               'Viola'),
        ('cello',             0, 'Violoncello',         'Cello'),
        ('double_bass',       0, 'Kontrabass',          'Double Bass'),
        ('mandolin',          0, 'Mandoline',           'Mandolin'),
    ]),
    ('plucked', 'Zupfinstrumente', 'Plucked', [
        ('guitar',            0, 'Gitarre',             'Guitar'),
        ('bass_guitar',       0, 'Bassgitarre',         'Bass Guitar'),
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
# sax_config.load_customs() and refreshed when the user adds a Custom… entry.
_CUSTOMS: list[tuple[str, int, str, str]] = []   # (key, transp, name_de, name_en)


def register_custom(key: str, transp: int,
                    name_de: str, name_en: str) -> None:
    """Register or update a custom instrument. Duplicate keys replace."""
    global _CUSTOMS
    _CUSTOMS = [c for c in _CUSTOMS if c[0] != key]
    _CUSTOMS.append((key, transp, name_de, name_en))


def clear_customs() -> None:
    """Remove all custom instruments (used by tests)."""
    global _CUSTOMS
    _CUSTOMS = []


def has_customs() -> bool:
    return bool(_CUSTOMS)


_CUSTOM_FAMILY_KEY = 'custom'


def families() -> list[tuple[str, str, str]]:
    """Return [(family_key, name_de, name_en)] in display order.

    A 'custom' family is appended automatically when any user instrument
    is registered."""
    out = [(k, de, en) for (k, de, en, _) in _FAMILIES]
    if _CUSTOMS:
        out.append((_CUSTOM_FAMILY_KEY, 'Eigene', 'Custom'))
    return out


def family_display_name(family_key: str, lang: str = 'en') -> str:
    for (k, de, en) in families():
        if k == family_key:
            return de if lang == 'de' else en
    return family_key


def instruments_in(family_key: str) -> list[tuple[str, str, str]]:
    """Return [(instrument_key, name_de, name_en)] for the given family."""
    if family_key == _CUSTOM_FAMILY_KEY:
        return [(k, de, en) for (k, _t, de, en) in _CUSTOMS]
    for (k, _de, _en, instrs) in _FAMILIES:
        if k == family_key:
            return [(ik, de, en) for (ik, _t, de, en) in instrs]
    return []


def transp_map() -> dict[str, int]:
    """Flat dict of {instrument_key: transp_semitones}, including customs."""
    m: dict[str, int] = {}
    for (_, _de, _en, instrs) in _FAMILIES:
        for (key, transp, _d, _e) in instrs:
            m[key] = transp
    for (key, transp, _d, _e) in _CUSTOMS:
        m[key] = transp
    return m


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


# Typical fingered MIDI range per instrument: (lo, hi) inclusive. Used to
# pre-seed the intonation table with blank rows when the user selects an
# instrument, so they immediately see what's "expected" for that horn. Notes
# played outside this range (overtones, altissimo, extensions) still pop in
# automatically through the normal _on_note path.
# Ranges sourced from MuseScore's instruments.xml (their `aPitchRange`
# — amateur range — converted from sounding to FINGERED via
# `transposeChromatic`). MuseScore is an open-source notation app with
# ~10M users; the ranges are vetted by orchestrators and notation
# experts and are about as close to a single source of truth as we can
# get programmatically. Real-player overrides (saxes at low A, contras
# at low C) layered on top — see tools/musescore/sync_ranges.py for
# the reproducible regeneration script.
_RANGES: dict[str, tuple[int, int]] = {
    # Saxophones — fingered low A (57) to altissimo C7 (96). Generous on
    # the top end so altissimo work isn't clipped; the filter-to-range
    # toggle is off by default, so this is mostly a display guide for the
    # grid layout. When players disagree, we include more.
    'eb_sopranino':           ( 57,  96),
    'bb_soprano':             ( 57,  96),
    'eb_alto':                ( 57,  96),
    'bb_tenor':               ( 57,  96),
    'eb_bari':                ( 57,  96),
    'bb_bass':                ( 57,  96),
    'eb_contrabass':          ( 57,  96),
    # Clarinets — top widened to cover altissimo where players disagree.
    'clar_eb':                ( 52,  96),   # MS via eb-clarinet + altissimo
    'clar_d':                 ( 52,  96),   # MS via d-clarinet + altissimo
    'clar_c':                 ( 52,  96),   # MS via c-clarinet + altissimo
    'clar_bb':                ( 52,  96),   # MS via bb-clarinet + altissimo
    'clar_a':                 ( 52,  96),   # MS via a-clarinet + altissimo
    'clar_basset_f':          ( 48,  96),   # MS via basset-horn + altissimo
    'clar_alto_eb':           ( 52,  91),   # MS via alto-clarinet
    'clar_bass_bb':           ( 48,  96),   # low-C extension; altissimo widened
    'clar_contraalto_eb':     ( 48,  91),   # override-lo + altissimo
    'clar_contrabass_bb':     ( 48,  91),   # override-lo + altissimo
    # Flutes
    'flute_piccolo':          ( 62,  93),   # MS via piccolo
    'flute_c':                ( 60,  93),   # MS via flute
    'flute_alto_g':           ( 60,  93),   # MS via alto-flute
    'flute_bass_c':           ( 60,  89),   # MS via bass-flute
    # Trumpets
    'trp_piccolo_bb':         ( 49,  76),   # MS via bb-piccolo-trumpet
    'trp_piccolo_a':          ( 49,  76),   # MS via a-piccolo-trumpet
    'trp_f':                  ( 60,  77),   # MS via f-trumpet
    'trp_e':                  ( 54,  81),   # MS via e-trumpet
    'trp_eb':                 ( 54,  81),   # MS via eb-trumpet
    'trp_d':                  ( 54,  81),   # MS via d-trumpet
    'trp_c':                  ( 54,  82),   # MS via c-trumpet
    'trp_bb':                 ( 54,  82),   # MS via bb-trumpet
    'trp_a':                  ( 54,  82),   # MS via a-trumpet
    'trp_bass_bb':            ( 54,  81),   # MS via bb-bass-trumpet
    'cornet_bb':              ( 54,  81),   # MS via bb-cornet
    'flugel_bb':              ( 54,  81),   # MS via flugelhorn
    # Horns
    'horn_f':                 ( 48,  76),   # MS via horn
    'horn_bb':                ( 46,  79),   # MS via bb-horn-alto
    'horn_eb_alto':           ( 54,  81),   # MS via eb-alto-horn
    'mellophone_f':           ( 54,  79),   # manual (no MuseScore entry)
    # Trombones
    'tbn_alto_eb':            ( 45,  74),   # MS via alto-trombone
    'tbn_tenor':              ( 40,  71),   # MS via trombone
    'tbn_bass':               ( 32,  65),   # MS via bass-trombone
    'tbn_contrabass':         ( 28,  62),   # MS via contrabass-trombone
    # Low brass
    'euph_bc':                ( 40,  70),   # MS via euphonium
    'euph_tc':                ( 54,  84),   # MS via euphonium-treble
    'baritone_bc':            ( 43,  64),   # MS via baritone
    'baritone_tc':            ( 54,  81),   # MS via baritone-horn-treble
    'tuba_f':                 ( 26,  64),   # MS via f-tuba
    'tuba_eb':                ( 26,  64),   # MS via eb-tuba
    'tuba_cc':                ( 26,  60),   # MS via c-tuba
    'tuba_bbb':               ( 28,  58),   # MS via bb-tuba
    'sousaphone_bbb':         ( 44,  74),   # MS via bb-sousaphone
    # Double reeds
    'oboe':                   ( 58,  87),   # MS via oboe
    'oboe_damore':            ( 59,  87),   # MS via oboe-d'amore
    'english_horn':           ( 59,  88),   # MS via english-horn
    'bassoon':                ( 34,  69),   # MS via bassoon
    'contrabassoon':          ( 34,  69),   # MS via contrabassoon
    # Recorders
    'rec_sopranino_f':        ( 77, 100),   # MS via sopranino-recorder
    'rec_soprano_c':          ( 72,  93),   # MS via soprano-recorder
    'rec_alto_f':             ( 65,  88),   # MS via alto-recorder
    'rec_tenor_c':            ( 60,  81),   # MS via tenor-recorder
    'rec_bass_f':             ( 53,  74),   # MS via bass-recorder
    # Strings
    'violin':                 ( 55,  88),   # MS via violin
    'viola':                  ( 48,  79),   # MS via viola
    'cello':                  ( 36,  67),   # MS via violoncello
    'double_bass':            ( 40,  74),   # MS via contrabass
    'mandolin':               ( 55,  85),   # MS via mandolin
    # Plucked
    'guitar':                 ( 40,  83),   # MS via guitar-steel
    'bass_guitar':            ( 40,  77),   # MS via bass-guitar
    'ukulele':                ( 60,  81),   # MS via ukulele
    'banjo':                  ( 48,  87),   # MS via banjo
    'harp':                   ( 23, 104),   # MS via harp
    # Concert / generic
    'voice':                  ( 41,  79),   # MS via voice
    'c':                      ( 48,  79),   # manual (generic concert-pitch fallback)
    'piano':                  ( 21, 108),   # MS via piano
}


def fingered_range(instrument_key: str) -> tuple[int, int]:
    """Return (lo, hi) inclusive fingered-MIDI range for this instrument.

    Falls back to a generic 2.5-octave middle range for keys without an
    explicit entry (mainly user-defined custom instruments)."""
    return _RANGES.get(instrument_key, (48, 79))


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
