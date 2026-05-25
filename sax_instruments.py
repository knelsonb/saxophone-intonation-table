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
        ('eb_sopranino',    +3, 'Eb-Sax · Sopranino', 'Eb Sax · Sopranino'),
        ('bb_soprano',      -2, 'Bb-Sax · Sopran',    'Bb Sax · Soprano'),
        ('eb_alto',         +3, 'Eb-Sax · Alt',       'Eb Sax · Alto'),
        ('bb_tenor',        -2, 'Bb-Sax · Tenor',     'Bb Sax · Tenor'),
        ('eb_bari',         -9, 'Eb-Sax · Bariton',   'Eb Sax · Baritone'),
        ('bb_bass',        -14, 'Bb-Sax · Bass',      'Bb Sax · Bass'),
        ('eb_contrabass',  -21, 'Eb-Sax · Kontrabass', 'Eb Sax · Contrabass'),
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
# Ranges reconciled across three Haiku validators. Where the consensus
# narrowed the range we kept it conservative; where common keyed extensions
# pushed it wider on a 2-of-3 vote we adopted the wider value. Goal: cover
# what most players reach most of the time, including standard extensions
# like the high-F# key on tenor sax, the low-A extension on baritone sax,
# and the F-attachment on bass trombone — but stopping short of altissimo,
# pedal tones, and other virtuoso-only registers.
_RANGES: dict[str, tuple[int, int]] = {
    # Saxophones — fingered Bb3 (58) to F#6 (78) covers the standard
    # low-Bb / high-F# range. Eb baritone gets the low-A extension that
    # most pro horns carry.
    'eb_sopranino':       (58, 78),
    'bb_soprano':         (58, 78),
    'eb_alto':            (58, 78),
    'bb_tenor':           (58, 78),
    'eb_bari':            (57, 78),   # low-A extension on pro baris
    'bb_bass':            (58, 78),
    'eb_contrabass':      (58, 78),
    # Clarinets — E3 (52) up to high C6 (84) on standard pedagogical
    # Boehm horns; bass clarinet often carries the low-C extension on
    # pro models so the floor moves from D2 down to C2 (38 → 36).
    'clar_bb':            (52, 84),
    'clar_a':             (52, 84),
    'clar_c':             (52, 84),
    'clar_eb':            (52, 80),
    'clar_d':             (52, 80),
    'clar_basset_f':      (48, 79),
    'clar_alto_eb':       (50, 78),
    'clar_bass_bb':       (36, 74),   # low-C extension on pro bass clarinets
    'clar_contraalto_eb': (38, 74),
    'clar_contrabass_bb': (34, 70),
    # Flutes — B3 (59) to high D7 (86) on the concert flute is reachable
    # by most intermediate players with the standard B-foot.
    'flute_c':            (59, 86),
    'flute_piccolo':      (62, 84),
    'flute_alto_g':       (55, 79),
    'flute_bass_c':       (48, 72),
    # Trumpets — F#3 (54) to high C6 (84) is the universal Bb/C/D/Eb range.
    # Piccolo trumpets shift up to C4 (60). Bass trumpet drops to A#2 (46).
    'trp_bb':             (54, 84),
    'trp_c':              (54, 84),
    'trp_d':              (54, 84),
    'trp_eb':             (54, 84),
    'trp_e':              (54, 84),
    'trp_f':              (54, 84),
    'trp_a':              (54, 84),
    'trp_piccolo_bb':     (60, 84),
    'trp_piccolo_a':      (60, 84),
    'trp_bass_bb':        (46, 72),
    'cornet_bb':          (54, 84),
    'flugel_bb':          (54, 82),
    # Horns
    'horn_f':             (43, 77),
    'horn_bb':            (43, 77),
    'horn_eb_alto':       (50, 78),
    'mellophone_f':       (54, 79),
    # Trombones — tenor reaches D5 (74) easily for most players; bass
    # trombone with F+Gb attachments fills the gap down to Bb1 (34) and
    # comfortably reaches Bb4 (70).
    'tbn_alto_eb':        (50, 79),
    'tbn_tenor':          (40, 74),
    'tbn_bass':           (34, 70),
    'tbn_contrabass':     (28, 60),
    # Low brass — baritone horn often plays higher than euphonium on the
    # treble-clef parts in British brass-band literature.
    'euph_bc':            (40, 72),
    'euph_tc':            (40, 72),
    'baritone_bc':        (40, 74),
    'baritone_tc':        (40, 74),
    'tuba_f':             (36, 67),
    'tuba_eb':            (34, 65),
    'tuba_cc':            (28, 60),
    'tuba_bbb':           (26, 58),
    'sousaphone_bbb':     (26, 58),
    # Double reeds — bassoon with the standard low-Bb extension and pro
    # players reaching high F (77).
    'oboe':               (58, 84),
    'oboe_damore':        (55, 81),
    'english_horn':       (52, 79),
    'bassoon':            (34, 74),
    'contrabassoon':      (22, 56),
    # Recorders
    'rec_sopranino_f':    (65, 84),
    'rec_soprano_c':      (60, 82),
    'rec_alto_f':         (53, 77),
    'rec_tenor_c':        (48, 72),
    'rec_bass_f':         (41, 65),
    # Strings — cello pros reach E5/F5 (76); 5-string double bass extends
    # up to G4 (67) on solo literature, common 4-string pros reach G4.
    'violin':             (55, 91),
    'viola':              (48, 84),
    'cello':              (36, 76),
    'double_bass':        (28, 65),
    'mandolin':           (55, 84),
    # Plucked
    'guitar':             (40, 76),
    'bass_guitar':        (28, 60),
    'ukulele':            (55, 79),
    'banjo':              (43, 76),
    'harp':               (24, 96),
    # Concert / generic
    'voice':              (48, 79),
    'c':                  (48, 79),
    'piano':              (21, 108),
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
