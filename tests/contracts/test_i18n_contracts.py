"""i18n contracts: placeholders, used keys and dynamic key families.

test_i18n.py checks that both languages carry the same keys with non-blank text.
This file adds the checks that catch the failures a user would actually see:

  * a translation that drops or renames a ``{placeholder}`` (KeyError or raw
    braces on screen, in one language only),
  * a call site asking for a key that does not exist (shows ``[key]``),
  * a call site that does not pass the arguments its string needs,
  * a key nothing uses any more,
  * a dynamic family (``nav_tab_{key}``, ``csv_mode_{key}``, ``theme_{name}``,
    ``instr_long_{key}``, ``drone_voice_{id}``) missing a member for something
    the code iterates.

The call sites are found by parsing the application sources (AST), so a new
``_t('...')`` literal is checked without anyone editing this file. Besides
``_t``, the scan reads the Translator's own entry points
(intonation.i18n.translator):
``x.t(key)``, ``x.msg(key)``, ``x.bind(setter, key)``,
``x.bind_item(combo, index, key)``, ``x.bind_items(combo, [(key, data), ...])``
and ``x.bind_tab(tabs, index, key, tip_key)``, each with the keyword
arguments it formats the string with.
"""
from __future__ import annotations

import ast
import string
from pathlib import Path

import pytest

from intonation.audio.sources.drone import DRONE_PRESETS, DRONE_SYNTH_VOICES
from intonation.catalog import instruments as catalog
from intonation.i18n import strings as i18n
from intonation.persistence.history import MeasurementLog
from intonation.ui import theme

REPO = Path(__file__).resolve().parents[2]
DE = i18n.STRINGS['de']
EN = i18n.STRINGS['en']
LANGS = ('de', 'en')

# Dynamic families the code builds with an f-string: (prefix, suffix).
EXPECTED_FAMILIES = {
    ('nav_tab_', ''),
    ('nav_tab_', '_tip'),
    ('csv_mode_', ''),
    ('theme_', ''),
    ('instr_long_', ''),
    ('drone_voice_', ''),
}

# Strings that are the same in German and English on purpose: loanwords,
# abbreviations, units and symbols. Anything else identical is an untranslated
# string and fails.
IDENTICAL_DE_EN = {
    'grp_instrument': 'loanword',
    'setup_instrument_group': 'loanword',
    'csv_instr_label': 'loanword',
    'layout_auto': 'loanword',
    'reset_title': 'loanword',
    'export_title': 'loanword',
    'chart_subtitle': 'placeholders and units only',
    'chart_subtitle_id': 'placeholders and units only',
    'txt_instr': 'placeholder after a loanword label',
    'col_std': 'sigma (ct) symbol',
    'txt_col_std': 'sigma (ct) symbol',
    'pdf_col_std': 'sigma (ct) symbol',
    'txt_col_mean': 'mean symbol (ct)',
    'col_n': 'symbol N',
    'pdf_col_n': 'symbol N',
    'audio_picker_apis_more': 'count and the acronym',
    'data_block_hop': 'technical terms',
    'metro_bpm': 'unit',
    'metro_start': 'loanword',
    'metro_nudge_down_tip': 'unit',
    'metro_nudge_up_tip': 'unit',
    'drone_voice_cello': 'loanword',
}


# ---------------------------------------------------------------------------
# Source scan
# ---------------------------------------------------------------------------
def _source_files() -> list[Path]:
    files = [p for p in REPO.glob('*.py')
             if not p.name.startswith('test_')
             and p.name not in ('conftest.py', 'sim_harness.py')]
    files += [p for p in (REPO / 'intonation').rglob('*.py')
              if p != REPO / 'intonation' / 'i18n' / 'strings.py']
    files += sorted((REPO / 'scripts').glob('*.py'))
    files += sorted((REPO / 'tools').glob('*.py'))
    return sorted(files)


def _strings_in(node: ast.AST | None) -> list[str]:
    """String constants a key expression can evaluate to: a literal, either
    branch of a conditional, or the members of a list or tuple (the comparison
    in a conditional's test is skipped)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _strings_in(node.body) + _strings_in(node.orelse)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [s for elt in node.elts for s in _strings_in(elt)]
    return []


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


# Names that hold lists or conditionals of keys which are passed to _t later.
_KEY_CARRIERS = {'mode_note_key', '_row_keys'}

# Translation calls: call name -> positions of the key arguments. ``_t`` and
# ``t`` may be a plain name (a local alias such as ``t = tr.t``) or an
# attribute; the Translator's other methods are always called on an object
# (``tr.bind(...)``), which keeps an unrelated function that happens to be
# called ``bind`` out of the scan. ``show_status`` is the TUNER tab's status
# line, which binds its key and arguments the same way.
_KEY_ARGS = {'_t': (0,), 't': (0,)}
_TRANSLATOR_KEY_ARGS = {'msg': (0,), 'bind': (1,), 'bind_item': (2,),
                        'bind_tab': (2, 3), 'show_status': (0,)}
# Helpers that take keys and bind or show them later (some append the reason
# an output could not open): call name -> positions of the key arguments.
_KEY_TAKING_HELPERS = {'_deck_show_status': (0,), '_show_status': (0,),
                       'show_with_reason': (2,), '_titled_group': (0,),
                       '_action_button': (0,), '_ask_save_path': (0, 2)}


def _key_items(node: ast.AST | None) -> list[str]:
    """Keys in a ``bind_items`` list literal: the first element of each
    ``(key, data)`` pair. A computed list (a comprehension over an f-string
    family) yields nothing here; the family check covers it."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    keys: list[str] = []
    for elt in node.elts:
        if isinstance(elt, (ast.Tuple, ast.List)) and elt.elts:
            keys += _strings_in(elt.elts[0])
    return keys


class SourceScan:
    def __init__(self) -> None:
        self.t_calls: list[tuple[str, int, str, set[str], bool]] = []
        # Call name -> number of keys found through it (scan self-check).
        self.by_call: dict[str, int] = {}
        self.carried: dict[str, str] = {}      # key -> where it was found
        self.families: set[tuple[str, str]] = set()
        self.constants: set[str] = set()
        self.tab_keys: list[str] = []
        for path in _source_files():
            self._scan(path)

    def _scan(self, path: Path) -> None:
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.constants.add(node.value)
            elif isinstance(node, ast.Call):
                self._scan_call(rel, node)
            elif isinstance(node, ast.Assign):
                self._scan_assign(rel, node)
            elif isinstance(node, ast.JoinedStr):
                self._scan_fstring(node)

    def _scan_call(self, rel: str, call: ast.Call) -> None:
        name = _call_name(call)
        on_object = isinstance(call.func, ast.Attribute)
        positions = _KEY_ARGS.get(name) or (
            _TRANSLATOR_KEY_ARGS.get(name) if on_object else None)
        if positions:
            kwargs = {k.arg for k in call.keywords if k.arg}
            splat = any(k.arg is None for k in call.keywords)
            for pos in positions:
                if pos < len(call.args):
                    for key in _strings_in(call.args[pos]):
                        self.t_calls.append((rel, call.lineno, key, kwargs, splat))
                        self.by_call[name] = self.by_call.get(name, 0) + 1
        elif name == 'bind_items' and on_object and len(call.args) >= 2:
            for key in _key_items(call.args[1]):
                # The items are plain strings: no arguments to format.
                self.t_calls.append((rel, call.lineno, key, set(), False))
                self.by_call[name] = self.by_call.get(name, 0) + 1
        elif name in _KEY_TAKING_HELPERS:
            for pos in _KEY_TAKING_HELPERS[name]:
                if pos < len(call.args):
                    for key in _strings_in(call.args[pos]):
                        self.carried[key] = f'{rel}:{call.lineno}'
        elif (name == 'setProperty' and len(call.args) == 2
              and _strings_in(call.args[0]) == ['_lang_title_key']):
            for key in _strings_in(call.args[1]):
                self.carried[key] = f'{rel}:{call.lineno}'

    def _scan_assign(self, rel: str, node: ast.Assign) -> None:
        for target in node.targets:
            ident = (target.id if isinstance(target, ast.Name)
                     else target.attr if isinstance(target, ast.Attribute)
                     else None)
            if ident in _KEY_CARRIERS:
                for key in _strings_in(node.value):
                    self.carried[key] = f'{rel}:{node.lineno}'
            elif (ident == 'key' and isinstance(node.value, ast.Subscript)
                  and isinstance(node.value.value, ast.Dict)):
                for value in node.value.value.values:
                    for key in _strings_in(value):
                        self.carried[key] = f'{rel}:{node.lineno}'
            elif (ident in ('_tab_keys', 'TAB_KEYS')
                  and isinstance(node.value, (ast.List, ast.Tuple))):
                self.tab_keys = _strings_in(node.value)

    def _scan_fstring(self, node: ast.JoinedStr) -> None:
        values = node.values
        holes = [v for v in values if isinstance(v, ast.FormattedValue)]
        if (len(holes) == 1 and isinstance(values[0], ast.Constant)
                and values[0].value.endswith('_')):
            suffix = ''
            if len(values) == 3 and isinstance(values[2], ast.Constant):
                suffix = values[2].value
            self.families.add((values[0].value, suffix))


# Parsed once at import: the scan is read-only and shared by every test below.
SCAN = SourceScan()


@pytest.fixture(scope='module')
def scan() -> SourceScan:
    return SCAN


def _fields(text: str) -> list[tuple[str, str, str | None]]:
    """(field name, format spec, conversion) for each placeholder, sorted."""
    return sorted(
        (name, spec or '', conv)
        for _lit, name, spec, conv in string.Formatter().parse(text)
        if name is not None)


def _dynamic_keys() -> set[str]:
    keys: set[str] = set()
    tabs = SCAN.tab_keys
    keys |= {f'nav_tab_{k}' for k in tabs} | {f'nav_tab_{k}_tip' for k in tabs}
    keys |= {f'csv_mode_{m}' for m in MeasurementLog.SLICE_MODES}
    keys |= {f'theme_{t}' for t in theme.THEME_ORDER}
    keys |= {k for k in EN if k.startswith('instr_long_')}
    keys |= {f'drone_voice_{vid}' for vid, _n, _p in DRONE_PRESETS + DRONE_SYNTH_VOICES}
    return keys


# ---------------------------------------------------------------------------
# Placeholder parity and formatting
# ---------------------------------------------------------------------------
def test_languages_have_identical_key_sets() -> None:
    assert set(DE) == set(EN)


@pytest.mark.parametrize('key', sorted(EN))
def test_placeholders_match_between_languages(key: str) -> None:
    """Field names, format specs and conversions agree, so one call site can
    serve both languages."""
    assert _fields(DE[key]) == _fields(EN[key])


@pytest.mark.parametrize('lang', LANGS)
def test_every_string_is_valid_format_syntax(lang: str) -> None:
    for key, text in i18n.STRINGS[lang].items():
        try:
            list(string.Formatter().parse(text))
        except ValueError as exc:
            pytest.fail(f'{lang}.{key}: {exc}')


def _sample_arguments() -> dict[str, object]:
    """One sample value per placeholder name, typed by the format specs the
    strings apply to it (a name used with ``.1f`` gets a float, ``+d`` an int)."""
    specs: dict[str, set[str]] = {}
    for table in (DE, EN):
        for text in table.values():
            for name, spec, _conv in _fields(text):
                specs.setdefault(name, set()).add(spec)
    args: dict[str, object] = {}
    for name, used in specs.items():
        floats = any(s.endswith('f') for s in used)
        ints = any(s.endswith('d') for s in used)
        assert not (floats and ints), f'{name!r} is used with both f and d formats'
        args[name] = 1234.5 if floats else 7 if ints else 'sample'
    return args


@pytest.mark.parametrize('lang', LANGS)
def test_every_string_formats_with_sample_arguments(lang: str) -> None:
    args = _sample_arguments()
    for key, text in i18n.STRINGS[lang].items():
        fields = _fields(text)
        if not fields:
            continue
        out = text.format(**args)
        assert out.strip(), f'{lang}.{key} formats to nothing'
        assert '{' not in out and '}' not in out, f'{lang}.{key}: {out!r}'


@pytest.mark.parametrize('lang', LANGS)
def test_strings_without_placeholders_have_no_doubled_braces(lang: str) -> None:
    """The window only formats a string when arguments are passed, so a doubled
    brace in a field-free string would be shown as typed."""
    for key, text in i18n.STRINGS[lang].items():
        if not _fields(text):
            assert '{{' not in text and '}}' not in text, f'{lang}.{key}'


def test_identical_german_and_english_strings_are_allow_listed() -> None:
    identical = {k for k in EN if DE[k] == EN[k]}
    assert identical == set(IDENTICAL_DE_EN), (
        f'untranslated: {sorted(identical - set(IDENTICAL_DE_EN))}; '
        f'stale allow-list entries: {sorted(set(IDENTICAL_DE_EN) - identical)}')


# ---------------------------------------------------------------------------
# Keys used by the code
# ---------------------------------------------------------------------------
def test_scan_finds_the_call_sites(scan: SourceScan) -> None:
    """Guards the AST scan against going vacuous after a refactor: every key
    in the catalogue is reached through a translation call, a key-taking
    helper or a dynamic family the scan knows. A key the code reaches some
    other way would escape the existence and argument checks below; teach
    the scan the new pattern instead."""
    reached = ({c[2] for c in scan.t_calls} | set(scan.carried) | _dynamic_keys())
    assert sorted(set(EN) - reached) == []
    assert {'intonation/ui/main_window.py', 'intonation/export/controller.py',
            'intonation/crash.py'} <= {c[0] for c in scan.t_calls}
    for key in ('deck_status_empty', 'range_mode_note_griff',
                'data_device', 'welcome_title', 'deck_unavailable',
                'metro_unavailable'):
        assert key in scan.carried, f'carrier pattern for {key} not found'
    # Each Translator entry point is seen, with a key known to go through it.
    asked = {c[2] for c in scan.t_calls}
    for key in ('setup_output_group', 'filter_fast', 'setup_output_default',
                'audio_banner_disconnect', 'audio_toast_switch'):
        assert key in asked, f'translator pattern for {key} not found'
    for call in ('_t', 't', 'msg', 'bind', 'bind_item', 'bind_items'):
        assert scan.by_call.get(call, 0) > 0, f'no keys found through {call}()'
    assert scan.tab_keys == ['tuner', 'metro', 'deck', 'setup']


def test_every_key_the_code_asks_for_exists_in_both_languages(scan: SourceScan) -> None:
    asked = {c[2] for c in scan.t_calls} | set(scan.carried)
    for lang in LANGS:
        missing = sorted(k for k in asked if k not in i18n.STRINGS[lang])
        assert not missing, f'{lang} lacks keys the code uses: {missing}'


def test_call_sites_pass_exactly_the_arguments_the_string_needs(scan: SourceScan) -> None:
    problems = []
    for rel, line, key, kwargs, splat in scan.t_calls:
        if key not in EN or splat:
            continue
        needed = {name for name, _s, _c in _fields(EN[key])}
        if needed != kwargs:
            problems.append(
                f'{rel}:{line} {key}: string needs {sorted(needed)}, '
                f'call passes {sorted(kwargs)}')
    assert not problems, '\n'.join(problems)


def test_only_the_expected_dynamic_families_are_built(scan: SourceScan) -> None:
    """A new f-string key family needs its members checked below."""
    assert scan.families == EXPECTED_FAMILIES


def test_no_key_is_unused(scan: SourceScan) -> None:
    referenced = (scan.constants | {c[2] for c in scan.t_calls}
                  | set(scan.carried) | _dynamic_keys())
    unused = sorted(k for k in EN if k not in referenced)
    assert not unused, f'keys nothing uses: {unused}'


# ---------------------------------------------------------------------------
# Dynamic families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('lang', LANGS)
def test_nav_tab_family_is_complete(lang: str) -> None:
    table = i18n.STRINGS[lang]
    for tab in SCAN.tab_keys:
        assert table.get(f'nav_tab_{tab}', '').strip(), tab
        assert table.get(f'nav_tab_{tab}_tip', '').strip(), tab
    declared = {k for k in table if k.startswith('nav_tab_')}
    expected = ({f'nav_tab_{t}' for t in SCAN.tab_keys}
                | {f'nav_tab_{t}_tip' for t in SCAN.tab_keys})
    assert declared == expected


@pytest.mark.parametrize('lang', LANGS)
def test_csv_mode_family_is_complete(lang: str) -> None:
    table = i18n.STRINGS[lang]
    for mode in MeasurementLog.SLICE_MODES:
        assert table.get(f'csv_mode_{mode}', '').strip(), mode
    # csv_mode_label is the combo caption and is excluded from the modes.
    declared = {k for k in table if k.startswith('csv_mode_')} - {'csv_mode_label'}
    assert declared == {f'csv_mode_{m}' for m in MeasurementLog.SLICE_MODES}


@pytest.mark.parametrize('lang', LANGS)
def test_theme_family_is_complete(lang: str) -> None:
    table = i18n.STRINGS[lang]
    for name in theme.THEME_ORDER:
        assert table.get(f'theme_{name}', '').strip(), name
    declared = {k for k in table if k.startswith('theme_')}
    assert declared == {f'theme_{n}' for n in theme.THEME_ORDER}


@pytest.mark.parametrize('lang', LANGS)
def test_drone_voice_family_is_complete(lang: str) -> None:
    """Every drone preset and synth voice has a name; the full GM voices keep
    their standard General MIDI names and have none."""
    table = i18n.STRINGS[lang]
    named = {vid for vid, _n, _p in DRONE_PRESETS + DRONE_SYNTH_VOICES}
    for vid in named:
        assert table.get(f'drone_voice_{vid}', '').strip(), vid
    declared = {k for k in table if k.startswith('drone_voice_')}
    assert declared == {f'drone_voice_{vid}' for vid in named}


@pytest.mark.parametrize('lang', LANGS)
def test_long_instrument_names_exist_for_catalog_keys(lang: str) -> None:
    """``instr_long_*`` is optional per instrument (the window falls back to the
    catalog name), so completeness here means: every entry names a real catalog
    instrument and carries real text with no placeholders."""
    known = {k for (_fk, _de, _en, instruments) in catalog._FAMILIES
             for (k, _t, _d, _e) in instruments}
    table = i18n.STRINGS[lang]
    long_keys = [k for k in table if k.startswith('instr_long_')]
    assert long_keys
    for key in long_keys:
        instrument = key[len('instr_long_'):]
        assert instrument in known, f'{key} names no catalog instrument'
        assert table[key].strip()
        assert not _fields(table[key])
    assert ({k for k in DE if k.startswith('instr_long_')}
            == {k for k in EN if k.startswith('instr_long_')})
