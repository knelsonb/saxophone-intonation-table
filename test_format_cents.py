"""Safety net for the two cents-formatting helpers — INTEGER-CENTS contract.

* format_cents(value_cents, freq_hz, sample_rate)        in sax_intonation_gui
* _format_cents_label(value_cents, freq_hz, sample_rate) in sax_intonation_chart

Contract ("never show decimal cents"): both format a finite cent value as a
SIGNED INTEGER ("+8" / "-12" / "+0") via int(round(value)). A real saxophone tone
vibratos by more than a cent, so sub-cent digits claim resolution the reading
can't hold and only jitter the display; integer is the correct tuner readout.
The previous adaptive tenths/halves/wholes tiers were removed.

* freq_hz is now used ONLY for the validity guard (<=0 / non-finite -> sentinel);
  the formatted text no longer depends on freq_hz or sample_rate (both retained
  for API stability).
* Non-finite value_cents or non-finite/non-positive freq_hz -> '–' (U+2013 EN DASH).
* Always a sign: -0.0 and tiny values round to '+0', never '-0' or a bare '0'.
* The two implementations live in separate modules but must agree on every input.

No QApplication needed (pure formatters); importing the GUI module needs PyQt6.
"""
from __future__ import annotations

import pytest

pytest.importorskip('PyQt6', reason='PyQt6 not installed; skipping cents-format tests')

from sax_intonation_gui import format_cents  # noqa: E402
from sax_intonation_chart import _format_cents_label  # noqa: E402

SENTINEL = '–'   # U+2013 EN DASH

_BOTH = pytest.mark.parametrize(
    "fn", [format_cents, _format_cents_label],
    ids=["format_cents", "_format_cents_label"])


# 1. Non-finite value_cents -> sentinel, never raises (the historic NaN guard).
@_BOTH
@pytest.mark.parametrize("bad", [float('nan'), float('inf'), float('-inf')],
                         ids=["nan", "+inf", "-inf"])
def test_nonfinite_value_returns_sentinel(fn, bad):
    assert fn(bad, 440.0, 44100) == SENTINEL


# 2. Non-finite / non-positive freq_hz -> sentinel (guard fires before arithmetic).
@_BOTH
@pytest.mark.parametrize("bad_freq", [float('nan'), float('inf'), float('-inf'),
                                      0.0, -1.0, -440.0],
                         ids=["nan", "+inf", "-inf", "zero", "neg1", "neg440"])
def test_bad_freq_returns_sentinel(fn, bad_freq):
    assert fn(-12.5, bad_freq, 44100) == SENTINEL


# 3. Integer formatting — signed, rounded to nearest int (Python round-half-to-even).
@_BOTH
@pytest.mark.parametrize("value,expected", [
    (0.0,    '+0'),
    (-0.0,   '+0'),     # -0.0 >= 0 -> '+'
    (-0.2,   '+0'),     # round(-0.2) -> 0 -> '+0', never '-0'
    (0.2,    '+0'),
    (7.85,   '+8'),
    (-12.5,  '-12'),    # half-to-even
    (-23.4,  '-23'),
    (50.0,   '+50'),
    (-7.6,   '-8'),
    (999.9,  '+1000'),
    (-500.0, '-500'),
], ids=["zero", "negzero", "neg_tiny", "pos_tiny", "round_up", "half_even",
        "neg", "fifty", "neg_round", "big", "neg500"])
def test_integer_format(fn, value, expected):
    assert fn(value, 440.0, 44100) == expected


# 4. Output is now freq/sample-rate INDEPENDENT (only the >0 guard reads freq).
@_BOTH
@pytest.mark.parametrize("freq", [50.0, 76.0, 178.0, 440.0, 2000.0])
@pytest.mark.parametrize("sr", [8000, 44100, 48000, 0])
def test_format_is_freq_and_sr_independent(fn, freq, sr):
    assert fn(7.85, freq, sr) == '+8'
    assert fn(-12.5, freq, sr) == '-12'


# 5. THE contract: no non-sentinel output ever contains a decimal point.
@_BOTH
@pytest.mark.parametrize("value", [0.0, 0.4, -0.4, 7.85, -12.5, 49.99, -100.1, 999.9])
def test_never_shows_a_decimal_point(fn, value):
    out = fn(value, 440.0, 44100)
    assert '.' not in out, f"{fn.__name__}({value}) = {out!r} shows a decimal point"


# 6. Always an explicit sign on non-sentinel output.
@_BOTH
@pytest.mark.parametrize("value", [0.0, -0.0, 7.85, -12.5, 50.0, -23.0])
def test_explicit_sign(fn, value):
    assert fn(value, 440.0, 44100)[0] in ('+', '-')


# 7. Return type is always str.
@_BOTH
@pytest.mark.parametrize("value,freq,sr", [
    (float('nan'), 440.0, 44100), (0.0, 440.0, 44100), (-12.5, 440.0, 44100),
    (7.85, 76.0, 44100), (-0.0, 77.0, 44100), (999.9, 440.0, 0)])
def test_return_type_str(fn, value, freq, sr):
    assert isinstance(fn(value, freq, sr), str)


# 8. The two implementations agree on every input (live readout == exported chart).
@pytest.mark.parametrize("value,freq,sr", [
    (float('nan'), 440.0, 44100), (float('inf'), 440.0, 44100), (float('-inf'), 440.0, 44100),
    (0.0, 0.0, 44100), (0.0, -1.0, 44100),
    (0.0, 440.0, 44100), (-0.0, 440.0, 44100), (0.0, 76.0, 44100), (-0.0, 77.0, 44100),
    (-12.5, 440.0, 44100), (7.85, 440.0, 44100), (50.0, 440.0, 44100), (-23.4, 440.0, 44100),
    (7.85, 76.0, 44100), (-12.5, 77.0, 44100), (999.9, 440.0, 44100), (-500.0, 440.0, 44100),
    (7.85, 440.0, 0), (7.85, 440.0, 8), (7.85, 440.0, 1_000_000_000)])
def test_implementations_agree(value, freq, sr):
    assert format_cents(value, freq, sr) == _format_cents_label(value, freq, sr)


# 9. Sentinel glyph identity — U+2013 EN DASH (paint path compares to the literal).
def test_sentinel_is_en_dash():
    assert SENTINEL == '–'
    assert format_cents(float('nan'), 440.0, 44100) == '–'
    assert _format_cents_label(float('nan'), 440.0, 44100) == '–'
    assert format_cents(float('nan'), 440.0, 44100) not in ('-', '—')
