"""Tests for AudioEngine._classify_error.

This function is a pure static classifier: it maps an exception (or any
object with .args / __str__) to an AudioEngineError enum value.  It was
untested despite several v0.5.7.x bugs being traced back to
misclassification (DEVICE_BUSY vs UNSUPPORTED_RATE changes the fallback
strategy used by _open_with_fallback, so a wrong result is a UX regression).

Design notes
------------
* We use a ``FakeErr`` class throughout so the test suite can run without
  sounddevice / PortAudio installed.
* All tests call ``AudioEngine._classify_error(exc)`` directly — the method
  is a ``@staticmethod`` so no engine instance is needed.
* The parametrize table is ordered to follow the branch order in the source:
  UNSUPPORTED_RATE first, then DEVICE_BUSY, then NO_DEVICE, then the
  HOSTAPI_FAILURE fallback.

Routing summary (extracted from the source, verified by running the logic):
  1. text contains 'invalid sample rate' OR '-9998'  → UNSUPPORTED_RATE
  2. text contains any of: 'device unavailable', 'busy',
       'unanticipated host error', '0xaa', '-9999', '-9993', '-9994',
       '-9996', 'wdmsyncioctl'                        → DEVICE_BUSY
  3. text contains any of: 'device not found', 'no device',
       'invalid device', 'querying device -1', '-1'   → NO_DEVICE
  4. everything else                                  → HOSTAPI_FAILURE

'text' is built from exc.args (joined, lowercased) when .args is non-empty;
falls back to str(exc).lower() only when .args is an empty tuple.  This means
an Exception() with no arguments routes through the str() path (empty string),
not the args path.

Latent hazard noted: the bare '-1' substring check (branch 3) matches any
string containing the two characters '-' followed by '1', including '-10',
'-12', '-1000', '-10000', etc.  Tests below lock in this current behaviour as
the documented contract so any future change is deliberate.
"""

from __future__ import annotations

import pytest

from sax_audio_engine import AudioEngine, AudioEngineError

# ---------------------------------------------------------------------------
# Fake exception — avoids any dependency on sounddevice.PortAudioError
# ---------------------------------------------------------------------------

class FakeErr(Exception):
    """Minimal stand-in for sounddevice.PortAudioError or any system OSError.

    Behaves exactly like a normal Python Exception: str(FakeErr('x')) == 'x',
    and FakeErr('x').args == ('x',).
    """


class NoArgsErr:
    """An exception-like object with no .args attribute at all.

    Exercises the str(exc).lower() fallback path that is skipped whenever
    .args is a non-empty tuple.  Standard Python exceptions never reach this
    path because Exception() always has .args (even if empty).
    """
    def __init__(self, msg: str) -> None:
        self._msg = msg

    def __str__(self) -> str:  # no .args — falls through to str(exc)
        return self._msg


# ---------------------------------------------------------------------------
# Helper: call the static method under test
# ---------------------------------------------------------------------------

def _classify(exc) -> AudioEngineError:
    return AudioEngine._classify_error(exc)


# ===========================================================================
# UNSUPPORTED_RATE branch
# ===========================================================================

@pytest.mark.parametrize("msg,expected", [
    # Canonical PortAudio error code -9998 as a string in the message.
    ("-9998",                          AudioEngineError.UNSUPPORTED_RATE),
    # Code embedded in a longer message (real PA error format).
    ("Invalid sample rate (-9998)",    AudioEngineError.UNSUPPORTED_RATE),
    # Phrase match without a numeric code.
    ("invalid sample rate",            AudioEngineError.UNSUPPORTED_RATE),
    # Case-insensitivity: keywords are lowercased in both the function and args.
    ("INVALID SAMPLE RATE",            AudioEngineError.UNSUPPORTED_RATE),
    ("Invalid Sample Rate",            AudioEngineError.UNSUPPORTED_RATE),
    # Integer -9998 passed as the sole exception argument.
    # The function converts args via str(c).lower(), so int(-9998) → '-9998'.
    # (FakeErr(-9998).args == (-9998,) → text == '-9998')
], ids=[
    "code_-9998_bare",
    "code_-9998_in_phrase",
    "phrase_lowercase",
    "phrase_all_caps",
    "phrase_title_case",
])
def test_unsupported_rate_string_messages(msg, expected):
    assert _classify(FakeErr(msg)) == expected


def test_unsupported_rate_integer_arg():
    """Integer -9998 in .args routes to UNSUPPORTED_RATE.

    str(int(-9998)) == '-9998', so the numeric code path fires even when the
    exception was raised with a bare integer rather than a formatted string.
    """
    assert _classify(FakeErr(-9998)) == AudioEngineError.UNSUPPORTED_RATE


# ===========================================================================
# DEVICE_BUSY branch
# ===========================================================================

@pytest.mark.parametrize("msg,expected", [
    # Documented string fragments.
    ("device unavailable",                     AudioEngineError.DEVICE_BUSY),
    ("busy",                                   AudioEngineError.DEVICE_BUSY),
    ("unanticipated host error",               AudioEngineError.DEVICE_BUSY),
    # Windows WDM-KS IOCTL failure string (Aragorn memo).
    ("wdmsyncioctl",                           AudioEngineError.DEVICE_BUSY),
    # Hex code 0xaa (decimal 170 — Windows host-API failure code).
    ("0xaa",                                   AudioEngineError.DEVICE_BUSY),
    # PortAudio numeric codes as strings.
    ("-9999",                                  AudioEngineError.DEVICE_BUSY),
    ("-9993",                                  AudioEngineError.DEVICE_BUSY),
    ("-9994",                                  AudioEngineError.DEVICE_BUSY),
    ("-9996",                                  AudioEngineError.DEVICE_BUSY),
    # Codes embedded in realistic error messages.
    ("PortAudio error -9999: Unknown error",   AudioEngineError.DEVICE_BUSY),
    ("stream open failed: -9993",              AudioEngineError.DEVICE_BUSY),
    # Case-insensitivity checks.
    ("Device Unavailable",                     AudioEngineError.DEVICE_BUSY),
    ("BUSY: port in use by another process",   AudioEngineError.DEVICE_BUSY),
    ("0xAA",                                   AudioEngineError.DEVICE_BUSY),  # uppercase hex
    ("Unanticipated Host Error",               AudioEngineError.DEVICE_BUSY),
    ("WDMSYNCIOCTL failed",                    AudioEngineError.DEVICE_BUSY),
], ids=[
    "device_unavailable",
    "busy",
    "unanticipated_host_error",
    "wdmsyncioctl",
    "hex_0xaa_lower",
    "code_-9999",
    "code_-9993",
    "code_-9994",
    "code_-9996",
    "code_-9999_in_phrase",
    "code_-9993_in_phrase",
    "device_unavailable_titlecase",
    "busy_allcaps",
    "hex_0xAA_upper",
    "unanticipated_host_error_titlecase",
    "wdmsyncioctl_upper",
])
def test_device_busy(msg, expected):
    assert _classify(FakeErr(msg)) == expected


@pytest.mark.parametrize("code,expected", [
    (-9999, AudioEngineError.DEVICE_BUSY),
    (-9993, AudioEngineError.DEVICE_BUSY),
    (-9994, AudioEngineError.DEVICE_BUSY),
    (-9996, AudioEngineError.DEVICE_BUSY),
], ids=["int_-9999", "int_-9993", "int_-9994", "int_-9996"])
def test_device_busy_integer_args(code, expected):
    """Integer error codes in .args route correctly to DEVICE_BUSY."""
    assert _classify(FakeErr(code)) == expected


# ===========================================================================
# NO_DEVICE branch
# ===========================================================================

@pytest.mark.parametrize("msg,expected", [
    # Documented string fragments.
    ("device not found",                       AudioEngineError.NO_DEVICE),
    ("no device",                              AudioEngineError.NO_DEVICE),
    ("invalid device",                         AudioEngineError.NO_DEVICE),
    # Exact phrase from PortAudio / sounddevice when default device is -1.
    ("querying device -1",                     AudioEngineError.NO_DEVICE),
    # Bare '-1' — the literal two-char sequence that is also a catch-all.
    ("-1",                                     AudioEngineError.NO_DEVICE),
    # Case-insensitivity.
    ("Device Not Found",                       AudioEngineError.NO_DEVICE),
    ("No Device available",                    AudioEngineError.NO_DEVICE),
    ("Invalid Device index",                   AudioEngineError.NO_DEVICE),
    ("Querying Device -1",                     AudioEngineError.NO_DEVICE),
], ids=[
    "device_not_found",
    "no_device",
    "invalid_device",
    "querying_device_-1",
    "bare_-1",
    "device_not_found_titlecase",
    "no_device_titlecase",
    "invalid_device_titlecase",
    "querying_device_-1_titlecase",
])
def test_no_device(msg, expected):
    assert _classify(FakeErr(msg)) == expected


# ---------------------------------------------------------------------------
# '-1' substring boundary — Phase 1 tightened the match.
#
# The old branch used a plain substring 'in' check for '-1', which over-
# matched on '-10', '-12', '-100', '-1000' and routed them to NO_DEVICE.
# The new regex `(?:^|\D)-1(?!\d)` matches '-1' only when it stands alone
# (not followed by another digit), so larger negative codes correctly fall
# through to HOSTAPI_FAILURE.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "-10",
    "-12",
    "-100",
    "-1000",
    "error -10: open failed",
    "error code -12",
], ids=["neg_10", "neg_12", "neg_100", "neg_1000",
        "neg_10_in_phrase", "neg_12_in_phrase"])
def test_minus_one_no_longer_overmatches(msg):
    """Larger negative codes containing '-1' as a prefix no longer route
    to NO_DEVICE; they fall through to the HOSTAPI_FAILURE fallback."""
    assert _classify(FakeErr(msg)) == AudioEngineError.HOSTAPI_FAILURE


@pytest.mark.parametrize("msg", [
    "-1",
    "error -1",
    "open failed: -1",
], ids=["bare", "in_phrase_leading", "in_phrase_trailing"])
def test_bare_minus_one_still_routes_to_no_device(msg):
    """The standalone '-1' code (not followed by another digit) is the
    legitimate PortAudio 'no device' signal — must still route here."""
    assert _classify(FakeErr(msg)) == AudioEngineError.NO_DEVICE


# ===========================================================================
# HOSTAPI_FAILURE fallback
# ===========================================================================

@pytest.mark.parametrize("msg,expected", [
    # Generic / unknown errors that match none of the above patterns.
    ("generic audio failure",           AudioEngineError.HOSTAPI_FAILURE),
    ("stream could not be opened",      AudioEngineError.HOSTAPI_FAILURE),
    ("portaudio error 9998",            AudioEngineError.HOSTAPI_FAILURE),  # positive code
    # Unlisted PA codes that don't contain '-1' as a substring.
    ("error -9997",                     AudioEngineError.HOSTAPI_FAILURE),
    ("error -9995",                     AudioEngineError.HOSTAPI_FAILURE),
    ("error -9992",                     AudioEngineError.HOSTAPI_FAILURE),
    ("error -9988",                     AudioEngineError.HOSTAPI_FAILURE),
    # Completely empty string (args path, since FakeErr('') has args=('',)).
    ("",                                AudioEngineError.HOSTAPI_FAILURE),
], ids=[
    "generic",
    "stream_open",
    "positive_9998",
    "unlisted_-9997",
    "unlisted_-9995",
    "unlisted_-9992",
    "unlisted_-9988",
    "empty_string",
])
def test_hostapi_failure_fallback(msg, expected):
    assert _classify(FakeErr(msg)) == expected


@pytest.mark.parametrize("code,expected", [
    (-9997, AudioEngineError.HOSTAPI_FAILURE),
    (-9995, AudioEngineError.HOSTAPI_FAILURE),
    (0,     AudioEngineError.HOSTAPI_FAILURE),
], ids=["int_-9997", "int_-9995", "int_0"])
def test_hostapi_failure_integer_args(code, expected):
    """Integer codes that are not in the classifier's lists fall through."""
    assert _classify(FakeErr(code)) == expected


def test_no_args_exception_routes_to_hostapi_failure():
    """Exception() with no arguments → .args is () → falls back to
    str(exc).lower() which is '' → no branch matches → HOSTAPI_FAILURE.

    This is the only normal path where the str(exc) fallback fires.
    """
    assert _classify(FakeErr()) == AudioEngineError.HOSTAPI_FAILURE


def test_none_arg_routes_to_hostapi_failure():
    """Exception(None) → .args == (None,) → text == 'none' →
    no branch matches → HOSTAPI_FAILURE.

    Distinct from no-arg because .args is non-empty, so the function uses
    the args path and produces the string 'none'.
    """
    assert _classify(FakeErr(None)) == AudioEngineError.HOSTAPI_FAILURE


# ===========================================================================
# str(exc) fallback path (no .args attribute)
# ===========================================================================

def test_no_args_attribute_uses_str_for_device_busy():
    """An object without .args at all — the function reads str(exc).lower().

    Standard Python exceptions always have .args, so this path is only
    reached by exotic exception-like objects (some C extensions).  The
    NoArgsErr helper simulates this.
    """
    exc = NoArgsErr("device unavailable")
    assert _classify(exc) == AudioEngineError.DEVICE_BUSY  # type: ignore[arg-type]


def test_no_args_attribute_uses_str_for_unsupported_rate():
    exc = NoArgsErr("invalid sample rate")
    assert _classify(exc) == AudioEngineError.UNSUPPORTED_RATE  # type: ignore[arg-type]


def test_no_args_attribute_empty_str_falls_back():
    exc = NoArgsErr("")
    assert _classify(exc) == AudioEngineError.HOSTAPI_FAILURE  # type: ignore[arg-type]


# ===========================================================================
# UNSUPPORTED_RATE / DEVICE_BUSY boundary — the critical UX boundary
#
# The engine's _open_with_fallback treats these two differently:
#   UNSUPPORTED_RATE → try next sample rate (or surface "device doesn't
#                       accept X Hz" if rate was pinned)
#   DEVICE_BUSY      → skip remaining rates for this device, move to next
#                       device candidate
# A misclassification at this boundary changes the retry strategy shown
# to the user, potentially hiding or exposing working configurations.
# ===========================================================================

class TestUnsupportedRateVsDeviceBusyBoundary:
    """Explicit boundary assertions between the two most consequential
    categories for fallback routing."""

    def test_minus_9998_is_unsupported_not_busy(self):
        """-9998 must be UNSUPPORTED_RATE, not DEVICE_BUSY.

        If this flips, the engine will give up on a device entirely rather
        than trying the next sample rate — a working 44.1 kHz device would
        appear broken when 192 kHz was tried first.
        """
        assert _classify(FakeErr("-9998")) == AudioEngineError.UNSUPPORTED_RATE
        assert _classify(FakeErr(-9998))   == AudioEngineError.UNSUPPORTED_RATE

    def test_minus_9999_is_device_busy_not_unsupported(self):
        """-9999 must be DEVICE_BUSY, not UNSUPPORTED_RATE.

        If this flips, the engine would retry other sample rates on a
        device that is in use by another application — wasting time and
        producing confusing error messages.
        """
        assert _classify(FakeErr("-9999")) == AudioEngineError.DEVICE_BUSY
        assert _classify(FakeErr(-9999))   == AudioEngineError.DEVICE_BUSY

    def test_invalid_sample_rate_phrase_is_unsupported_not_busy(self):
        """The phrase 'invalid sample rate' must route to UNSUPPORTED_RATE.

        This is the exact string many PortAudio backends produce when the
        requested Hz is not supported by the device hardware.
        """
        assert _classify(FakeErr("invalid sample rate")) \
            == AudioEngineError.UNSUPPORTED_RATE

    def test_device_unavailable_is_busy_not_unsupported(self):
        """'device unavailable' must route to DEVICE_BUSY, not UNSUPPORTED_RATE."""
        assert _classify(FakeErr("device unavailable")) \
            == AudioEngineError.DEVICE_BUSY

    def test_adjacent_codes_do_not_cross_boundary(self):
        """-9997 and -9995 (not in any list) fall to HOSTAPI_FAILURE,
        not mistakenly to UNSUPPORTED_RATE or DEVICE_BUSY.
        """
        assert _classify(FakeErr("-9997")) == AudioEngineError.HOSTAPI_FAILURE
        assert _classify(FakeErr("-9995")) == AudioEngineError.HOSTAPI_FAILURE

    def test_0xaa_is_device_busy_not_hostapi_failure(self):
        """Windows WDM-KS GLE 0xAA must be DEVICE_BUSY.

        Documented in Aragorn's memo: the hex string appears in the
        exception message when WDM-KS grabs exclusive access.
        """
        assert _classify(FakeErr("0xaa")) == AudioEngineError.DEVICE_BUSY
        assert _classify(FakeErr("0xAA")) == AudioEngineError.DEVICE_BUSY


# ===========================================================================
# Multi-argument exceptions
# ===========================================================================

@pytest.mark.parametrize("args,expected", [
    # Real sounddevice.PortAudioError passes (message, host_error_code) or
    # just a numeric code.  The function joins all args, so a two-element
    # tuple containing both the string and the code should still classify.
    (("device unavailable", -9994),    AudioEngineError.DEVICE_BUSY),
    (("invalid sample rate", -9998),   AudioEngineError.UNSUPPORTED_RATE),
    (("no device found", 0),           AudioEngineError.NO_DEVICE),
    # The string 'none' produced by joining (None, 0) does not match any
    # branch; the tuple contains enough info in the first element.
    (("generic", 12345),               AudioEngineError.HOSTAPI_FAILURE),
], ids=[
    "multi_device_unavailable_-9994",
    "multi_invalid_sample_rate_-9998",
    "multi_no_device",
    "multi_generic_unknown_code",
])
def test_multi_argument_exceptions(args, expected):
    """Exceptions with multiple .args (common in C-extension errors) are
    handled by joining all elements.  The dominant keyword in the joined
    string determines the category.
    """
    exc = FakeErr(*args)
    assert _classify(exc) == expected


# ===========================================================================
# Case-insensitivity — explicit group asserting every keyword lowercases
# ===========================================================================

@pytest.mark.parametrize("msg,expected", [
    ("INVALID SAMPLE RATE",            AudioEngineError.UNSUPPORTED_RATE),
    ("Invalid Sample Rate",            AudioEngineError.UNSUPPORTED_RATE),
    ("DEVICE UNAVAILABLE",             AudioEngineError.DEVICE_BUSY),
    ("Device Unavailable",             AudioEngineError.DEVICE_BUSY),
    ("BUSY",                           AudioEngineError.DEVICE_BUSY),
    ("Busy",                           AudioEngineError.DEVICE_BUSY),
    ("UNANTICIPATED HOST ERROR",       AudioEngineError.DEVICE_BUSY),
    ("Unanticipated Host Error",       AudioEngineError.DEVICE_BUSY),
    ("WDMSYNCIOCTL",                   AudioEngineError.DEVICE_BUSY),
    ("WdmSyncIoctl",                   AudioEngineError.DEVICE_BUSY),
    ("0XAA",                           AudioEngineError.DEVICE_BUSY),   # unusual casing
    ("DEVICE NOT FOUND",               AudioEngineError.NO_DEVICE),
    ("Device Not Found",               AudioEngineError.NO_DEVICE),
    ("NO DEVICE",                      AudioEngineError.NO_DEVICE),
    ("No Device",                      AudioEngineError.NO_DEVICE),
    ("INVALID DEVICE",                 AudioEngineError.NO_DEVICE),
    ("Invalid Device",                 AudioEngineError.NO_DEVICE),
    ("QUERYING DEVICE -1",             AudioEngineError.NO_DEVICE),
    ("Querying Device -1",             AudioEngineError.NO_DEVICE),
], ids=[
    "UNSUPPORTED_RATE/all_caps",
    "UNSUPPORTED_RATE/title_case",
    "DEVICE_BUSY/device_unavailable_caps",
    "DEVICE_BUSY/device_unavailable_title",
    "DEVICE_BUSY/busy_caps",
    "DEVICE_BUSY/busy_title",
    "DEVICE_BUSY/unanticipated_caps",
    "DEVICE_BUSY/unanticipated_title",
    "DEVICE_BUSY/wdmsyncioctl_caps",
    "DEVICE_BUSY/wdmsyncioctl_mixed",
    "DEVICE_BUSY/0XAA_upper_x",
    "NO_DEVICE/device_not_found_caps",
    "NO_DEVICE/device_not_found_title",
    "NO_DEVICE/no_device_caps",
    "NO_DEVICE/no_device_title",
    "NO_DEVICE/invalid_device_caps",
    "NO_DEVICE/invalid_device_title",
    "NO_DEVICE/querying_-1_caps",
    "NO_DEVICE/querying_-1_title",
])
def test_case_insensitivity(msg, expected):
    """Every documented keyword must match regardless of case because the
    function applies .lower() to both the exception text and the patterns.
    """
    assert _classify(FakeErr(msg)) == expected


# ===========================================================================
# Return-type contract
# ===========================================================================

def test_return_type_is_always_audioengine_error():
    """_classify_error must always return an AudioEngineError member —
    never None, never a string, never raise.
    """
    for msg in ["", "device unavailable", "-9998", "totally unknown", None]:
        result = _classify(FakeErr(msg))
        assert isinstance(result, AudioEngineError), (
            f"Expected AudioEngineError, got {type(result)} for msg={msg!r}")


def test_never_returns_none_member():
    """AudioEngineError.NONE is the 'no error' sentinel; _classify_error
    should never return it (that value is reserved for the success path).
    """
    for msg in ["", "device unavailable", "-9998", "totally unknown"]:
        result = _classify(FakeErr(msg))
        assert result is not AudioEngineError.NONE, (
            f"_classify_error returned NONE for {msg!r} — "
            f"NONE is a non-error sentinel that should never come from classification")
