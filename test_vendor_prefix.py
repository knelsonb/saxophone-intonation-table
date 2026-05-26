"""
Regression tests for ``_promote_vendor_prefix`` in ``sax_intonation_gui``.

These tests were written after three consecutive rendering-defect fixes
(v0.5.7.2, v0.5.7.4, v0.5.7.5) showed the function is high-churn and
prone to regressions.  Each case locks current behaviour.  Cases where
current behaviour disagrees with the obviously-correct answer are marked
``# NOTE: Phase-1 fix needed`` so they can be addressed without touching
this file.

Import notes
------------
``sax_intonation_gui`` imports PyQt6 class definitions at module scope but
does NOT instantiate ``QApplication`` at import time (that is guarded by
``if __name__ == '__main__'``).  Importing the module in a headless pytest
run therefore requires PyQt6 to be installed but does not require a
display.  If PyQt6 is unavailable the whole module is skipped.

Separator note
--------------
The function returns ``f'{VENDOR} · {body}'``.  The separator is a
MIDDLE DOT (U+00B7) with a single space on each side: ``" · "``.
"""
from __future__ import annotations

import pytest

pytest.importorskip('PyQt6', reason='PyQt6 not installed; skipping vendor-prefix tests')

from sax_intonation_gui import _promote_vendor_prefix  # noqa: E402


# ---------------------------------------------------------------------------
# Parametrised table
# ---------------------------------------------------------------------------
# Each row: (test_id, raw_name, expected_output)
#
# The separator used by the function is " · " (space-middledot-space, U+00B7).
# ---------------------------------------------------------------------------
CASES = [
    # ------------------------------------------------------------------
    # 1. Vendor wrapped in parentheses — hoisted, parens stripped.
    #    This is the primary use-case that motivated the function.
    # ------------------------------------------------------------------
    pytest.param(
        'vendor_in_parens',
        'Headset (FIIO DSP Audio)',
        'FIIO · Headset',
    ),

    # ------------------------------------------------------------------
    # 2. Windows numeric-prefix in parens: "(2- Scarlett Solo)".
    #    v0.5.7.4 introduced Step 2 (bare-vendor stripping from the body).
    #    The entire paren block is dropped when it contains a vendor token,
    #    so "Solo" and "2-" are lost along with "Scarlett".
    #
    # NOTE: Phase-1 fix needed — ideally the non-vendor parts of the paren
    # body ("2- Solo") would be re-attached: "SCARLETT · Microphone (2- Solo)".
    # Current behaviour locks at "SCARLETT · Microphone".
    # ------------------------------------------------------------------
    pytest.param(
        'vendor_in_parens_with_numeric_prefix',
        'Microphone (2- Scarlett Solo)',
        'SCARLETT · Microphone',
    ),

    # ------------------------------------------------------------------
    # 3. Bare vendor (no parens) — v0.5.7.5 fix: must NOT produce a
    #    duplicate vendor token in the output.
    # ------------------------------------------------------------------
    pytest.param(
        'bare_vendor_no_parens_no_duplication',
        'FIIO Q3',
        'FIIO · Q3',
    ),

    # ------------------------------------------------------------------
    # 4. Non-vendor parens preserved with correct spacing — v0.5.7.2 fix.
    #    The separator around " - " must be " - " (space-dash-space), not
    #    "- " or " -".
    # ------------------------------------------------------------------
    pytest.param(
        'non_vendor_parens_preserved_space_around_dash',
        'Line In (FIIO) - ASUS',
        'FIIO · Line In - ASUS',
    ),

    # ------------------------------------------------------------------
    # 5. No vendor present — function must return the name unchanged.
    # ------------------------------------------------------------------
    pytest.param(
        'no_vendor_unchanged',
        'Realtek HD Audio',
        'Realtek HD Audio',
    ),

    # ------------------------------------------------------------------
    # 6. Vendor at end of string — vendor is hoisted to the front.
    # ------------------------------------------------------------------
    pytest.param(
        'vendor_at_end_of_string',
        'Audio Device FIIO',
        'FIIO · Audio Device',
    ),

    # ------------------------------------------------------------------
    # 7. Mixed-case vendor in parens — regex is IGNORECASE; vendor is
    #    always uppercased in the output.
    # ------------------------------------------------------------------
    pytest.param(
        'mixed_case_vendor_in_parens',
        'headset (fiio dsp)',
        'FIIO · headset',
    ),

    # ------------------------------------------------------------------
    # 8. Empty string — must not raise; must return "".
    # ------------------------------------------------------------------
    pytest.param(
        'empty_string',
        '',
        '',
    ),

    # ------------------------------------------------------------------
    # 9. None input — the guard ``if not name: return name`` treats None
    #    as falsy and returns it unchanged.  This is current behaviour;
    #    NOTE: Phase-1 consideration — a str-typed function should perhaps
    #    raise TypeError on None rather than silently returning None.
    # ------------------------------------------------------------------
    pytest.param(
        'none_input',
        None,
        None,
    ),

    # ------------------------------------------------------------------
    # 10. Multiple vendors in one name — re.search returns the leftmost
    #     match.  "FIIO" sits before "Behringer", so FIIO is hoisted and
    #     Behringer stays in the body.
    # ------------------------------------------------------------------
    pytest.param(
        'multiple_vendors_leftmost_wins',
        'FIIO Behringer mic',
        'FIIO · Behringer mic',
    ),

    # ------------------------------------------------------------------
    # 11. Vendor as a substring of a non-vendor word — "FIIOX" is not a
    #     vendor token, but VENDOR_REGEX has no word boundary around "fiio",
    #     so it matches "FIIO" inside "FIIOX".  Step 2 uses \b, so the bare
    #     "FIIO" inside "FIIOX" is NOT stripped from the body (word boundary
    #     prevents it).  Result: "FIIO · Studio FIIOX".
    #
    # NOTE: Phase-1 fix needed — VENDOR_REGEX should use \b word boundaries
    #     around each vendor token so "FIIOX" is not treated as a FIIO device.
    # ------------------------------------------------------------------
    pytest.param(
        'vendor_as_substring_of_longer_word',
        'Studio FIIOX',
        'FIIO · Studio FIIOX',
    ),

    # ------------------------------------------------------------------
    # 12. Zoom — must match (brand) but "Zoom call microphone" must NOT
    #     match because of the (?! call) lookahead in VENDOR_REGEX.
    # ------------------------------------------------------------------
    pytest.param(
        'zoom_brand_matches',
        'Zoom F8n',
        'ZOOM · F8n',
    ),
    pytest.param(
        'zoom_call_not_matched',
        'Zoom call microphone',
        'Zoom call microphone',
    ),

    # ------------------------------------------------------------------
    # 13. Vendor already at position 0 in a bare name (no parens) —
    #     the docstring says this case returns the name unchanged, but the
    #     code does NOT implement that guard.  Step 2 strips the bare vendor
    #     and Step 3 re-prefixes it.  For a bare "FIIO DSP Audio" the result
    #     is "FIIO · DSP Audio" (vendor stripped, then re-hoisted).
    #
    # NOTE: Phase-1 — the docstring and the code disagree.  If the intent
    #     is to return "FIIO DSP Audio" unchanged, a guard such as
    #     ``if m.start() == 0: return name`` should be added.  Current
    #     behaviour locks here as "FIIO · DSP Audio".
    # ------------------------------------------------------------------
    pytest.param(
        'vendor_already_at_position_0_bare',
        'FIIO DSP Audio',
        'FIIO · DSP Audio',
    ),

    # ------------------------------------------------------------------
    # 14. Vendor-only input (no body) — after stripping the vendor token
    #     the body collapses to ""; the function returns the bare vendor.
    # ------------------------------------------------------------------
    pytest.param(
        'vendor_only_input',
        'FIIO',
        'FIIO',
    ),

    # ------------------------------------------------------------------
    # 15. "UMC202HD (Behringer)" — compound word-boundary bug:
    #     VENDOR_REGEX has no word boundary around "umc", so it matches
    #     the prefix of "UMC202HD".  "UMC" is hoisted as vendor, and
    #     because \b in Step 2 does NOT fire (UMC202HD has no boundary
    #     after the C), "UMC202HD" survives in the body, producing the
    #     redundant "UMC · UMC202HD".
    #
    # NOTE: Phase-1 fix needed — adding \b around vendor tokens in
    #     VENDOR_REGEX would let "behringer" win here, giving
    #     "BEHRINGER · UMC202HD".  Lock current behaviour.
    # ------------------------------------------------------------------
    pytest.param(
        'umc_prefix_grabbed_over_behringer',
        'UMC202HD (Behringer)',
        'UMC · UMC202HD',
    ),
]


@pytest.mark.parametrize('raw_name,expected', [c.values[1:] for c in CASES],
                         ids=[c.values[0] for c in CASES])
def test_promote_vendor_prefix(raw_name: str | None, expected: str | None) -> None:
    """``_promote_vendor_prefix`` must return the expected string for every
    locked input/output pair in the regression table above."""
    result = _promote_vendor_prefix(raw_name)
    assert result == expected, (
        f'_promote_vendor_prefix({raw_name!r})\n'
        f'  got:      {result!r}\n'
        f'  expected: {expected!r}'
    )
