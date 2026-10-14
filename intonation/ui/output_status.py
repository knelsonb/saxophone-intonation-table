# =============================================================================
# intonation/ui/output_status.py — saying why a sound source cannot play
#
# Every part of the window that sounds through the output (test tone,
# metronome, drone, pitch pipes, deck playback) shows the same kind of inline
# message when the output could not open: what failed, then why, naming the
# device and, for a refused sample rate, the rates it refused. The wording is
# built here once, and bound to the Translator so a message that is showing
# follows a language switch.
# =============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from intonation.audio.engine import AudioEngineError, format_rates

if TYPE_CHECKING:
    from intonation.i18n.translator import Translator


def output_reason(t: Callable[..., str], status: Any) -> str:
    """Why an output open failed, naming the device (and the rates it
    refused), in the player's language. '' when there is nothing to add."""
    if status is None:
        return ''
    name = status.device
    err = status.error
    if err == AudioEngineError.UNSUPPORTED_RATE and name and status.rates:
        return t('output_err_rate', name=name, rates=format_rates(status.rates))
    if err == AudioEngineError.DEVICE_BUSY and name:
        return t('output_err_busy', name=name)
    if err in (AudioEngineError.NO_DEVICE, AudioEngineError.DEVICE_DISCONNECTED):
        if name:
            return t('output_err_missing', name=name)
        return t('output_err_none')
    if name:
        return t('output_err_failed', name=name)
    return ''


def with_reason(t: Callable[..., str], text: str, status: Any) -> str:
    reason = output_reason(t, status)
    return f'{text} {reason}' if reason else text


def show_with_reason(tr: 'Translator', label, key: str, status: Any) -> None:
    """Show the message ``key`` on ``label``, followed by why the output
    could not open, in whatever language is current, now and later."""
    tr.bind_text(label.setText, lambda: with_reason(tr.t, tr.t(key), status))
