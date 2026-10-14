"""Pure decision logic for D3 output/mic coordination.

When the desktop sounds its own pitched output (a drone / pitch-pipe) the
microphone hears it too. Two problems follow:

  1. The output's own notes leak into the tuner's incumbent-pitch vote and can
     win it — the readout would lock onto the drone instead of the player.
  2. Sustained leakage means the output is genuinely too loud relative to the
     player; ducking it restores a usable signal-to-leak ratio.

``OutputCoordinator`` is the *policy* for both. Every detection frame it:

  * vote-excludes the output's currently-sounding MIDIs from the incumbent
    vote (so mic-bleed of those notes can never win it), and
  * on *confirmed* suspicion of leakage, ramps a duck gain down toward
    ``duck_depth``; once the leakage clears (past a post-duck confirm window)
    it ramps back to fully open;
  * re-admits a sounding note the player is playing in unison: when the note
    is still heard ``readmit_frames`` frames after the duck reached its floor,
    at a level the duck could not have produced from leakage alone, it is the
    player, and it leaves the exclusion set until the note stops.

It is a pure, deterministic function of per-detection-frame inputs plus an
internal counter: no audio, no Qt, no sounddevice, no numpy, no locks. It is
called once per detection frame — the engine's hop cadence (~46 ms; see
``intonation.audio.engine.HOP_MS``) — and is allocation-light because it runs on the
detection path.

The CONSUMER wiring (engine input callback -> ``update()`` -> drone
``source.set_gain(duck_level)``) lives in ``intonation.audio.engine`` and
``intonation.audio.sources.drone``; this module holds only the policy. See
``test_coordination.py`` for the canonical contract.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Decision", "OutputCoordinator"]

# Pitch detection runs at the engine's hop cadence — one update() per hop. The
# duck ramp is specified in milliseconds (``duck_ms``), so converting it into a
# per-frame step needs the hop *duration*.
#
# CRITICAL (dimensional): the engine holds the hop DURATION constant
# (HOP_MS ≈ 46.44 ms) across ALL sample rates by SCALING the hop sample count
# with sr (hop_samples = round(sr * HOP_MS/1000)). So the detection cadence is
# rate-INVARIANT. Do NOT derive it from samples/sr (e.g. 2048/sr) — that is
# correct only at 44.1 kHz and silently stretches the ramp at every other rate
# (≈186 ms at 96 k, ≈372 ms at 192 k vs. the 80 ms contract; auto-negotiation
# probes 192 k first). We therefore use the constant duration directly; the
# ``samplerate`` constructor param is retained for signature stability but is
# reserved/unused because the cadence does not depend on it.
_DETECTION_HOP_MS = 1000.0 * 2048.0 / 44100.0   # == intonation.audio.engine.HOP_MS


@dataclass(frozen=True)
class Decision:
    """Immutable per-frame coordination decision.

    Attributes
    ----------
    excluded_midis:
        The output's currently-sounding MIDIs, to be removed from the
        incumbent-pitch vote: the ``sounding_output_midis`` passed to
        :meth:`OutputCoordinator.update`, minus a note re-admitted as the
        player's unison (see ``readmit_frames``).
    duck_level:
        The CURRENT (already-ramped) output gain multiplier for this frame, in
        ``[duck_depth, 1.0]``: ``1.0`` = fully open, ``duck_depth`` = fully
        ducked. The consumer applies it verbatim (``set_gain(duck_level)``) —
        the policy owns the ramp shape, the mechanism does no second ramp.
    suspicious:
        ``True`` once leakage is *confirmed* — the detected pitch has been in
        the sounding set for ``suspicion_frames`` consecutive frames.
    """

    excluded_midis: frozenset[int]
    duck_level: float
    suspicious: bool


class OutputCoordinator:
    """D3 coordination policy: vote-exclude + leakage duck.

    Deterministic and self-contained — each instance owns its own counters, so
    two coordinators never share state. Construct one per engine.

    Parameters
    ----------
    suspicion_frames:
        Consecutive frames the detected pitch must sit *inside* the sounding
        set before leakage is confirmed and ducking begins. Suspicion fires at
        exactly the ``suspicion_frames``-th consecutive in-set frame (the
        2-vs-3 off-by-one is intentional: with the default 3, two in a row do
        NOT fire, three do). It doubles as the post-duck confirm window: the
        output is held ducked until the leakage has been clear for the same
        number of consecutive frames, then release begins.
    duck_depth:
        The fully-ducked gain floor (``1.0`` = no ducking).
    duck_ms:
        Time the duck ramp spans, converted to a per-frame step via the hop
        cadence. The release ramp uses the same step (symmetric).
    samplerate:
        Reserved; the detection cadence does not depend on the rate (see
        ``_DETECTION_HOP_MS``).
    readmit_frames:
        Unison re-admission window: consecutive frames the detected note must
        stay in the sounding set with the duck at its floor before its level is
        compared with the level heard before the duck. The default (5 frames,
        ~230 ms) covers the output and input buffering between a gain change
        and the microphone, with margin. Re-admission needs a ``level`` on
        every frame; without one (no duck consumer, so ducking proves nothing)
        a sounding note is never re-admitted.
    """

    def __init__(self, suspicion_frames: int = 3, duck_depth: float = 0.30,
                 duck_ms: float = 80.0, samplerate: int = 44100,
                 readmit_frames: int = 5):
        self.suspicion_frames = int(suspicion_frames)
        self.duck_depth = float(duck_depth)
        self.duck_ms = float(duck_ms)
        self.samplerate = int(samplerate)
        self.readmit_frames = int(readmit_frames)
        # Leakage scales with the duck; a player does not. Halfway between the
        # two outcomes (level ratio duck_depth for pure leakage, 1.0 for a
        # player alone) is the decision point.
        self._readmit_ratio = (1.0 + self.duck_depth) / 2.0

        # Per-frame ramp step: span ``duck_ms`` over (duck_ms / hop_ms) frames.
        # Cadence is rate-invariant (see _DETECTION_HOP_MS) — NOT samples/sr.
        ramp_frames = max(1, round(self.duck_ms / _DETECTION_HOP_MS))
        self._step = (1.0 - self.duck_depth) / ramp_frames

        # The post-duck confirm window before release begins (symmetric with
        # the suspicion threshold): hysteresis against detection flicker.
        self._release_after = self.suspicion_frames

        # Mutable per-frame state.
        self._match_streak = 0   # consecutive frames detected IN the sounding set
        self._clear_streak = 0   # consecutive frames detected OUT of the set
        self._duck = 1.0         # current (already-ramped) gain, fully open
        # Unison re-admission state.
        self._open_level: float | None = None   # level heard before the duck
        self._floor_streak = 0                  # in-set frames at the duck floor
        self._readmitted: int | None = None     # the player's unison note

    def update(self, detected_midi: int | None,
               sounding_output_midis: frozenset[int],
               level: float | None = None) -> Decision:
        """Advance one detection frame and return the coordination Decision.

        ``detected_midi`` is the post-detection incumbent MIDI for this frame,
        or ``None`` on silence/rejection (``None`` is never "in" the sounding
        set, so it can never accumulate suspicion). ``sounding_output_midis``
        is the set the output is currently sounding. ``level`` is the mic level
        of the newest audio in the frame (any linear unit, the same every
        frame), or ``None`` when ducking cannot be observed.
        """
        # Vote-exclude the sounding set. It is already an immutable frozenset
        # per the contract, so pass it straight through (no per-frame
        # allocation); only normalise if a caller hands us another iterable.
        excluded: frozenset[int] = (
            sounding_output_midis
            if isinstance(sounding_output_midis, frozenset)
            else frozenset(sounding_output_midis))

        in_set = (detected_midi is not None
                  and detected_midi in sounding_output_midis)
        if in_set:
            self._match_streak += 1
            self._clear_streak = 0
        else:
            self._match_streak = 0
            self._clear_streak += 1

        suspicious = self._match_streak >= self.suspicion_frames

        if suspicious:
            # Confirmed leakage -> ramp the output gain down toward the floor.
            self._duck = max(self.duck_depth, self._duck - self._step)
        elif self._clear_streak >= self._release_after:
            # Leakage has stayed clear past the confirm window -> release open.
            self._duck = min(1.0, self._duck + self._step)
        # else: still building a suspicion streak, or inside the confirm
        # window -> hold the current gain (hysteresis).

        self._track_unison(detected_midi if in_set else None, level)
        if self._readmitted is not None and self._readmitted in excluded:
            excluded = excluded - {self._readmitted}

        return Decision(
            excluded_midis=excluded,
            duck_level=self._duck,
            suspicious=suspicious,
        )

    def _track_unison(self, midi: int | None, level: float | None) -> None:
        """Decide whether the in-set note this frame is the player.

        Ducking scales whatever the mic hears from the speaker by about
        ``duck_depth``. A note that is still heard at more than halfway between
        that and the level heard before the duck, once the duck has settled at
        its floor, is coming from somewhere the duck does not reach: the player.
        """
        if midi is None or (self._readmitted is not None
                            and midi != self._readmitted):
            # The note stopped or changed. The pre-duck level stays: it is
            # still the reference for as long as the duck has not reopened.
            self._floor_streak = 0
            self._readmitted = None
            if midi is None:
                return
        if level is None:
            # Ducking cannot be observed: nothing can be re-admitted.
            self._open_level = None
            self._floor_streak = 0
            return
        if self._readmitted == midi:
            return
        if self._duck >= 1.0:
            self._open_level = level
        if self._duck <= self.duck_depth + 1e-9:
            self._floor_streak += 1
        else:
            self._floor_streak = 0
        if (self._open_level is not None
                and self._floor_streak >= self.readmit_frames
                and level >= self._open_level * self._readmit_ratio):
            self._readmitted = midi
