"""Virtual verification bench — pitch + timing of every input and output,
measured from the ACTUAL rendered/detected signal with no audio hardware.

Operator standard: "nothing should ever feel off — we should be able to verify
the timing and pitch of every input and output virtually before sending this
into the wild." Most modules test their MATH (beat_onset_sample, midi_to_freq);
this bench renders the real sources through the real mixer (and feeds the real
engine) and MEASURES the result — the end-to-end "does it actually sound at the
right pitch, at the right time" gate.

Pure numpy (no tsf / no PortAudio) so it runs on system-python and the venv. The
drone-pitch + deck-timing + input-latency benches build on this in follow-ups.
"""
from __future__ import annotations

import numpy as np
import pytest

import sim_harness as H
from sax_metronome import MetronomeController, beat_onset_sample, beats_per_bar
from sax_mixer import Mixer, TestToneSource

pp = pytest.importorskip("sax_pitch_pipes", reason="pitch pipes module")
PitchPipeSource = pp.PitchPipeSource

SR = 48000


# ---------------------------------------------------------------------------
# 0. Lock the onset primitive itself (so the timing checks below are trustworthy).
# ---------------------------------------------------------------------------
def test_onset_samples_finds_bursts_at_known_positions():
    sig = np.zeros(60000, dtype=np.float32)
    positions = [1000, 13000, 25000, 49000]
    n = 1500
    t = np.arange(n)
    burst = (np.sin(2 * np.pi * 1000 * t / SR) * np.exp(-t / 400.0) * 0.5).astype(np.float32)
    for p in positions:
        sig[p:p + n] += burst
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=4000)
    assert len(onsets) == len(positions), f"expected {len(positions)} onsets, got {onsets}"
    for det, true in zip(onsets, positions):
        assert abs(det - true) <= 5, f"onset {det} vs true {true}"


# ---------------------------------------------------------------------------
# 1. OUTPUT timing — the metronome's RENDERED clicks land on the BPM grid with
#    no drift. (beat_onset_sample's integer-drift proof is the math; this is the
#    audio a speaker would actually emit.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bpm", [60, 120, 200])
def test_metronome_rendered_clicks_on_bpm_grid_no_drift(bpm):
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, SR, bpm=bpm)
    metro.start()
    interval = SR * 60.0 / bpm
    nbeats = 8
    sig = H.render_mixer(mixer, int(interval * nbeats + interval), block=mb)
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=int(interval * 0.4))
    assert len(onsets) >= nbeats, f"{len(onsets)} click onsets < {nbeats}: {onsets}"
    onsets = onsets[:nbeats]
    intervals = np.diff(onsets)
    assert np.all(np.abs(intervals - interval) <= 2), (
        f"rendered beat spacing must be {interval:.1f}±2 samples (no drift): "
        f"{intervals.tolist()}")
    # cumulative drift over all beats — the thing that "feels off" if it creeps
    drift = abs((onsets[-1] - onsets[0]) - interval * (nbeats - 1))
    assert drift <= 2, f"cumulative drift over {nbeats} beats @ {bpm}bpm = {drift} samples"
    # and the absolute grid matches beat_onset_sample (the math) within the
    # click's ~1 ms attack offset
    anchor = onsets[0]
    for i, o in enumerate(onsets):
        expected = anchor + beat_onset_sample(i, bpm, SR)
        assert abs(o - expected) <= 4, f"beat {i} @ {bpm}bpm: onset {o} vs grid {expected}"


# ---------------------------------------------------------------------------
# 2. OUTPUT pitch — the RENDERED test tone is actually at its target frequency
#    (independent FFT readback, not the source's own dphi).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f", [110.0, 220.0, 440.0, 880.0, 1760.0])
def test_test_tone_rendered_pitch_matches_target(f):
    tone = TestToneSource(f, SR, 2048, gain=0.3)
    sig = H.render_stream(tone, 16384, block=2048)
    det = H.fft_peak_hz(sig, SR)
    err = H.cents(det, f)
    assert abs(err) < 1.5, f"test tone {f}Hz rendered as {det:.2f}Hz ({err:+.2f}c)"


# ---------------------------------------------------------------------------
# 3. OUTPUT pitch — the RENDERED pitch-pipe pad sounds at midi_to_freq(midi,a4).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("midi", [45, 57, 69, 81, 93])  # A2..A6
def test_pitch_pipe_rendered_pitch_matches_target(midi):
    a4 = 440.0
    src = PitchPipeSource(midi, SR, 2048, a4=a4)
    sig = H.render_stream(src, 16384, block=2048, warmup=2)  # skip the 8 ms attack
    det = H.fft_peak_hz(sig, SR)
    true = pp.midi_to_freq(midi, a4)
    err = H.cents(det, true)
    assert abs(err) < 1.5, (
        f"pitch pad midi={midi} ({true:.2f}Hz) rendered {det:.2f}Hz ({err:+.2f}c)")


# ---------------------------------------------------------------------------
# 4. INPUT pitch — a synthetic tone driven through the REAL engine pipeline
#    (silence gate -> YIN -> confirm/lock -> emit) confirms the right note + cents.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,tol", [("sine", 0.6), ("sax", 0.6), ("noisy", 2.0)],
                         ids=["sine", "sax", "noisy"])
@pytest.mark.parametrize("f,midi", [
    (110.0, 45), (220.0, 57), (440.0, 69), (659.25, 76), (880.0, 81)],
    ids=["A2", "A3", "A4", "E5", "A5"])
def test_input_pipeline_confirms_correct_pitch(f, midi, kind, tol):
    """INPUT: 5 notes x 3 timbres through the FULL engine pipeline (gate -> YIN
    -> confirm -> emit) confirm the right note. Clean tones measure sub-cent (so
    the old 2c band WAS wide) -> 0.6c; the 20 dB-SNR 'noisy' timbre is the
    realistic worst case -> 2c."""
    res = H.detect_tone(f, kind=kind, hops=24)
    assert res.emitted, f"{kind} {f}Hz must confirm + emit"
    assert res.dominant_midi == midi, f"{kind} {f}Hz -> midi {res.dominant_midi} (expected {midi})"
    assert abs(res.cents_error()) < tol, (
        f"{kind} {f}Hz cents err {res.cents_error():+.2f}c (tol {tol})")


# ---------------------------------------------------------------------------
# 5. INPUT timing — detection latency (the confirm/lock machinery is not
#    instant, but must be PROMPT). The input twin of onset timing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f", [220.0, 440.0, 880.0])
def test_input_detection_latency_bounded(f):
    hops = H.detection_latency_hops(f, kind="sax")
    assert hops is not None, f"{f}Hz never confirmed within the hop cap"
    assert hops <= 12, f"{f}Hz took {hops} hops to confirm (latency regression?)"


# ---------------------------------------------------------------------------
# 6. OUTPUT pitch — 5 notes x 3 drone VOICES (patches). The intended FUNDAMENTAL
#    is present + in tune, measured by peak_near_hz (a global FFT peak finds
#    whichever partial is loudest — measured an octave / +2792c off for these
#    rich SF2 voices — and YIN octave-errs on the organ's sub-octave; the app
#    never self-measures the drone, it knows the note from active_midi). The ~12c
#    band is FFT-on-rich-timbre MEASUREMENT spread, not the synth's tuning (tsf
#    set_tuning is exact). Voices chosen are the non-ensemble patches; the
#    ensemble ones (strings/warmpad) carry intentional chorus detune (~15-25c
#    measured at low notes) — a fat-drone feature, excluded from this precision
#    gate by design. tsf/venv only.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("voice", ["organ", "cello", "tenorsax"])
def test_drone_rendered_pitch_matches_reference(voice):
    pytest.importorskip("tinysoundfont", reason="drone synth is venv/CI only")
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id=voice, volume=0.7)
    c.set_enabled(True)                                    # one SF2 load per voice
    for midi, name in [(45, "A2"), (57, "A3"), (69, "A4"), (76, "E5"), (81, "A5")]:
        c.set_reference_midi(midi)
        assert c.source.active_midi == midi, f"{voice} must declare midi {midi}"
        sig = H.render_mixer(mixer, 32768, block=2048)[6000:]   # skip the note attack
        true = D.midi_to_freq(midi, 440.0)
        det = H.peak_near_hz(sig, true, SR)
        assert det > 0, f"drone {voice} {name}: no fundamental near {true:.1f}Hz"
        err = H.cents(det, true)
        assert abs(err) < 12.0, f"drone {voice} {name} ({true:.1f}Hz) -> {det:.1f}Hz ({err:+.1f}c)"


# ---------------------------------------------------------------------------
# 7. OUTPUT timing + pitch — a recorded take plays back for its full duration
#    at the right pitch, then stops on time. (Cross-rate resample pitch is also
#    in test_deck.py::test_playback_resample_preserves_pitch.)
# ---------------------------------------------------------------------------
def test_deck_playback_duration_and_pitch():
    from sax_deck import DeckPlaybackSource
    n, f = 24000, 440.0                                    # 0.5 s take @ A4
    take = (np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.5).astype(np.float32)
    src = DeckPlaybackSource(take, SR, SR, 2048)
    out = H.render_stream(src, n + 4096, block=2048)
    det = H.fft_peak_hz(out[300:n - 300], SR)              # skip the 5 ms edge fades
    assert abs(H.cents(det, f)) < 2.0, f"deck playback pitch {det:.2f}Hz vs {f}"
    assert H.max_abs(out[n + 1000:]) < 1e-3, "playback must STOP at the take's end (timing)"


# ---------------------------------------------------------------------------
# 8. OUTPUT timing — the metronome ACCENT GRID across 5 tempos x each stock time
#    signature. Verifies, from the rendered audio: (a) every click on the BPM
#    grid with no drift, and (b) the downbeat accent falls on every
#    beats_per_bar-th beat (the accent click is louder than the plain click).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("time_sig", ["2/4", "3/4", "4/4", "6/8"])
@pytest.mark.parametrize("bpm", [40, 80, 120, 160, 200])
def test_metronome_accent_grid_across_tempos_and_time_sigs(time_sig, bpm):
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, SR, bpm=bpm, time_sig=time_sig)
    metro.start()
    n = beats_per_bar(time_sig)
    interval = SR * 60.0 / bpm
    nbeats = n * 3 + 1                                     # ~3 full bars + 1
    sig = H.render_mixer(mixer, int(interval * (nbeats + 1)), block=mb)
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=int(interval * 0.4))
    assert len(onsets) >= nbeats, f"{time_sig}@{bpm}: {len(onsets)} onsets < {nbeats}"
    onsets = onsets[:nbeats]
    # (a) grid: even spacing, no drift over all beats
    intervals = np.diff(onsets)
    assert np.all(np.abs(intervals - interval) <= 2), (
        f"{time_sig}@{bpm}: beat spacing drifts from {interval:.0f}: {intervals.tolist()}")
    # (b) accent pattern: every n-th beat (the downbeat) is LOUDER than the rest
    win = min(int(interval * 0.3), 1500)
    peaks = [float(np.max(np.abs(sig[o:o + win]))) for o in onsets]
    accented = [peaks[i] for i in range(nbeats) if i % n == 0]
    plain = [peaks[i] for i in range(nbeats) if i % n != 0]
    assert plain, f"{time_sig}: expected some non-downbeat beats"
    assert min(accented) > max(plain), (
        f"{time_sig}@{bpm}: downbeat accent not louder than plain beats — "
        f"accent~{min(accented):.2f} vs plain~{max(plain):.2f}")


# ---------------------------------------------------------------------------
# 9. OUTPUT pitch STABILITY — the organ drone is the steady PRECISION reference
#    (vibrato-free), centred on pitch. The sampled-instrument voices carry
#    baked-in vibrato/chorus (measured at A4: strings ~21c, tenorsax ~14c,
#    warmpad ~12c, cello ~7c spread) and are COLOUR voices, not precision
#    references -- documented here, not asserted. The pure-sine test-tone /
#    pitch-pipe (sections 2-3) are the sub-cent precision references.
# ---------------------------------------------------------------------------
def test_drone_organ_is_a_steady_reference():
    """The organ is the steadiest, most-centred drone voice — verified at concert
    A4, where FFT-based stability measurement of a rich tone is reliable: pitch
    spread < 6c (vibrato-free) and centre within 4c. (Low notes like A2 are
    FFT-resolution-limited for this measure — ~46c/bin at 110 Hz — so the lock is
    the reliably-measurable A4 point, not a claim the organ is untestable there.
    For sub-cent precision across the full range, the pure-sine test-tone /
    pitch-pipe are the references; the drone is a musical reference.)"""
    pytest.importorskip("tinysoundfont", reason="drone synth is venv/CI only")
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id="organ", volume=0.7)  # ref 69 = A4
    c.set_enabled(True)
    sig = H.render_mixer(mixer, 180000, block=2048)[20000:]   # ~3.3 s steady
    true = D.midi_to_freq(69, 440.0)
    mean_c, spread_c = H.pitch_stability(sig, true, SR)
    assert spread_c < 6.0, f"organ A4 pitch spread {spread_c:.1f}c — not steady (vibrato?)"
    assert abs(mean_c) < 4.0, f"organ A4 centre {mean_c:+.1f}c off"


# ---------------------------------------------------------------------------
# 10. INPUT x OUTPUT coordination — the DRONE must not fool the TUNER. When the
#     output sounds a note that bleeds into the mic, the engine vote-excludes it
#     (D3 mic-coordination) so the drone's own note can never become the locked
#     reading. The most direct "feels off" scenario: turn on the drone and watch
#     the tuner start reading the DRONE instead of you. Driven through the REAL
#     detection hot path (feed_input_frames -> _on_input -> coordination_step).
# ---------------------------------------------------------------------------
def test_player_note_detected_while_drone_sounds_a_different_note():
    """Drone sounds A4 (69); player plays C5 (72). The tuner must read the
    PLAYER, not the drone — your live readout stays yours while the drone sounds."""
    res = H.detect_with_sounding_output(pp.midi_to_freq(72, 440.0), [69], hops=24)
    assert res.emitted, "player note must still be detected while the drone sounds"
    assert res.dominant_midi == 72, (
        f"tuner read midi {res.dominant_midi}, expected the player's 72 "
        f"(the drone's sounding 69 must be vote-excluded, not read)")


def test_drone_leakage_at_its_own_pitch_is_rejected():
    """Player silent; the mic hears ONLY the drone's A4 bleed. The tuner must
    stay quiet — the drone's own sounding note is vote-excluded, never locked."""
    res = H.detect_with_sounding_output(pp.midi_to_freq(69, 440.0), [69], hops=24)
    assert not res.emitted, (
        f"drone's own note leaked into the locked reading: {res.notes}")


def test_negative_control_same_leakage_locks_when_no_drone_sounds():
    """The control that proves the rejection above is REAL: feed the identical
    A4 tone with NOTHING sounding on the output and the engine locks 69 normally.
    So the rejection is the coordination doing its job — not A4 failing to lock."""
    res = H.detect_with_sounding_output(pp.midi_to_freq(69, 440.0), [], hops=24)
    assert res.dominant_midi == 69, (
        f"with no drone sounding, A4 must lock normally; got {res.dominant_midi}")


# ---------------------------------------------------------------------------
# 11. OUTPUT timing — the metronome survives an output-device switch (Option-B
#     resync) cleanly in the RENDERED audio. A device switch has an unknown
#     SILENT gap (old stream stops, new device opens), so resync deliberately
#     re-anchors the next beat at the START of the new clock and continues the
#     beat INDEX/accent — it does NOT try to preserve a sub-beat phase across
#     silence that the musician can't hear anyway. test_metro_resync proves the
#     scheduler STATE; this proves the audio: a drift-free grid on EACH side of
#     the switch + a PROMPT re-attack on the new stream (no hang, no silence).
#     The "feels off" scenario: swap headphones for speakers mid-practice and
#     the metronome hangs or the beat starts drifting.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bpm", [60, 120, 200])
def test_metronome_resync_renders_clean_beat_across_device_switch(bpm):
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, SR, bpm=bpm)
    metro.start()
    interval = SR * 60.0 / bpm

    def grid(sig):
        on = H.onset_samples(sig, threshold=1e-2, min_gap=int(interval * 0.4))
        return on, np.diff(on)

    before = H.render_mixer(mixer, int(interval * 5), block=mb)
    on_b, iv_b = grid(before)
    assert len(on_b) >= 4, f"@{bpm}bpm too few pre-switch clicks: {on_b}"
    assert np.all(np.abs(iv_b - interval) <= 2), (
        f"@{bpm}bpm pre-switch beat grid drifted: {iv_b.tolist()}")
    beat_before = metro._beat

    mixer.reset_clock()                  # open_output_device() does this
    metro.resync()                       # Option B: re-anchor onto the new clock

    after = H.render_mixer(mixer, int(interval * 5), block=mb)
    on_a, iv_a = grid(after)
    assert len(on_a) >= 4, f"@{bpm}bpm too few post-switch clicks: {on_a}"
    # PROMPT re-attack: the first click on the new stream lands within one beat,
    # not after a hang or a dropped beat.
    assert on_a[0] <= interval, f"@{bpm}bpm first post-switch click late: {on_a[0]}"
    # the new stream's grid is itself drift-free
    assert np.all(np.abs(iv_a - interval) <= 2), (
        f"@{bpm}bpm post-switch beat grid drifted: {iv_a.tolist()}")
    # and the beat INDEX continued across the switch (not a reset to beat 0) —
    # ties the rendered re-attack to the accent-pattern continuity.
    assert metro._beat > beat_before, (
        f"@{bpm}bpm beat index reset across the switch ({beat_before} -> {metro._beat})")


# ---------------------------------------------------------------------------
# 12. The verifier has TEETH — prove every measurement class the bench leans on
#     actually RESPONDS to a known fault. A bench that only sees correct inputs
#     and passes is unproven: a too-loose gate or a measurement that reads the
#     wrong thing would rubber-stamp a real regression. Sections 0 (onset
#     accuracy) and 10 (coordination negative control) are this discipline; here
#     it is applied to the pitch + timing measurements the WHOLE bench depends on.
#     Each test injects a KNOWN error and asserts the metric tracks it ~linearly,
#     so the sub-cent / sub-sample gates elsewhere demonstrably have a real error
#     bar rather than a green light that can't turn red.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("detune_c", [-50, -25, -10, 10, 25, 50])
def test_fft_peak_tracks_injected_detune(detune_c):
    """fft_peak_hz reads a KNOWN detune as ~that many cents — so the sub-1.5c
    output-pitch gates (§2/§3) would actually fail on a real detune this size."""
    target = 440.0
    detuned = target * 2.0 ** (detune_c / 1200.0)
    sig = H.sine(detuned, SR, 16384, 0.5)
    measured = H.cents(H.fft_peak_hz(sig, SR), target)
    assert abs(measured - detune_c) <= 2.5, (
        f"injected {detune_c:+d}c read {measured:+.2f}c — the pitch gate is blind to it")


@pytest.mark.parametrize("detune_c", [-45, -20, 20, 45])
def test_engine_cents_tracks_injected_detune(detune_c):
    """The full engine pipeline's cents_error tracks a known detune — the INPUT
    pitch gate (§4) has a real error bar, not a rubber stamp. (Measured vs the
    INTENDED pitch; |detune| < 50c keeps the nearest semitone = the target note.)"""
    target = 440.0
    detuned = target * 2.0 ** (detune_c / 1200.0)
    res = H.detect_tone(detuned, kind='sine', hops=24)
    res.target_hz = target
    assert abs(res.cents_error() - detune_c) <= 3.0, (
        f"injected {detune_c:+d}c, engine read {res.cents_error():+.2f}c")


@pytest.mark.parametrize("detune_c", [-40, -15, 15, 40])
def test_peak_near_tracks_detune_on_rich_timbre(detune_c):
    """peak_near_hz tracks a known detune on a HARMONICALLY RICH tone — the drone
    case where a global FFT peak can land on a partial. Proves the drone-pitch
    gate (§6) catches a real detune rather than locking the on-target bin."""
    target = 440.0
    detuned = target * 2.0 ** (detune_c / 1200.0)
    sig = H.sax_like(detuned, SR, 16384, 0.5)        # 5 harmonics, fundamental detuned
    measured = H.cents(H.peak_near_hz(sig, target, SR), target)
    assert abs(measured - detune_c) <= 6.0, (
        f"peak_near read {measured:+.2f}c for an injected {detune_c:+d}c detune")


def test_onset_metric_sees_injected_timing_drift():
    """The no-drift gates (§1/§8/§11 assert beat spacing within 2-3 samples) must
    be able to SEE drift. A synthetic click grid whose interval grows every beat
    must surface intervals well past the gate — so a real drifting metronome
    could not slip through as 'no drift'."""
    base = 12000
    t = np.arange(1500)
    burst = (np.sin(2 * np.pi * 1000 * t / SR) * np.exp(-t / 400.0) * 0.5).astype(np.float32)
    sig = np.zeros(base * 9, dtype=np.float32)
    pos = 0
    for k in range(6):
        sig[pos:pos + 1500] += burst
        pos += base + k * 100                        # each interval 100 samples longer
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=int(base * 0.4))
    intervals = np.diff(onsets)
    assert len(onsets) >= 5, f"injected grid under-detected: {onsets.tolist()}"
    assert intervals.max() - intervals.min() >= 100, (
        f"onset metric blind to injected drift: intervals={intervals.tolist()}")


# ---------------------------------------------------------------------------
# 13. INTERACTIONS / simultaneous use — sections 1-12 verify each source in
#     ISOLATION; this covers the scenarios where they run TOGETHER, which is how
#     the app is actually used and where "feels off" hides. (1) The single most
#     common pairing — metronome + tuner — bleeds UNPITCHED clicks into the mic:
#     clicks have active_midi=None so they are NOT vote-excluded by D3, so the
#     detector itself must reject the transient. (2) Two outputs sounding at once
#     must each keep their pitch through the additive mix (test_mixer pins the
#     sample-level sum; this is the musical property a player would hear).
# ---------------------------------------------------------------------------
def test_metronome_click_bleed_does_not_false_trigger_detection():
    """Player silent, metronome running: the clicks bleeding into the mic are
    broadband transients with no stable pitch — the confirm/lock machinery must
    NOT emit a phantom note. (Clicks are active_midi=None, so D3 can't help; the
    detector's own transient rejection is what's under test.)"""
    sr = H.DEFAULT_SR
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, sr, bpm=120)
    metro.start()
    # enough hops that a REAL pitched tone WOULD confirm + emit (cf. §4 hops=24),
    # so "clicks don't emit" tests rejection, not just too-short a signal.
    total = H.DEFAULT_N * 24
    clicks = H.render_mixer(mixer, total, block=mb).astype(np.float32)
    from sax_audio_engine import AudioEngine
    eng = AudioEngine()
    eng.set_filter_mode('normal')
    eng.set_a4(440.0)
    res = H.feed_engine(eng, clicks)
    assert not res.emitted, (
        f"metronome click bleed false-triggered a phantom note: {res.notes}")


def test_metronome_click_bleed_does_not_corrupt_player_pitch():
    """Player on A4 WITH metronome clicks bleeding into the mic: the tuner must
    still read the player's note and pitch — the transient clicks are outlier-
    rejected, not averaged into the reading."""
    sr = H.DEFAULT_SR
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, sr, bpm=120)
    metro.start()
    total = H.DEFAULT_N * 24          # enough hops to confirm + emit (cf. §4)
    clicks = H.render_mixer(mixer, total, block=mb).astype(np.float32)
    player = H.sax_like(440.0, sr, total, 0.4)
    m = min(len(player), len(clicks))
    mix = (player[:m] + 0.4 * clicks[:m]).astype(np.float32)
    from sax_audio_engine import AudioEngine
    eng = AudioEngine()
    eng.set_filter_mode('normal')
    eng.set_a4(440.0)
    res = H.feed_engine(eng, mix)
    res.target_hz = 440.0
    assert res.emitted and res.dominant_midi == 69, (
        f"clicks corrupted the note read: midi={res.dominant_midi} notes={res.notes}")
    assert abs(res.cents_error()) < 3.0, (
        f"metronome clicks pulled the player's cents to {res.cents_error():+.2f}c")


def test_two_simultaneous_output_tones_each_keep_pitch_in_the_mix():
    """Two pitched outputs sounding together must each keep their own pitch in the
    COMBINED mix — additive mixing must not pull either off-frequency. peak_near at
    each target reads that target back out of the two-tone spectrum."""
    mb = 2048
    mixer = Mixer(max_block=mb)
    fa, fb = 220.0, 330.0                    # a fifth apart; neither in the other's ±70c window
    mixer.register(TestToneSource(fa, SR, mb, gain=0.3))
    mixer.register(TestToneSource(fb, SR, mb, gain=0.3))
    sig = H.render_mixer(mixer, 32768, block=mb)
    ea = H.cents(H.peak_near_hz(sig, fa, SR), fa)
    eb = H.cents(H.peak_near_hz(sig, fb, SR), fb)
    assert abs(ea) < 2.0, f"tone A {fa}Hz pulled to {ea:+.2f}c in the mix"
    assert abs(eb) < 2.0, f"tone B {fb}Hz pulled to {eb:+.2f}c in the mix"


# ---------------------------------------------------------------------------
# 14. RATE-INVARIANCE — pitch + timing must hold at a NON-default sample rate.
#     Invariant #1 (the engine hop is a constant DURATION, scaled in SAMPLES by
#     sr) and the drift-zero scheduler exist BECAUSE a real rate-scaling bug bit
#     at 96k/192k. Bare yin_pitch (test_yin_accuracy) and the drone (test_drone_
#     pitch) are already rate-checked; this covers the bench-distinctive END-TO-
#     END measurements that were single-rate: the FULL input pipeline (gate ->
#     YIN -> confirm -> emit), the rendered test-tone / pitch-pipe pitch, and the
#     rendered metronome grid. 48k is the baseline; 96k is the doubling that
#     catches a samples-vs-duration scaling bug.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sr", [48000, 96000])
def test_input_pipeline_pitch_is_rate_invariant(sr):
    res = H.detect_tone(440.0, kind='sax', sr=sr, hops=24)
    assert res.emitted and res.dominant_midi == 69, (
        f"@{sr}Hz full pipeline read midi={res.dominant_midi} (expected 69)")
    assert abs(res.cents_error()) < 1.0, (
        f"@{sr}Hz full pipeline cents {res.cents_error():+.2f}c")


@pytest.mark.parametrize("sr", [48000, 96000])
def test_test_tone_rendered_pitch_is_rate_invariant(sr):
    f = 440.0
    tone = TestToneSource(f, sr, 2048, gain=0.3)
    sig = H.render_stream(tone, sr // 3, block=2048)
    err = H.cents(H.fft_peak_hz(sig, sr), f)
    assert abs(err) < 1.5, f"test tone {f}Hz @{sr}Hz rendered {err:+.2f}c off"


@pytest.mark.parametrize("sr", [48000, 96000])
def test_pitch_pipe_rendered_pitch_is_rate_invariant(sr):
    midi, a4 = 69, 440.0
    src = PitchPipeSource(midi, sr, 2048, a4=a4)
    sig = H.render_stream(src, sr // 3, block=2048, warmup=2)
    err = H.cents(H.fft_peak_hz(sig, sr), pp.midi_to_freq(midi, a4))
    assert abs(err) < 1.5, f"pitch pipe midi={midi} @{sr}Hz rendered {err:+.2f}c off"


@pytest.mark.parametrize("sr", [48000, 96000])
def test_metronome_rendered_grid_is_rate_invariant(sr):
    mb = 1024
    mixer = Mixer(max_block=mb)
    metro = MetronomeController(mixer, sr, bpm=120)
    metro.start()
    interval = sr * 60.0 / 120
    sig = H.render_mixer(mixer, int(interval * 6), block=mb)
    onsets = H.onset_samples(sig, threshold=1e-2, min_gap=int(interval * 0.4))[:5]
    intervals = np.diff(onsets)
    assert len(onsets) >= 4, f"@{sr}Hz too few clicks: {onsets.tolist()}"
    assert np.all(np.abs(intervals - interval) <= 2), (
        f"@{sr}Hz metronome grid drifted: {intervals.tolist()} (interval {interval:.1f})")


# ---------------------------------------------------------------------------
# 15. CONCERT-PITCH (A4) REFERENCE end-to-end across a RANGE. The bench runs at
#     A4=440 almost everywhere, but orchestras tune to 442/443 and baroque
#     ensembles to 415 — and the reference must flow through BOTH the input cents
#     AND the rendered output pitch. The "feels off" scenario: you set 442 but the
#     app secretly tunes to 440, so every reading and every reference tone is ~8c
#     off. (test_sim has one a4=442 input case; test_pitch_pipes tests the freq
#     MATH — this is the end-to-end measurement across a range.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a4,detune_c", [
    (415.0, 0.0), (442.0, 10.0), (443.0, -10.0)],
    ids=["baroque-415-intune", "442+10c", "443-10c"])
def test_input_cents_are_relative_to_the_set_concert_pitch(a4, detune_c):
    """At a non-440 concert A, the EMITTED cents are measured against THAT
    reference: a tone detuned detune_c from A4@a4 reads ~detune_c (and the exact
    reference reads ~0). Proves the cents aren't silently hardcoded to 440."""
    f = a4 * 2.0 ** (detune_c / 1200.0)
    res = H.detect_tone(f, kind='sax', a4=a4, hops=24)
    res.target_hz = a4                              # the in-tune A4 at this concert pitch
    assert res.emitted and res.dominant_midi == 69, (
        f"A4@{a4}Hz {detune_c:+.0f}c: read midi {res.dominant_midi} (expected 69)")
    assert abs(res.cents_error() - detune_c) < 1.5, (
        f"A4@{a4}Hz {detune_c:+.0f}c: engine read {res.cents_error():+.2f}c")


@pytest.mark.parametrize("a4", [415.0, 442.0, 443.0])
def test_pitch_pipe_output_honors_concert_pitch(a4):
    """The reference tone the player tunes TO must itself follow the concert pitch:
    the A4 pitch-pipe pad rendered at concert A=a4 sounds at exactly a4 Hz."""
    src = PitchPipeSource(69, SR, 2048, a4=a4)
    sig = H.render_stream(src, 16384, block=2048, warmup=2)
    err = H.cents(H.fft_peak_hz(sig, SR), a4)       # A4 must render at a4 Hz, not 440
    assert abs(err) < 1.5, f"pitch-pipe A4 @ concert {a4}Hz rendered {err:+.2f}c off"


# ---------------------------------------------------------------------------
# 16. The PURE-SINE drone — the drone's sub-cent PRECISION reference. The SF2
#     voices (§6/§9) carry musical vibrato/chorus (organ steadiest, ~3c spread);
#     this synthesised voice (sax_drone.SineDroneSource — pure numpy) is
#     vibrato-free and exact, a tuning reference you can hold a note against.
#     Unlike §6/§9 it needs NO SoundFont, so it is an ALWAYS-ON gate (system-
#     python, no tsf importorskip) — the drone path's only always-runnable pitch
#     lock. Renders through the REAL DroneController/Mixer, then measures.
# ---------------------------------------------------------------------------
def test_pure_sine_drone_is_a_sub_cent_stability_reference():
    """The pure-sine drone holds A4 with no vibrato — spread far under the organ's
    <6c (§9) and centred sub-cent: the precision reference the sampled voices
    can't be."""
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id=D.SINE_VOICE_ID, volume=0.7)
    c.set_enabled(True)
    assert type(c.source).__name__ == "SineDroneSource", "sine id must build the sine source"
    sig = H.render_mixer(mixer, 180000, block=2048)[20000:]   # ~3.3 s steady
    true = D.midi_to_freq(69, 440.0)
    mean_c, spread_c = H.pitch_stability(sig, true, SR)
    assert spread_c < 0.5, f"pure-sine A4 spread {spread_c:.3f}c — should be vibrato-free"
    assert abs(mean_c) < 1.5, f"pure-sine A4 centre {mean_c:+.3f}c off"


@pytest.mark.parametrize("midi,name", [
    (45, "A2"), (57, "A3"), (69, "A4"), (76, "E5"), (81, "A5")])
def test_pure_sine_drone_pitch_is_exact_across_the_range(midi, name):
    """The sounded fundamental is the reference note at every pitch. The synth is
    exact (inc = 2*pi*f/sr); the residual is FFT bin resolution, widest at the
    lowest note (~46c/bin at 110 Hz, recovered to ~1c by parabolic interp), so the
    band is <2c — still 6x tighter than the rich-timbre SF2 gate (§6, <12c)."""
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id=D.SINE_VOICE_ID, volume=0.7)
    c.set_reference_midi(midi)
    c.set_enabled(True)
    assert c.source.active_midi == midi, f"sine must declare midi {midi}"
    sig = H.render_mixer(mixer, 49152, block=2048)[6000:]
    true = D.midi_to_freq(midi, 440.0)
    err = H.cents(H.fft_peak_hz(sig, SR), true)
    assert abs(err) < 2.0, f"pure-sine {name} ({true:.1f}Hz) -> {err:+.2f}c"


@pytest.mark.parametrize("a4", [415.0, 442.0, 443.0])
def test_pure_sine_drone_reference_follows_concert_pitch(a4):
    """A tuning reference must track the concert pitch: A4 on the pure-sine drone
    sounds at exactly the set a4 (baroque 415 / orchestral 442 / 443), not a
    hardcoded 440 (the SineDroneSource concert-pitch twin of §15)."""
    import sax_drone as D
    mixer = Mixer(max_block=2048)
    c = D.DroneController(mixer, SR, voice_id=D.SINE_VOICE_ID, volume=0.7, a4=a4)
    c.set_enabled(True)                                  # ref midi 69 = A4
    sig = H.render_mixer(mixer, 49152, block=2048)[6000:]
    err = H.cents(H.fft_peak_hz(sig, SR), a4)            # A4 must render at a4, not 440
    assert abs(err) < 1.0, f"pure-sine A4 @ concert {a4}Hz rendered {err:+.2f}c off"
