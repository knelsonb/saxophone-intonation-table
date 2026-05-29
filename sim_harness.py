"""Simulation + validation harness for the saxophone-intonation app.

Hardware-free tooling to SIMULATE both audio paths and VALIDATE behaviour:

  INPUT  (mic -> detection): feed_engine / detect_tone drive the engine's REAL
         hot path (silence gate -> YIN -> confirm/lock -> emit) with synthetic
         mic signals and capture the CONFIRMED notes — the input twin of the
         output renderers below.
  OUTPUT (sources -> speaker): render_source / render_stream / render_mixer
         pull MixerSources and the real Mixer offscreen (summing, duck, reap).

  * pitch accuracy   — FFT-peak AND YIN, with cents error vs a target
  * detection        — full engine pipeline: gated? locked? emitted note + cents
  * transients       — boundary discontinuities (clicks/pops), fade-in/out
  * allocation       — tracemalloc a hot-path callable (steady-state bytes)
  * drift            — per-block continuity of a rendered stream

Pure numpy + the app's own modules (NO Qt, NO sounddevice), so it imports and
runs under system python and the test venv alike. Used by the test_* accuracy
/ transient / alloc / sim nets and for ad-hoc validation during reliability work.

Design note — FFT/parabolic peak interpolation uses the vertex formula
  offset = 0.5*(a-c)/(a - 2b + c)
with the denominator sign that actually points at the vertex. (Getting this
sign wrong is exactly the bug that made yin_pitch read +8 cents sharp; this
harness exists partly to catch that class of error, so it must be right here.)
"""
from __future__ import annotations

import math
import tracemalloc

import numpy as np

DEFAULT_SR = 44100
DEFAULT_N = 16384  # the engine's real detection window (DEFAULT_BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Signal generators (float32, mono). amp is peak amplitude.
# ---------------------------------------------------------------------------
def sine(f: float, sr: int = DEFAULT_SR, n: int = DEFAULT_N,
         amp: float = 0.5, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return (amp * np.sin(2.0 * math.pi * f * t / sr + phase)).astype(np.float32)


def sax_like(f: float, sr: int = DEFAULT_SR, n: int = DEFAULT_N,
             amp: float = 0.7) -> np.ndarray:
    """Five-harmonic saxophone-ish tone (same recipe as the YIN nets)."""
    t = np.arange(n, dtype=np.float64)
    out = (0.50 * np.sin(2 * math.pi * 1 * f * t / sr)
           + 0.30 * np.sin(2 * math.pi * 2 * f * t / sr + 0.4)
           + 0.20 * np.sin(2 * math.pi * 3 * f * t / sr + 1.1)
           + 0.10 * np.sin(2 * math.pi * 4 * f * t / sr + 2.0)
           + 0.05 * np.sin(2 * math.pi * 5 * f * t / sr + 0.7))
    peak = float(np.max(np.abs(out))) or 1.0
    return (amp * out / peak).astype(np.float32)


def noisy(f: float, sr: int = DEFAULT_SR, n: int = DEFAULT_N,
          snr_db: float = 20.0, seed: int = 42, amp: float = 0.5) -> np.ndarray:
    s = sine(f, sr, n, amp).astype(np.float64)
    rms_s = float(np.sqrt(np.mean(s * s))) or 1.0
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    rms_n = float(np.sqrt(np.mean(noise * noise))) or 1.0
    scale = rms_s / (rms_n * (10.0 ** (snr_db / 20.0)))
    return (s + scale * noise).astype(np.float32)


def silence(n: int = DEFAULT_N) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Rendering MixerSource objects (protocol: render(out, frames, t0) additive)
# ---------------------------------------------------------------------------
def render_source(source, frames: int, t0: int = 0) -> np.ndarray:
    """Render one block of `frames` from a source into a fresh zero buffer."""
    out = np.zeros(frames, dtype=np.float32)
    source.render(out, frames, t0)
    return out


def render_stream(source, total: int, block: int = 2048,
                  warmup: int = 0) -> np.ndarray:
    """Render `total` samples from a source in `block`-sized chunks, advancing
    t0 each block (mirrors how the mixer pulls it). Discards the first
    `warmup` blocks (to skip an attack envelope), returns the concatenation of
    the rest. Captures real block-boundary behaviour for transient analysis."""
    out = []
    t0 = 0
    nblocks = (total + block - 1) // block
    for i in range(warmup + nblocks):
        buf = np.zeros(block, dtype=np.float32)
        source.render(buf, block, t0)
        t0 += block
        if i >= warmup:
            out.append(buf.copy())
    return np.concatenate(out)[:total] if out else np.zeros(0, dtype=np.float32)


def render_mixer(mixer, total: int, block: int = 2048) -> np.ndarray:
    """Render `total` samples from a real ``Mixer`` through its ACTUAL output
    path — every registered source summed, ducked, and auto-reaped exactly as
    the output stream pulls it. Sources must be ``register()``-ed beforehand.
    Returns the concatenated mono output (the true thing a speaker would get),
    so output checks cover mixing/reap, not just one source in isolation."""
    out = []
    remaining = int(total)
    while remaining > 0:
        nb = min(block, remaining)
        buf = np.zeros(nb, dtype=np.float32)
        mixer.render(buf, nb)
        out.append(buf.copy())
        remaining -= nb
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pitch measurement
# ---------------------------------------------------------------------------
def fft_peak_hz(buf: np.ndarray, sr: int = DEFAULT_SR) -> float:
    """Dominant spectral-peak frequency (Hz), Hann-windowed + parabolic
    interpolation. Independent of YIN — used to cross-check it."""
    x = np.asarray(buf, dtype=np.float64)
    n = x.shape[0]
    if n < 4 or not np.any(x):
        return 0.0
    sp = np.abs(np.fft.rfft(x * np.hanning(n)))
    k = int(np.argmax(sp))
    if 1 <= k < sp.shape[0] - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        d = a - 2.0 * b + c
        if d:
            k = k + 0.5 * (a - c) / d           # vertex offset (correct sign)
    return float(k) * sr / n


def yin_hz(buf: np.ndarray, sr: int = DEFAULT_SR) -> tuple[float, float]:
    """(freq_hz, aperiodicity) from the engine's own yin_pitch."""
    from sax_audio_engine import yin_pitch
    return yin_pitch(np.asarray(buf, dtype=np.float32), sr)


def cents(detected: float, target: float) -> float:
    """Cents of `detected` relative to `target` (NaN if either non-positive)."""
    if detected <= 0 or target <= 0:
        return float('nan')
    return 1200.0 * math.log2(detected / target)


# ---------------------------------------------------------------------------
# Transient / discontinuity detection (clicks & pops)
# ---------------------------------------------------------------------------
def max_abs(buf: np.ndarray) -> float:
    return float(np.max(np.abs(buf))) if buf.size else 0.0


def rms(buf: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(buf, dtype=np.float64)))) if buf.size else 0.0


def boundary_jump(prev_block: np.ndarray, next_block: np.ndarray) -> float:
    """|discontinuity| at the seam between two consecutive rendered blocks —
    the last sample of `prev_block` to the first of `next_block`. A click is a
    jump much larger than the signal's natural per-sample step."""
    if prev_block.size == 0 or next_block.size == 0:
        return 0.0
    return abs(float(next_block[0]) - float(prev_block[-1]))


def edge_level(buf: np.ndarray, head: bool, n: int = 1) -> float:
    """Peak |amplitude| of the first (head) or last (tail) `n` samples — used
    to confirm a source starts/ends near zero (faded), not mid-cycle."""
    if buf.size == 0:
        return 0.0
    seg = buf[:n] if head else buf[-n:]
    return float(np.max(np.abs(seg)))


def has_fade_in(buf: np.ndarray, ms: float = 5.0, sr: int = DEFAULT_SR,
                ratio: float = 0.5) -> bool:
    """True if the signal ramps UP over the first `ms` — the mean |amp| of the
    first half-window is < `ratio` * the mean |amp| of the steady body. Catches
    a missing attack (which would start at full amplitude → click)."""
    nwin = max(2, int(sr * ms / 1000.0))
    if buf.size < nwin * 4:
        return True  # too short to judge; don't false-fail
    head = float(np.mean(np.abs(buf[:nwin // 2])))
    body = float(np.mean(np.abs(buf[nwin * 2:nwin * 4]))) or 1e-12
    return head < ratio * body


def has_fade_out(buf: np.ndarray, ms: float = 5.0, sr: int = DEFAULT_SR,
                 ratio: float = 0.5) -> bool:
    """True if the signal ramps DOWN over the last `ms` (release present)."""
    nwin = max(2, int(sr * ms / 1000.0))
    if buf.size < nwin * 4:
        return True
    tail = float(np.mean(np.abs(buf[-nwin // 2:])))
    body = float(np.mean(np.abs(buf[-nwin * 4:-nwin * 2]))) or 1e-12
    return tail < ratio * body


# ---------------------------------------------------------------------------
# Allocation measurement (hot-path no-alloc gates)
# ---------------------------------------------------------------------------
def alloc_bytes(fn, *args, iterations: int = 200, warmup: int = 5,
                **kwargs) -> float:
    """Average bytes allocated per call of `fn` in steady state. Warms up
    first (one-time allocations), then tracemalloc-measures `iterations` calls
    and returns total_alloc / iterations. Use for 'no unbounded alloc in the
    hot path' gates: a steady-state-clean callable returns ~0."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(iterations):
        fn(*args, **kwargs)
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    total = sum(s.size_diff for s in snap1.compare_to(snap0, 'filename'))
    return max(0.0, total / float(iterations))


# ---------------------------------------------------------------------------
# Engine INPUT-path simulation — the FULL detection pipeline, device-free.
#
# yin_hz() above tests the bare pitch function; this drives the real engine
# hot path (AudioEngine.feed_input_frames -> silence gate -> YIN -> confirm /
# lock / median -> emit), capturing the CONFIRMED notes the live app would
# show. No PortAudio, no Qt: we synthesize the mic and inject our own note
# sink. This is the input twin of render_mixer()'s output simulation.
# ---------------------------------------------------------------------------
class _Emit:
    """Records one signal's .emit(...) payloads into a shared list."""
    def __init__(self, store: list) -> None:
        self._store = store

    def emit(self, midi, freq, cents) -> None:
        self._store.append((int(midi), float(freq), float(cents)))


class _NoteSink:
    """Stand-in for the engine's Qt ``signals`` object. The engine emits a
    confirmed note via ``signals.note_detected.emit(midi, freq, cents)``;
    recording it lets a headless sim capture the engine's REAL output (the
    post-confirm/median note), not just the raw per-hop YIN estimate."""
    def __init__(self) -> None:
        self.notes: list[tuple[int, float, float]] = []
        self.note_detected = _Emit(self.notes)


class DetectionResult:
    """Outcome of feeding a signal through the engine detection pipeline."""
    def __init__(self, engine, notes, target_hz=None) -> None:
        self.notes = list(notes)                  # [(midi, freq, cents), ...]
        self.last_freq = float(engine.last_freq)  # raw last-hop YIN estimate
        self.last_rms_db = float(engine.last_rms_db)
        self.locked_midi = engine.last_locked_midi
        self.target_hz = target_hz

    @property
    def emitted(self) -> bool:
        """True if the pipeline confirmed + emitted at least one note."""
        return bool(self.notes)

    @property
    def dominant_midi(self):
        if not self.notes:
            return None
        from collections import Counter
        return Counter(m for m, _, _ in self.notes).most_common(1)[0][0]

    @property
    def median_freq(self) -> float:
        fs = sorted(f for _, f, _ in self.notes)
        return fs[len(fs) // 2] if fs else 0.0

    @property
    def median_cents(self) -> float:
        cs = sorted(c for _, _, c in self.notes)
        return cs[len(cs) // 2] if cs else float('nan')

    def cents_error(self) -> float:
        """Cents of the median detected freq vs the target tone (NaN if no
        target was set or nothing was detected)."""
        if not self.target_hz or not self.notes:
            return float('nan')
        return cents(self.median_freq, self.target_hz)


def feed_engine(engine, signal, block: int = DEFAULT_N, capture: bool = True):
    """Feed a 1-D float32 ``signal`` into the engine's REAL input hot path in
    ``block``-sized chunks (as PortAudio delivers it), capturing every
    confirmed note. Returns a DetectionResult. No device, no Qt."""
    sig = np.asarray(signal, dtype=np.float32)
    sink = _NoteSink()
    if capture:
        engine.signals = sink
    i, n = 0, sig.shape[0]
    while i < n:
        engine.feed_input_frames(sig[i:i + block])
        i += block
    return DetectionResult(engine, sink.notes)


def detect_tone(freq, *, kind='sax', mode='normal', a4=440.0, mic_gain=1.0,
                snr_db=None, hops=20, sr=DEFAULT_SR, amp=0.5):
    """Synthesize a steady tone and run it through a FRESH engine's detection
    pipeline for ``hops`` block-sized passes — enough for the silence gate to
    fill, then the confirm/lock machinery to emit. Returns a DetectionResult
    with ``target_hz`` set, so ``.cents_error()`` is meaningful.

    kind : 'sax' (5-harmonic), 'sine', or 'noisy' (sine + noise at snr_db).
    mode : engine response preset ('fast' / 'normal' / 'slow').
    """
    from sax_audio_engine import AudioEngine
    eng = AudioEngine()
    eng.set_filter_mode(mode)
    eng.set_a4(a4)
    eng.set_mic_gain(mic_gain)
    total = DEFAULT_N * max(1, int(hops))
    if kind == 'sine':
        sig = sine(freq, sr, total, amp)
    elif kind == 'noisy' or snr_db is not None:
        sig = noisy(freq, sr, total,
                    snr_db=(snr_db if snr_db is not None else 20.0), amp=amp)
    else:
        sig = sax_like(freq, sr, total, amp)
    res = feed_engine(eng, sig, block=DEFAULT_N)
    res.target_hz = freq
    return res


if __name__ == '__main__':  # quick self-check / demo
    import sax_audio_engine as eng  # noqa
    print("— bare yin_pitch (single window) —")
    for f in (110.0, 440.0, 880.0, 1760.0):
        buf = sax_like(f)
        fp = fft_peak_hz(buf)
        yf, ap = yin_hz(buf)
        print(f"{f:8.1f} Hz: fft {fp:8.2f} ({cents(fp, f):+.2f}ct)  "
              f"yin {yf:8.2f} ({cents(yf, f):+.2f}ct) ap={ap:.2e}")
    print("— full engine detection pipeline (feed_input_frames) —")
    for f in (110.0, 220.0, 440.0, 880.0):
        r = detect_tone(f, kind='sax')
        print(f"{f:8.1f} Hz: midi={r.dominant_midi} "
              f"median={r.median_freq:8.2f} ({r.cents_error():+.2f}ct) "
              f"emitted={r.emitted}")
