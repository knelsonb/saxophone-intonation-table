"""YIN pitch estimation, vectorised.

One function, ``yin_pitch(sig, sr) -> (hz, aperiodicity)``, used by the
detection hop. It is sample-rate agnostic: every bound is a lag in samples
derived from ``sr``, so the same code serves 44.1 kHz and 192 kHz.

The cost per hop is two real FFTs plus a handful of O(tmax) array passes. The
FFT is zero-padded only as far as a linear (non-circular) autocorrelation needs
on the lags YIN reads, N + tmax, rounded up to a length with small prime
factors. Padding to a power of two at or above 2N, as earlier versions did,
roughly doubled the transform for no change in the result.

The lag is interpolated on the raw difference function and, for short
periods, refined over several periods (see REFINE_LAG); a dip whose minimum
falls between two lags counts as below the threshold when its interpolated
minimum is. Together these hold every note from A0 to C8 within 0.6 cents of
a synthetic tone at 44.1 and 48 kHz.
"""
from __future__ import annotations

import numpy as np

# Detection band. 25.5 Hz sits below piano A0 at the lowest selectable concert
# pitch (A4 = 430 Hz puts A0 at 26.88 Hz), so the lag search always reaches the
# period of the lowest catalog note with a sample to spare for interpolation.
MIN_FREQ = 25.5
# 4500 Hz covers C8 at A4 = 450 Hz plus about 50 cents (the C8/C#8 rounding
# boundary is near 4409 Hz there). The precise upper bound is the MIDI-range
# gate in the detector; this one only keeps the lag search away from lags of a
# few samples, where YIN has no resolution left.
MAX_FREQ = 4500.0
YIN_THRESHOLD = 0.12
# Short periods are refined on the dip at the multiple of the period nearest
# this many lags. Parabolic interpolation misplaces a dip by a roughly fixed
# fraction of a lag whatever the period, so measuring k periods at once
# divides the error by k: at C8 (about 10.5 lags at 44.1 kHz) the residual
# falls from up to 7.5 cents to about 0.5. Periods longer than this keep their
# first dip (k = 1), so the low and middle registers are unchanged.
REFINE_LAG = 128.0


def next_fast_len(n: int) -> int:
    """Smallest 2**a * 3**b * 5**c >= n (a "5-smooth" FFT length).

    numpy's pocketfft is fastest on lengths with small prime factors. This is
    the scipy.fft.next_fast_len idea without the scipy dependency; the search
    is over at most a few hundred candidates for audio-sized n.
    """
    n = int(n)
    if n <= 1:
        return 1
    best = 1 << (n - 1).bit_length()        # the power of two is always a candidate
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            # Smallest power of two that lifts p35 to at least n.
            q = -(-n // p35)                   # ceil(n / p35)
            cand = p35 << (q - 1).bit_length()
            if cand < best:
                best = cand
            p35 *= 3
        p5 *= 5
    return best


def yin_pitch(sig: np.ndarray, sr: int,
              fmin: float = MIN_FREQ, fmax: float = MAX_FREQ,
              thr: float = YIN_THRESHOLD) -> tuple[float, float]:
    """Estimate the fundamental of ``sig`` (one detection window) at rate ``sr``.

    Returns ``(hz, aperiodicity)``. ``hz`` is 0.0 when no lag in range could be
    measured; aperiodicity is the cumulative-mean-normalised difference at the
    chosen lag (0 = perfectly periodic, about 1 = noise).
    """
    N = len(sig)
    tmin = max(1, int(sr / fmax))
    tmax = min(N // 2, int(sr / fmin))
    if tmax <= tmin:
        return 0.0, 1.0
    x = np.asarray(sig, dtype=np.float64)   # fp64 for stable energy sums at large N
    # Linear autocorrelation r[0..tmax] via FFT. A circular correlation of
    # length M aliases r[tau - M] onto r[tau]; that term is zero for every
    # tau <= tmax once M >= N + tmax.
    M = next_fast_len(N + tmax)
    X = np.fft.rfft(x, M)
    r = np.fft.irfft(X * np.conj(X), M)[:tmax + 1]
    # Difference function d(tau) = E1(tau) + E2(tau) - 2 r(tau), with
    #   E1(tau) = sum of x[j]^2 for j in [0, N - tau)  (left window energy)
    #   E2(tau) = sum of x[j]^2 for j in [tau, N)      (right window energy).
    # The shortcut 2 (r(0) - r(tau)) assumes both windows carry the same
    # energy, which a windowed note onset or decay does not.
    S = np.concatenate(([0.0], np.cumsum(x * x)))       # prefix sums, length N+1
    t = np.arange(tmax + 1)
    diff = S[N - t] + (S[N] - S[t]) - 2.0 * r
    # Cumulative-mean-normalised difference d'(tau) = d(tau) * tau / sum d(1..tau).
    # A running sum that is not positive (silence, or rounding on a pure DC
    # input) carries no periodicity information and reads as 1.
    run = np.cumsum(diff[1:])
    cmnd = np.ones(tmax + 1)
    np.divide(diff[1:] * t[1:], run, out=cmnd[1:], where=run > 0)
    # Absolute threshold: the first lag in [tmin, tmax) below thr, then walk down
    # that dip to its local minimum. NaN compares false on both tests, which
    # stops the walk and skips the lag, as a scalar loop would.
    below = np.flatnonzero(cmnd[tmin:tmax] < thr)
    tau = -1
    if below.size:
        tau = tmin + int(below[0])
        stops = np.flatnonzero(~(cmnd[tau + 1:tmax] < cmnd[tau:tmax - 1]))
        tau = tau + int(stops[0]) if stops.size else tmax - 1
    # At the top of the range a period is about ten lags, and a dip whose true
    # minimum falls between two lags can sit above thr at both. The parabola
    # through each local minimum estimates the value between the samples; the
    # first dip whose estimate is below thr is the period, even ahead of the
    # first lag that is below thr on its own (taking that later dip read C8 an
    # octave low, or not at all).
    lo = max(tmin, 1)
    hi = tau if tau >= 0 else tmax - 1
    if hi > lo:
        s0, s1, s2 = cmnd[lo - 1:hi - 1], cmnd[lo:hi], cmnd[lo + 1:hi + 1]
        curv = s0 + s2 - 2.0 * s1
        dip = (s0 > s1) & (s2 >= s1) & (curv > 0)
        vertex = s1 - (s0 - s2) ** 2 / (8.0 * np.where(dip, curv, 1.0))
        early = np.flatnonzero(dip & (vertex < thr))
        if early.size:
            tau = lo + int(early[0])
    if tau < 0:
        tau = tmin + int(np.argmin(cmnd[tmin:tmax]))
    mv = float(cmnd[tau])
    lag = float(tau)
    if 1 < tau < tmax - 1:
        # Parabolic interpolation of the raw difference function d(tau), as in
        # the YIN paper. d'(tau) carries the factor tau / mean(d), which tilts
        # its dip by a few percent per lag where the period is short, and its
        # vertex read up to 10 cents sharp in the top octave.
        # Offset 0.5*(s0-s2)/(s0-2*s1+s2). The denominator SIGN matters:
        # inverting it (2*s1-s0-s2, the pre-v0.11 bug) moves the lag the wrong
        # way and reads about 8 cents sharp across the mid and high range
        # (A440 -> 441.99 Hz). test_yin_accuracy pins it.
        s0, s1, s2 = diff[tau - 1], diff[tau], diff[tau + 1]
        d = s0 + s2 - 2 * s1
        if d > 0:
            lag += 0.5 * (s0 - s2) / d
        # Refine on the k-th period's dip (see REFINE_LAG). The first estimate
        # is good to a few hundredths of a lag, so the k-th dip lies within a
        # lag of k * lag; it is used only if it is a true local minimum there.
        k = max(1, int(round(REFINE_LAG / lag)))
        if k > 1 and k * lag + 3 < tmax:
            c = int(round(k * lag))
            j = c - 1 + int(np.argmin(diff[c - 1:c + 2]))
            s0, s1, s2 = diff[j - 1], diff[j], diff[j + 1]
            d = s0 + s2 - 2 * s1
            if d > 0 and s1 <= s0 and s1 <= s2:
                lag = (j + 0.5 * (s0 - s2) / d) / k
        # Aperiodicity is the dip's minimum, interpolated the same way on
        # d'(tau) and kept within [0, d'(tau)].
        c0, c1, c2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        cd = c0 + c2 - 2.0 * c1
        if cd > 0:
            mv = min(mv, max(0.0, float(c1 - (c0 - c2) ** 2 / (8.0 * cd))))
    # Clamp lag >= 1 before dividing, so a degenerate near-flat dip cannot turn
    # sr/lag into a nonsense high frequency.
    return (sr / lag if lag >= 1.0 else 0.0), mv
