# Virtual Verification Bench

How the tuner's audio is verified **without hardware** — the pitch *and* timing
of every input and output, measured from the actual rendered / detected signal in
pure numpy (no PortAudio, no Qt, no soundcard).

> Standard: *"nothing should ever feel off, because we should be able to verify
> the timing and pitch of every input and output virtually before sending this
> into the wild."*

| | |
|---|---|
| Bench | `test_virtual_bench.py` — 104 tests, 17 sections (`# 0` … `# 16`) |
| Toolkit | `sim_harness.py` — signal generators, renderers, pitch/timing/level meters, engine drivers |
| Runs on | system `python3` (numpy) for the pure paths — including the §16 **pure-sine drone**; a venv with `tinysoundfont` only for the SF2 drone sections (§6, §9 `importorskip` cleanly otherwise) |
| Principle | render/feed the **real** sources + engine, then **measure** the result — not "assert the math", but "does it actually sound/read at the right pitch, at the right time" |

**Two kinds of hop.** In the live app the engine runs one detection pass per
*engine hop*: 2048 new samples, about 46 ms at 44.1 kHz, analysed in a trailing
16384-sample window (see `docs/response-modes.md`). The bench drives the engine
through `feed_input_frames` one *bench hop* at a time: one 16384-sample block,
about 372 ms of audio at 44.1 kHz, which triggers one detection pass on entirely
fresh samples. Bench hop counts (§5, the `hops=` arguments) therefore count
detection passes; they are not a measure of live latency in milliseconds.

**When it runs.** The v1.2.0 release workflow (`.github/workflows/release.yml`)
runs the full pytest suite, this bench included, before it builds any binary;
its test job installs `tinysoundfont`, so the SF2 drone sections run there too.

Related: `docs/response-modes.md` (engine filter presets).

---

## The five axes

Each source/path is verified along the axes that matter for "feels off". The 17
bench sections map onto them:

### 1. Per-source isolation — every source's pitch + timing, measured alone
| § | What | Measured |
|---|------|----------|
| 0 | onset primitive itself | bursts at known sample positions (so every timing check below is trustworthy) |
| 1 | metronome BPM grid (60/120/200) | rendered click onsets land on the grid, **no drift** (≤2 samples per beat + cumulative) |
| 2 | test-tone pitch (110–1760 Hz) | FFT readback of the rendered tone < 1.5 c |
| 3 | pitch-pipe pad pitch (A2–A6) | rendered pad at `midi_to_freq(midi, a4)` < 1.5 c |
| 4 | **input** pitch — 5 notes × 3 timbres (sine/sax/noisy) | full engine pipeline (gate → YIN → confirm → emit) reads the right note; clean < 0.6 c, 20 dB-SNR noisy < 2 c |
| 5 | **input** detection latency | first confirmed emit within ≤ 12 bench hops (detection passes) — a coarse bound that catches a stuck confirm/lock path; live first-reading latency in ms is in `docs/response-modes.md` |
| 6 | drone pitch — 5 notes × 3 voices (organ/cello/tenorsax) | `peak_near` reads the fundamental of the rich timbre < 12 c |
| 7 | deck playback | the played-back take is at the right pitch (< 2 c, FFT) and the output is silent after the take ends |
| 8 | metronome accent grid — 5 tempos (40–200) × 4 time sigs (2/4,3/4,4/4,6/8) | every click on the grid + the downbeat is the loud accent |
| 9 | drone pitch **stability** | the organ is the steadiest SF2 reference (A4 spread < 6 c, centre < 4 c) |
| 16 | drone pitch — the **pure-sine** precision voice (no SF2, always-on) | vibrato-free + exact: A4 spread < 0.5 c & centre < 1.5 c; < 2 c across A2–A5; follows concert A4 (415/442/443) — the SF2 voices' sub-cent twin |

### 2. Interactions — sources running *together* (where "feels off" hides)
| § | Scenario | Verified |
|---|----------|----------|
| 10 | drone sounding **while** the tuner is live | the drone's note is vote-excluded (D3 mic-coordination) so it can't fool the tuner — with a **negative control** proving the rejection is real |
| 13 | metronome + tuner; two outputs at once | unpitched click bleed never false-triggers a phantom note + never corrupts a live read (< 3 c through the bleed); two pitched outputs each keep pitch in the additive mix |

### 3. Resilience — survives a device switch mid-session
| § | Scenario | Verified |
|---|----------|----------|
| 11 | output device swapped while the metronome runs (Option-B `resync`) | drift-free grid on **each** side of the switch + a prompt re-attack on the new stream |

### 4. Self-verification — the bench has *teeth* (it can turn red)
| § | What | Why |
|---|------|-----|
| 0, 10-control, 12 | inject a **known** fault and confirm the metric tracks it | a bench that only ever sees correct inputs and passes is unproven. §12 proves a known detune reads as ~that many cents (fft < 2.5 c, engine < 3 c, `peak_near` < 6 c) and injected drift surfaces well past the no-drift gate |

The §1 gate renders the *real* metronome, so it is wired to production code. Its
grid has a limit: at the bench rate of 48 kHz, 60 / 120 / 200 BPM give whole-sample
beat intervals (48000, 24000, 14400 samples), and so do the tempos in §8, §11 and
§14. A regression that rounds or truncates the beat interval drifts only when the
interval has a fractional part, so the bench grid cannot see it. The test that
catches that class is `test_metronome.py::test_beat_onset_round_once_is_bounded_and_does_not_drift`
(130 BPM at 44.1 kHz, 20353.8 samples per beat, with a rounded-accumulator
control that must drift).

### 5. Invariance — pitch + timing hold across sample rate *and* concert pitch
| § | What | Why |
|---|------|-----|
| 14 | full input pipeline + test-tone + pitch-pipe + metronome grid at **48 k and 96 k** | the drift-zero scheduler and the constant-duration engine hop exist *because* a real rate-scaling bug bit at 96 k / 192 k. Scope: §14 runs YIN and the confirm path at 96 kHz, but `detect_tone` sets `eng.samplerate` and feeds fixed 16384-sample blocks, so the engine's rate-dependent hop and window sizing (computed when a stream opens) is outside what it exercises |
| 15 | input cents + rendered pitch-pipe output at concert **A4 = 415 / 442 / 443** | the concert-pitch reference must flow through both the emitted cents and the reference tone the player tunes to — not silently hardcoded to 440 (orchestra-at-442, baroque-at-415) |

---

## The `sim_harness.py` toolkit

Hardware-free building blocks the bench composes from:

- **Generators** — `sine`, `sax_like` (5 harmonics), `noisy` (sine + SNR noise), `silence`.
- **Renderers** — `render_source` / `render_stream` (one MixerSource) · `render_mixer` (the full additive mix), block by block as PortAudio would deliver it.
- **Pitch meters** — `fft_peak_hz` (Hann + parabolic interp) · `peak_near_hz` (target-aware peak; robust on rich SF2 timbres where the global FFT peak lands on a partial) · `yin_hz` · `cents` · `pitch_stability` (mean + spread across windows = vibrato extent).
- **Timing meters** — `onset_samples` / `onset_intervals` (burst onset detection).
- **Level / transient meters** — `max_abs`, `rms`, `boundary_jump`, `edge_level`, `has_fade_in`, `has_fade_out`.
- **Allocation gate** — `alloc_bytes` (tracemalloc delta; the no-alloc-in-render check).
- **Engine drivers** — `feed_engine` (push a signal through the real input hot path) · `detect_tone` (synth → fresh-engine pipeline → `DetectionResult`) · `detection_latency_hops` · `detect_with_sounding_output` (detection *while* the output sounds given MIDIs — the §10 coordination driver).

`DetectionResult` exposes `emitted`, `dominant_midi`, `median_freq`, `median_cents`,
`cents_error()` — the engine's confirmed output, not raw YIN.

---

## How to run

```bash
# pure paths (metronome, test-tone, pitch-pipe, input, deck, coordination, …)
python3 -m pytest test_virtual_bench.py -q

# include the drone sections (§6, §9) — needs a venv with tinysoundfont
#   (they skip cleanly via importorskip if it's absent)
python -m pytest test_virtual_bench.py -q          # from the project venv
```

`sim_harness.py` also runs as a script (`python3 sim_harness.py`) for a quick
YIN-vs-FFT vs full-pipeline readout across a few frequencies.

---

## Tolerance budget (what "right" means)

| Quantity | Budget | Where |
|----------|--------|-------|
| Rendered output pitch (pure sine: test-tone, pitch-pipe) | < 1.5 c | §2, §3, §14 |
| Detected input pitch (clean sine/sax) | < 0.6 c | §4 |
| Detected input pitch (20 dB-SNR noisy) | < 2 c | §4 |
| Detected input pitch (rate-invariance) | < 1 c | §14 |
| Drone pitch (rich SF2 timbre via `peak_near`) | < 12 c | §6 |
| Drone stability (organ A4 spread / centre) | < 6 c / < 4 c | §9 |
| Drone pitch (pure-sine voice: range / A4 stability) | < 2 c / < 0.5 c | §16 |
| Player pitch through metronome-click bleed | < 3 c | §13 |
| Metronome beat spacing (per-beat + cumulative) | ≤ 2 samples | §1, §8, §11, §14 |
| Detection latency | ≤ 12 bench hops (detection passes) | §5 |

Cents are displayed to the user as **signed integers only** — a real sax tone
vibratos by more than a cent, so decimal cents claim resolution the reading can't
hold (`format_cents`; locked by `test_format_cents.py`).

---

## Findings & decisions banked here

- **The drone is a *musical* reference, not a precision one.** The sampled SF2
  voices carry baked-in vibrato/chorus — measured A4 pitch spread: strings ~21 c,
  tenorsax ~14 c, warmpad ~12 c, cello ~7 c; only the **organ** is steady (~3 c).
  The sub-cent precision references are the pure-sine **test-tone** and
  **pitch-pipe** (§2, §3) — and now a vibrato-free pure-sine **drone** voice too
  (§16, `SineDroneSource`; A4 spread ~0 c, follows concert pitch). It needs no
  SoundFont, so it is the drone path's one *always-on* precision gate, and gives
  the player a drone reference they can hold a note against. (Surfaced in the SETUP
  voice picker as "Pure Sine"; the 5-voice Android catalog `DRONE_PRESETS` is left
  untouched — it's a desktop-only synthesised voice, `DRONE_SYNTH_VOICES`.)
- **Device-switch resync re-attacks; it does not phase-preserve.** A switch has an
  unknown silent gap, so `resync()` re-anchors the next beat at the start of the
  new clock and continues the beat *index*/accent — it does not preserve a
  sub-beat phase across silence no one hears (§11; `sax_metronome` resync).
- **`detect_tone(sr=…)` configures the whole pipeline.** Building §14 surfaced (and
  fixed) a latent harness incoherence: the `sr` kwarg set tone-generation but not
  the engine's detection rate, so it only behaved at the 44.1 k default. It now
  sets `eng.samplerate` too.
- **Audio-visual sync has no per-beat drift seam — by design.** Tracing the GUI
  side of "feels off": the metronome's audible click is sample-accurate (scheduled
  on the mixer clock, §1/§8), but there is **no per-beat visual flash** to drift
  against it. The metronome's only on-screen surface is a *running-state dot*
  (green = running), updated through `on_state_changed`, which always fires on the
  **GUI thread** via the existing repaint/poll tick — never from the render-thread
  beat closure (`sax_deck.py`: "so the GUI never needs a `QTimer.singleShot`
  marshal"). The tuner needle/visualizer pull the engine's real `_buf`/detections
  at ~30 fps, so their only audio→visual offset is *bounded detection latency*
  (§5; in the live app about 0.2–0.65 s depending on the response mode) plus one refresh tick — lag, not accumulating drift. There is
  thus nothing sample-synced on screen for the hardware-free bench to measure; the
  requirement is met by the *absence* of a per-beat visual claim, not by syncing
  one. (Adding a per-beat visual indicator later would reopen this — it would need
  to read the same beat the click sounds from, not a separate GUI timer.)

---

## Extending the bench (the disciplines)

When you add a source, path, or scenario:

1. **Measure the signal, don't assert the math.** Render/feed the real thing and
   read the pitch/timing back out.
2. **Prove the new gate has teeth.** Pair it with a negative control or a
   fault-injection (§0, §10-control, §12) — a gate that can't turn red is a rubber
   stamp. For a detection-interaction test, feed enough hops that a *real* signal
   would emit, or "the bad input didn't emit" passes for the wrong reason.
3. **Check the interaction surface, not just the unit.** "Comprehensive on each
   source in isolation" is *not* "comprehensive on the system" — enumerate the
   combinations the sources actually run in together.
4. **Vary the sample rate** for anything pitch- or time-sensitive (48 k baseline,
   96 k doubling).
