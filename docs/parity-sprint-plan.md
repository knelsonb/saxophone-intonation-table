# Saxophone Intonation — Desktop Feature-Parity Sprint Plan

**Goal:** bring `saxophone-intonation-table` (PyQt6 desktop, v0.6.2) up to feature
parity with `saxophone-intonation-android` (React Native / Expo, v1.1.0).

**Scope direction is one-way.** Parity means the desktop gains what Android has.
The desktop already *exceeds* Android in several areas (statistical intonation
table with per-note mean ± σ, CSV slice modes, PDF/TXT export, autotune A4
detection, diagnostics/spectrum panel). Those are not on the table for removal or
simplification — preserve them. We are closing Android-only gaps, not
homogenizing the two apps.

---

## 1. Current state, verified against source

### What the desktop already has (parity already met)

| Area | Desktop status |
|---|---|
| Real-time tuner readout | `TunerWidget` — large cents/needle display, live engine SR, response-mode-aware precision |
| Intonation table | Live statistical matrix, mean ± σ per note (richer than Android) |
| Instruments | `sax_instruments.py` — sax / clarinet / flute / horn families with transposition (not just Eb/Bb/C) |
| A4 / concert pitch | 430–450 Hz, plus autotune detection from measured data |
| Pitch detection | YIN + FFT-YIN, ring buffer, RMS hardening, response modes (`fast`/`normal`/`slow`) |
| Export | TXT, PDF, CSV (multiple slice modes), chart PNG, JSONL persistence |
| Import | Raw-CSV resume/compare |
| Horn nickname | `config.last_nickname` / per-instrument `nickname`, used by CSV slicing |
| Range editor | Added Phase-4 |
| i18n | German / English (`sax_i18n.STRINGS`) |
| Device handling | Selection, host-API filtering, hot-plug recovery banner |

### Android-only features the desktop lacks (the parity gap)

1. **Audio output of any kind.** The desktop engine is **input-only** —
   `sd.InputStream`, no `OutputStream` anywhere in the tree. This is the keystone
   constraint: metronome, drone, deck-playback, and pitch-pipes all need an output
   path that does not currently exist.
2. **Metronome** — Android: BPM 30–300, time sigs 2/4 · 3/4 · 4/4 · 6/8, tap
   tempo, accent on downbeat, click volume, running indicator.
3. **Drone** (Android v1.1) — TinySoundFont + GeneralUser-GS SF2, 5 curated
   presets + full 128 GM voice catalog, pitch (semitone) offset, volume,
   mute-coordination with the live mic.
4. **Tape deck** — record mic to a take, play it back, share/export; modes
   `idle → recording → have-take → playing`.
5. **Pitch pipes** — reference-tone generator over a chromatic reference octave
   (Android `pitchTones.ts`). Shares synthesis with the drone.
6. **Theme switching** — Android has `dark` / `night` / `light` plus a night-tuning
   mode; desktop is dark-only.
7. **A navigation shell** to house the new surfaces. Android uses a 4-tab bottom
   bar (TUNER / METRO / DECK / SETUP). Desktop is a single-window splitter.

### Parity matrix → sprint mapping

| Android feature | Desktop today | Closes in |
|---|---|---|
| Audio output / mixer | absent | Sprint 1 |
| Tab/nav shell | single-window | Sprint 1 |
| Metronome | absent | Sprint 2 |
| Drone (SF2, 128 GM) | absent | Sprint 3 |
| Pitch pipes | absent | Sprint 3 |
| Tape deck record/playback | absent | Sprint 4 |
| Theme dark/night/light | dark only | Sprint 5 |
| Drone voice picker in settings | absent | Sprint 3 (engine) / 5 (settings polish) |
| Tuner / table / instruments / A4 / export | present (meets or exceeds) | — |

---

## 2. Architecture foundation (the part that gates everything)

Three of the four new tabs are output consumers. Build the output path **once**,
correctly, and reuse it. The design that keeps this BIFL:

- **One full-duplex `sd.Stream`** (input + output on the same callback), or a
  dedicated `sd.OutputStream` alongside the existing input stream if device
  constraints force separate streams. Duplex is preferred — one clock, one
  callback, no drift between mic-read and click-emit.
- **A numpy software mixer** owned by the engine. Sources (metronome, drone,
  pitch pipe, deck playback) register as pull-based generators; the output
  callback sums their per-block contributions, clamps, and writes. Sample-accurate
  scheduling lives here (metronome needs it).
- **Callback discipline is non-negotiable.** The engine docstring already warns:
  no PortAudio calls, no allocation, no locks held long inside the audio worker
  thread. Every new source must pre-allocate its working buffers and communicate
  with the GUI thread via the existing lock-snapshot pattern, not by allocating in
  the hot path. The Phase-2 "ring buffer / Float64Array / Map-per-tick" hardening
  is the precedent to follow.
- **Mic ↔ output coordination.** When the drone or a pitch pipe is sounding into
  the same room as the mic, the tuner will hear it. Android already fought this
  ("drone re-mute coordination," "deck close-path races"). Desktop needs the same:
  a coordination layer that ducks/gates pitch emissions or flags the readout as
  unreliable while output is active. Decide the policy in Sprint 1, enforce it in
  3 and 4.

This sprint also delivers the **navigation shell** so later features have a home.
A `QTabWidget` (or `QStackedWidget` + segmented control) with TUNER / METRO / DECK /
SETUP. TUNER keeps the current tuner-widget-plus-live-table layout — it is the
desktop's centerpiece and is more capable than Android's tuner screen, so it
absorbs drone controls and the pitch-pipes launcher rather than shrinking.

---

## 3. Cross-cutting concerns (apply to every sprint)

- **i18n:** every user-facing string lands in `sax_i18n.STRINGS` with `de` and
  `en` entries in the same commit that introduces it. No English-only stragglers.
- **Tests:** the repo has a real test culture (`test_*.py`, `conftest.py`). Each
  new module ships with tests in the same sprint — mixer summation/clamping,
  metronome scheduling and accent placement, deck record→playback roundtrip, drone
  voice resolution, output-stream lifecycle and hot-plug. Audio I/O is mocked the
  way the existing `test_hotplug_recovery` / `test_orphan_stream_cleanup` tests
  mock PortAudio.
- **Packaging:** `intonation_analyzer.spec` and `.github/workflows/release.yml`
  get updated in the sprint that adds a new native dependency or data file. The
  SF2 soundfont and any click samples are bundled via the spec's `datas`; new
  native libs (e.g. libfluidsynth) via `binaries` / `collect_dynamic_libs`.
- **Config:** new persisted settings extend `sax_config.py` dataclasses with
  defaults and `_as_*` coercion, mirroring the existing fields. Atomic writes via
  the existing `atomic_write_json` helper.
- **Controllers:** follow the Phase-5 pattern — feature logic in a dedicated
  controller (`MetronomeController`, `DroneController`, `DeckController`), GUI wiring
  thin. Do not grow `sax_intonation_gui.py` (already 3.6k lines) with feature
  internals.
- **Commits:** conventional-commit, scoped per feature (`feat(metro):`,
  `feat(drone):`, `feat(deck):`), continuing the repo's phase/version bump pattern.
  Comments explain *why*, not *what*; no jokes in code/comments/TODOs; TODOs state
  the real gap and the consequence of leaving it open.

---

## 4. Sprints

Effort is sized for solo dev + Claude Code. Treat each sprint as a shippable,
testable increment behind a version bump. Dependency order is strict: 1 gates
everything; 2 validates the foundation cheaply before the harder synths.

### Sprint 1 — Audio output foundation + navigation shell  ·  *keystone*

**Why first:** nothing else can make sound without it, and the nav shell gives the
later features a place to land. Doing the hard threading work once here de-risks 2–4.

**Scope**
- Add output capability to `sax_audio_engine.py`: full-duplex `sd.Stream` (fallback
  to separate `sd.OutputStream` if the selected device can't do duplex).
- Software mixer: pull-based source registry, per-block sum + clamp, pre-allocated
  buffers, no allocation in the callback.
- Mic↔output coordination policy (gate/duck + readout-unreliable flag) — interface
  defined and enforced for a trivial test tone.
- `QTabWidget` shell: TUNER (existing layout) / METRO / DECK / SETUP placeholders.
- Output device selection in SETUP (the input-device picker pattern already exists;
  generalize it).

**Files**
- `sax_audio_engine.py` (duplex stream, mixer, source registry)
- new `sax_mixer.py` (if the mixer warrants its own module — likely yes)
- `sax_intonation_gui.py` (tab shell), `sax_session_state.py` (persist active tab)
- `sax_config.py` (output device fields, active-tab persistence)

**Acceptance**
- A built-in test tone plays through the mixer while the tuner still reads the mic,
  with no xruns at the negotiated block size and no allocation in the callback
  (verify under a profiler / with a callback-allocation assert in debug).
- Hot-plug of the output device recovers the same way input hot-plug does today.
- Tabs switch without restarting audio.

**Tests:** mixer sum/clamp correctness; source register/unregister; duplex-stream
open/close/orphan cleanup (extend `test_orphan_stream_cleanup`); output hot-plug.

**Risks:** duplex unsupported on some Windows WDM-KS / WASAPI configs → the
separate-OutputStream fallback must be real, not theoretical. Latency/xruns at small
blocks. Mic feedback when the test tone hits the same device — coordination policy
must be in place before Sprint 3.

---

### Sprint 2 — Metronome

**Why second:** simplest output consumer; it exercises the mixer and sample-accurate
scheduling without the SF2 dependency. If the foundation is wrong, the metronome
exposes it cheaply.

**Scope (match Android `useMetronome`)**
- BPM 30–300 (default 100), ± nudge, tap-tempo (`registerTap` equivalent).
- Time signatures 2/4 · 3/4 · 4/4 · 6/8; accent on the downbeat.
- Click volume; start/stop/toggle; running indicator in the tab bar (Android shows a
  status dot — replicate).
- Click synthesis: short enveloped click(s), accent click distinct from beat click.
  Synthesize in numpy or bundle two small WAVs — synthesis avoids a data-file
  dependency, prefer it.
- Sample-accurate scheduling in the mixer (compute next-click sample index, not
  wall-clock timer ticks).

**Files:** new `sax_metronome.py` (`MetronomeController`), GUI MetroScreen panel,
`sax_config.py` (last BPM / time sig / click volume), `sax_i18n.py`.

**Acceptance:** click timing drift < 1 ms over 5 min at 120 BPM (scheduling is
sample-counted, so this should be exact bar instability is the bug if it appears);
accent lands on beat 1 for every time sig; tap-tempo converges within 4 taps;
running dot reflects state.

**Tests:** beat-index → sample-offset math; accent placement per time sig; tap-tempo
averaging; volume scaling into the mixer.

**Risks:** scheduling against block boundaries (a click that should fire mid-block
must be placed at the right intra-block sample). Output-route latency comp from
Android is largely irrelevant on desktop (single low-latency device) — explicitly
out of scope, note it.

---

### Sprint 3 — Drone + pitch pipes

**Why third:** highest-complexity audio feature (soundfont synthesis, native dep,
mic coordination). Pitch pipes ride along because they reuse the same synth.

**Scope (match Android v1.1 drone + `pitchTones`)**
- Soundfont synthesis via **pyfluidsynth + the GeneralUser-GS SF2** — the same
  soundfont Android uses, giving voice-faithful parity across all 128 GM programs.
  (Alternative: port TinySoundFont or a pure-numpy oscillator bank to avoid the
  native dependency. Trade-off below; pyfluidsynth is the pragmatic call for exact
  parity, oscillator bank is the lighter-packaging fallback.)
- 5-up preset row (Organ · Strings · Cello · Tenor Sax · Warm Pad, GM programs
  19/48/42/66/89 — copy verbatim from `droneVoices.ts`) + full 128 GM catalog
  picker.
- Drone pitch as a semitone offset from the current reference; volume; on/off.
  Drone controls live on the TUNER tab (as on Android), voice picker in SETUP.
- **Mic coordination:** while the drone sounds, gate pitch emissions or flag the
  readout unreliable (the Sprint-1 policy). Mirror Android's "drone re-mute
  coordination."
- **Pitch pipes:** reference-tone pads over a chromatic reference octave (Android
  default C4–B4), routed through the same synth/mixer. Launch from the TUNER tab.

**Files:** new `sax_drone.py` (`DroneController` + voice catalog ported from
`droneVoices.ts`), new `sax_pitch_pipes.py`, GUI drone controls + pitch-pipes
modal, `sax_config.py` (drone voice id, volume, semitone), `sax_i18n.py`,
`requirements.txt` (+ pyfluidsynth), `intonation_analyzer.spec` (+ libfluidsynth
binary, + SF2 data file), `release.yml`.

**Acceptance:** all 5 presets and a spot-check of GM voices sound correct; drone
holds a stable pitch at the selected semitone offset; tuner does not chase the
drone (coordination works); pitch-pipe pads play the correct reference frequencies
at the current A4.

**Tests:** voice resolution (`resolveDroneVoice` equivalent — id → program, default
fallback); semitone → frequency math at non-440 A4; pitch-pipe frequency table;
coordination state machine (drone on ⇒ readout gated).

**Risks:** **packaging is the real risk** — bundling libfluidsynth + a multi-MB SF2
through PyInstaller on Windows is the most likely thing to break CI. Validate the
built artifact early in the sprint, not at the end. SF2 licensing/redistribution:
confirm GeneralUser-GS terms allow bundling (they do, but record it). If pyfluidsynth
packaging proves fragile, fall back to the numpy oscillator-bank drone for the 5
presets and document GM-catalog parity as deferred.

---

### Sprint 4 — Tape deck (record / playback)

**Why fourth:** depends on the output path (playback) and benefits from the
coordination work in 3. Self-contained otherwise.

**Scope (match Android `useDeck`)**
- State machine `idle → recording → have-take → playing → (back to have-take)`.
- Record the mic to a take (WAV, 16-bit/PCM via `soundfile` or stdlib `wave`).
- Play the take back through the mixer; stop/replay.
- Share/export the take to a user-chosen path (reuse the export file-dialog
  patterns from `sax_export.py`).
- Recording indicator (pulsing dot) in the tab bar, matching Android.
- Single take is parity (Android holds one take). Multi-take is a non-goal for
  parity.

**Files:** new `sax_deck.py` (`DeckController`), GUI DeckScreen panel,
`sax_config.py` (last-take path / scratch dir), `sax_i18n.py`. `requirements.txt`
(+ soundfile only if not using stdlib `wave`).

**Acceptance:** record → stop → playback reproduces the captured audio; export
writes a valid WAV that opens in external players; close/stop during recording or
playback never leaves an orphaned stream (Android's "deck close-path races").

**Tests:** record→playback roundtrip (mocked stream, byte-exact buffer
comparison); WAV header correctness; state-machine transitions including
interrupt-during-record and interrupt-during-playback; orphan-stream cleanup on
abrupt close.

**Risks:** scratch-file lifecycle (temp dir, cleanup on exit/crash). Simultaneous
record + metronome/drone playback — decide whether the deck records the room only
(mic) or also captures the mixer output; parity = mic only, state it. Stream
teardown races (the Android close-path bug is a warning).

---

### Sprint 5 — Theme switching + SETUP parity + release

**Why last:** cosmetic and consolidation; depends on all feature surfaces existing
so the settings screen can reference them.

**Scope**
- Theme switching: `dark` (current) + `light` + `night`, plus a night-tuning
  emphasis mode (Android shows night-only tuning options). PyQt6 via QSS
  stylesheets keyed off a `ThemePalette`; centralize the currently-inline dark
  colors into a palette module first, then add the other two.
- SETUP screen parity sweep: response-mode selector (already exists in toolbar —
  surface it in SETUP too for parity), mic-gain floor control, drone voice picker,
  theme picker, A4, horn-nickname editor (promote the existing config field to a
  first-class editor), instrument range editor link.
- Accessibility pass (Android sets a11y labels throughout — add Qt accessible
  names/descriptions to the new controls).
- Final i18n sweep (no English-only strings across the new features).
- Packaging finalization, version bump, `release.yml` green on a real built
  artifact, README feature-list update.

**Files:** new `sax_theme.py` (palette + QSS), `sax_intonation_gui.py` (SETUP
panel, theme application), `sax_config.py` (theme name), `sax_i18n.py`, README,
`intonation_analyzer.spec` / `release.yml` (final).

**Acceptance:** all three themes apply live without restart and persist; every new
control is reachable from SETUP and labeled for accessibility and i18n; CI produces
a working bundled binary on the target platforms.

**Tests:** theme name persistence/coercion; palette completeness (no missing color
keys per theme); i18n key coverage (every new STRINGS key has both `de` and `en`).

**Risks:** light theme can expose contrast bugs in the custom-painted tuner/table
widgets (they assume dark) — budget rework time for the painted widgets, not just
QSS.

---

## 5. Sequencing and rationale

```
Sprint 1  ──┬──> Sprint 2 (metronome — cheap validation of the mixer)
 output     ├──> Sprint 3 (drone + pitch pipes — hardest synth + native dep)
 + nav      └──> Sprint 4 (deck — record/playback)
                                   └──> Sprint 5 (themes + settings + release)
```

- **1 is the long pole.** If duplex audio or callback discipline goes wrong,
  everything downstream inherits the bug. Spend the time here.
- **2 before 3** deliberately: the metronome is the cheapest possible exercise of
  the new output path and scheduling. Find foundation bugs with a sine click, not
  while also debugging a soundfont.
- **3's risk is packaging, not DSP.** Build and run the PyInstaller artifact on day
  one of the sprint. The fallback (numpy oscillator drone) exists specifically so a
  libfluidsynth packaging wall doesn't block the release.
- **5 last** because the settings screen and theming need every feature present to
  reference.

## 6. Non-goals (explicitly out of scope for parity)

- The exact bottom-tab visual metaphor — desktop uses a top/side `QTabWidget`;
  parity is *functional* (the four surfaces exist), not pixel-identical.
- Output-route selection + latency compensation (Android's speaker/wired/bluetooth
  logic) — desktop has one low-latency output device; not applicable.
- Multi-take deck management — Android holds a single take; match that.
- Removing or simplifying desktop-only strengths (statistical table, CSV slice
  modes, PDF export, autotune, diagnostics) — keep all of it.

## 7. Open questions — RESOLVED (Phase-0 discovery, 2026-05-28)

The three §7 questions were resolved by a parallel read-only discovery pass before
any code landed. Decisions are canonical for the rest of the sprint.

- **D5 — Stream topology: separate `sd.OutputStream` PRIMARY; duplex only as a
  same-device opt-in.** This *inverts* the plan's original "duplex preferred."
  Rationale (Gandalf): the engine selects an *input* device (often an external
  interface — the vendor-regex exists for exactly this) while output usually goes
  to a *different* device (laptop speakers/headphones), so true duplex needs one
  shared device or the fragile cross-device WASAPI/WDM-KS clock-sync the plan's own
  risk warned about. A separate OutputStream leaves the hard-won, well-tested input
  lifecycle (picker, host-API fallback, SR negotiation, worker-thread open-timeout,
  orphan disposal, hot-plug, `_transitioning` guard) UNTOUCHED. Sample-accurate
  timing (D2) *wants* the output stream's own monotonic DAC-frame counter as the
  master clock — duplex would muddy it. Output lifecycle stays on the Qt main
  thread to respect the `_transitioning` bool-guard invariant; it mirrors the
  input's orphan-disposal contract (0.8 s worker join + `cancelled` flag).

- **D1 — Drone synth: `tinysoundfont` (PyPI, MIT-licensed, ships binary wheels,
  sf2/sf3/sfo).** This *overrides* the plan's pyfluidsynth default, per operator
  requirement ("solid MIT-license MIDI engine"). It is the *same* TinySoundFont
  engine Android v1.1 drives (`audioGen.ts` / `@local/raw-audio-output`), giving
  exact GM voice parity, and it is a self-contained wheel — no system libfluidsynth,
  no LGPL, far lower PyInstaller risk. Bundle `GeneralUser-GS.sf2` (already in the
  android repo; consider sf3 to shrink). **Pitch pipes and metronome clicks are
  PURE NUMPY sine** (confirmed: android `pitchTones.ts` / `audioGen.ts`) — no synth
  engine needed. Sprint 3's "packaging is the real risk" drops to low.

- **D3 — Mic coordination: DUCK THE OUTPUT, keep the readout LIVE.** Neither the
  plan's "hard-gate" nor "duck + flag-unreliable" matches Android. The tuner stays
  live the entire time the drone sounds — tuning *against* the drone is the feature;
  gating the readout would destroy it. Leakage is handled in the engine: (1)
  vote-exclude the active output MIDI from incumbent pitch selection, (2)
  duck-on-suspicion — after ≥3 consecutive frames matching the output pitch, duck
  the output to ~30 % for ~80 ms, then a post-duck window confirms genuine unison vs
  pure bleed. Engine-layer, not GUI: the mixer publishes a frozenset of sounding
  output MIDIs via lock-snapshot; the input callback reads it.

- **D6 — Input-callback per-frame allocation: DEFER.** The existing input callback
  allocates per frame (`np.concatenate`, `sig = buf/rms`). Plan §2 forbids
  allocation in *new* sources, so the output path is held to a stricter bar
  (tracemalloc no-alloc gate) than the input meets today. We deliberately do NOT
  fix the input path mid-Sprint-1 — destabilizing the one proven path serves no
  one. Tracked as a backlog item: preallocate the input unwrap later, isolated and
  with its own no-alloc test.

## 8. Sprint-1 locked architecture

**Stream:** separate `sd.OutputStream` primary (D5). Engine gains an output mirror:
`_out_stream`, `open_output_device(sel)`, `_teardown_output_stream()`, an output
hot-plug recovery path parallel to the input one, and `get_sounding_output_midis()`
for D3 coordination. Output device persisted separately (coerce + clamp + default,
per the existing `sax_config` pattern).

**Mixer (`sax_mixer.py`, pure numpy, sounddevice-free):**

```
Mixer.render(out, frames, t0)   # additive sum into preallocated accumulator,
                                # clamp [-1, 1], NO allocation. t0 = abs sample
                                # index of the block's first frame = master clock.
Mixer.clock -> int              # readable; abs index of the NEXT frame to render
Mixer.schedule(abs_sample, ev)  # idempotent enqueue, sorted by abs_sample
Mixer.dropped_events -> int     # observable late-event canary

Source protocol: render(out, frames, t0) additive, pre-allocated buffers.
  Pitched sources expose active_midi: Optional[int].
  Source registry = lock-snapshot of an immutable source tuple, swapped on
  register/unregister (mirrors the engine's _buf-snapshot discipline).
```

**Scheduling contract (D2 — drift budget zero):** offset = `abs_sample − t0`.
- `offset ∈ [0, frames)` → fire at `out[offset]`.
- `offset == frames` → next block (half-open `[t0, t0+frames)`; kills double-fire).
- two events, same block → both fire, ascending sample order, additive.
- `offset < 0` (past) → DROP + `dropped_events += 1` (never clamp, never raise;
  late ≠ silent-skip — lateness is observable and testable).
- blocksize change between pulls → blocksize-invariant by construction (absolute
  samples).

**Nav (D4):** global toolbar ABOVE a `QTabWidget`; TUNER tab = the existing
splitter verbatim (zero churn to the centerpiece); METRO/DECK/SETUP added for later
sprints. Tab status dots: METRO green while running, DECK pulsing red while
recording (subscribe to source `running`/`active_midi` state — no GUI logic).

**File ownership (Sprint 1):**
- Gandalf — `sax_mixer.py` (+ reference `TestToneSource`), `sax_audio_engine.py`
  (output mirror). Engine-layer only.
- Frodo — `sax_intonation_gui.py` (nav shell, all engine/mixer wiring, output-device
  picker, status-dot hooks, test-tone control), `sax_i18n.py`.
- Treebeard — `sax_config.py` (output-device fields + `last_active_tab` + coercers),
  new `test_mixer.py` / `test_output_stream.py` / `test_config_roundtrip.py`.
