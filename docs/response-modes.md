# Response Modes — Fast / Normal / Slow

The app has three response presets that control how quickly the pitch readout
reacts to what you're blowing, and how aggressively it smooths out the noise
inherent in real-world wind-instrument sound. Each preset is tuned for a
specific working use case: live play, practice, or instrument setup and
repair. Pick the one that matches what you're doing right now and the readout
will behave the way you expect.

You change the preset from the **Response** group in the toolbar or in the
SETUP tab; the two stay in sync. The current preset is also shown in the
Diagnostics panel (checkbox "Show spectrum analyzer & diagnostics" under the
tuner) next to the live RMS and YIN aperiodicity values.

The internal preset key — `fast`, `normal`, `slow` — is what gets saved to
your config; the label is just a friendly name that depends on your UI
language.

**Timing terms used below.** The engine analyses the microphone once per
*hop* of about 46 ms (2048 samples at 44.1 kHz). Each hop that passes the
filters adds one reading to the table. The *first reading* of a note takes
`confirm + 2 × edge_hops` hops: `confirm` matching detections to lock onto
the note, then the edge guard holds back `edge_hops` hops at the start of the
note and keeps a further `edge_hops` readings buffered so the release end of
the note can be trimmed. After that, readings arrive once per hop, about 20
per second.

---

## Fast — live play

**Use this when** you're running scales, working through études, playing
along to a backing track, or anything else where you want the readout to
react to each note as you blow it. The goal is per-note feedback while you
play; a single sustained tone gets measured more precisely in Normal or Slow.

**What it does:** the first reading of a note arrives about 190 ms (4 hops)
after you start a clean, sustained pitch. Smoothing window is short (~140 ms
median), so the displayed cents value tracks small expression changes —
vibrato, scoops, fall-offs — instead of averaging them out. Edge guard
truncates the first hop (~46 ms) of attack chiff and the last hop of release
pitch drop, so the displayed value isn't biased by the transient ends of each
note.

**Limits:** wide vibrato and pitch bends pass through into the reading. The
value you see during a live phrase jitters; that's by design.

---

## Normal — practice (default)

**Use this when** you're tuning the horn to a reference, holding long tones,
working through Remington-style sustained-pitch drills, or matching pitch
inside a section. This is the everyday default — balanced between
responsiveness and stability.

**What it does:** the first reading arrives after about 330 ms (7 hops).
Smoothing window is medium (~230 ms median over same-MIDI detections), so
passing pitch bobble inside a sustained note averages out but the readout
still moves when you actually shade the pitch. Edge guard is two hops
(~92 ms) on each side of the note — attack chiff and release droop are gone
before any value is committed to the chart.

**Limits:** for the deepest average on a perfectly steady note, for example
when cataloguing an instrument's intonation map, switch to Slow.

---

## Slow — instrument setup / repair / cataloging

**Use this when** you're measuring an instrument's true pitch tendencies
during a setup or repair pass — leak-testing, neck-pull cataloging,
mouthpiece comparison, building an intonation chart for a horn you've never
played before. You're trying to learn the **true sustained pitch** of a
fully-blown note, with attack and release behavior treated as noise.

**What it does:** the first reading arrives after about 650 ms (14 hops).
Smoothing window is deep (~460 ms median over the last ~10 same-MIDI
detections). Edge guard is four hops (~186 ms) on each end of the note —
that fully suppresses the attack chiff and any release pitch drop. Noise
gates are stricter: lower YIN aperiodicity ceiling (0.07 vs. 0.10 in Normal),
higher RMS floor (3e-4 vs. 1.5e-4). Frames with marginal periodicity or weak
signal are dropped before they can pollute the median.

**Practical rule for repair work:** always use Slow. Hold each test note
for at least 2 seconds: the first reading takes about 0.65 s, and the rest of
the note adds roughly 20 readings per second. Trust the readings from the
settled middle of the note. The moment a note first locks is when the YIN
detector cleared the gate, which can come before the player's embouchure has
settled.

---

## Comparison

| Preset | First reading | Smoothing window | Edge guard | YIN gate | RMS floor |
|--------|--------------:|-----------------:|-----------:|---------:|----------:|
| Fast   |   ~190 ms (4 hops) |       ~140 ms |   ~46 ms  |    0.15  |  8e-5     |
| Normal |   ~330 ms (7 hops) |       ~230 ms |   ~93 ms  |    0.10  |  1.5e-4   |
| Slow   |  ~650 ms (14 hops) |       ~460 ms |  ~186 ms  |    0.07  |  3e-4     |

First reading = `(confirm + 2 × edge_hops)` hops, measured through the engine;
smoothing window = `window` hops. One hop is 46.4 ms, so the first-reading
times are 4 × 46.4 = 186 ms, 7 × 46.4 = 325 ms and 14 × 46.4 = 650 ms; the
table and the in-app tooltips round them to the nearest 10 ms (190, 330, 650).
The hop is about 46 ms at every sample rate:
the engine scales the hop size in samples with the negotiated rate, so these
times hold at 44.1, 48, 96 and 192 kHz alike. The actual numeric constants are
`window`, `confirm`, `yin_thr`, `rms_floor`, and `edge_hops` in
`FILTER_PRESETS` (see `intonation/audio/engine.py`).

---

## Baseline — how this compares to other tuners

For context, here is how the established hardware/software tuners that this
app's users tend to already own define their response settings:

- **Korg OT-120 / AW-LT100 (orchestral tuners).** Two-way slow/fast toggle.
  Slow averages ~500 ms – 1 s for steady-state long tones; Fast is a
  ~100–200 ms snap. Our **Slow** sits inside the range of their Slow;
  our **Fast** is roughly equivalent to their Fast.
- **Peterson StroboPlus HD / iStroboSoft.** No toggle. The strobe display
  smooths visually via its rotation rate, and the "capture / analyze" mode
  averages the last several seconds — that's the equivalent of our Slow
  preset plus a sustained note.
- **Boss TU-3 / Yamaha TD-1 (stage tuners).** Fixed ~200–300 ms needle time
  constant. No user setting. This is roughly equivalent to our **Normal**.
- **TonalEnergy Tuner (the modern wind-player standard).** Continuous
  smoothing slider 0–100; most teachers leave it around 30–40. Our three
  presets correspond roughly to TET smoothing values of ~15 (Fast), ~35
  (Normal), and ~70 (Slow).

If you're switching to this app from one of those, "Normal" is the closest
match to what you're already used to. Move to "Fast" if the readout feels
sluggish during scale work. Move to "Slow" if you're sitting down with a
horn to write an intonation report.

---

## Which one am I using?

Tick "Show spectrum analyzer & diagnostics" (or the equivalent in your
language) below the tuner on the TUNER tab. The current preset name appears in
the diagnostics field ("Filter mode") alongside RMS, YIN aperiodicity, and the
running detection counters. The preset is also saved into your config file
between runs, so the next launch starts in whichever mode you last selected.
