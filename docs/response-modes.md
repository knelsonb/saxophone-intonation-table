# Response Modes — Fast / Normal / Slow

The app has three response presets that control how quickly the pitch readout
reacts to what you're blowing, and how aggressively it smooths out the noise
inherent in real-world wind-instrument sound. Each preset is tuned for a
specific working use case: live play, practice, or instrument setup and
repair. Pick the one that matches what you're doing right now and the readout
will behave the way you expect.

You change the preset from the **Response** group in the toolbar. The current
preset is also shown in the Diagnostics panel (toggle: "Show spectrum
analyzer & diagnostics") next to the live RMS and YIN aperiodicity values.

The internal preset key — `fast`, `normal`, `slow` — is what gets saved to
your config; the label is just a friendly name that depends on your UI
language.

---

## Fast — live play

**Use this when** you're running scales, working through études, playing
along to a backing track, or anything else where you want the readout to
react to each note as you blow it. The goal is per-note feedback while you
play, not laboratory precision on a single sustained tone.

**What it does:** ~140 ms lock latency from the moment you hit a clean
sustained pitch to the moment the cents number stabilizes. Smoothing window
is short (~140 ms median), so the displayed cents value tracks small
expression changes — vibrato, scoops, fall-offs — instead of averaging them
out. Edge guard truncates the first hop (~46 ms) of attack chiff and the
last hop of release pitch drop, so the displayed value isn't biased by the
transient ends of each note.

**What it doesn't do:** It will not filter out wide vibrato or pitch bends.
The reading you see during a live phrase will jitter; that's by design.

---

## Normal — practice (default)

**Use this when** you're tuning the horn to a reference, holding long tones,
working through Remington-style sustained-pitch drills, or matching pitch
inside a section. This is the everyday default — balanced between
responsiveness and stability.

**What it does:** ~230 ms lock latency. Smoothing window is medium
(~230 ms median over same-MIDI detections), so passing pitch bobble inside a
sustained note averages out but the readout still moves when you actually
shade the pitch. Edge guard is two hops (~92 ms) on each side of the note —
attack chiff and release droop are gone before any value is committed to the
chart.

**What it doesn't do:** It will not give you the deepest possible average on
a perfectly steady note; if you're cataloging an instrument's intonation
map, switch to Slow.

---

## Slow — instrument setup / repair / cataloging

**Use this when** you're measuring an instrument's true pitch tendencies
during a setup or repair pass — leak-testing, neck-pull cataloging,
mouthpiece comparison, building an intonation chart for a horn you've never
played before. You're trying to learn the **true sustained pitch** of a
fully-blown note, with attack and release behavior treated as noise.

**What it does:** ~460 ms lock latency. Smoothing window is deep
(~460 ms median over the last ~10 same-MIDI detections). Edge guard is four
hops (~184 ms) on each end of the note — that fully suppresses the attack
chiff and any release pitch drop. Noise gates are stricter: lower YIN
aperiodicity ceiling (0.07 vs. 0.10 in Normal), higher RMS floor (3e-4 vs.
1.5e-4). Frames with marginal periodicity or weak signal are dropped before
they can pollute the median.

**Practical rule for repair work:** always use Slow. Hold each test note
for at least 1 full second. The reading you'll trust is the median of the
last ~10 frames, not the first one that locks on. The first frame in the
locked region is the moment the YIN detector cleared the gate, not the
moment the player's embouchure settled.

---

## Comparison

| Preset | Lock latency | Smoothing window | Edge guard | YIN gate | RMS floor |
|--------|-------------:|-----------------:|-----------:|---------:|----------:|
| Fast   |       ~140 ms |          ~140 ms |   ~46 ms  |    0.15  |  8e-5     |
| Normal |       ~230 ms |          ~230 ms |   ~92 ms  |    0.10  |  1.5e-4   |
| Slow   |       ~460 ms |          ~460 ms |  ~184 ms  |    0.07  |  3e-4     |

All numbers assume the negotiated audio sample rate is 44.1 kHz; the hop
size adapts so that timing in milliseconds stays roughly constant at higher
rates. The actual numeric constants are `window`, `confirm`, `yin_thr`,
`rms_floor`, and `edge_hops` in `FILTER_PRESETS` (see `sax_audio_engine.py`).

---

## Baseline — how this compares to other tuners

For context, here is how the established hardware/software tuners that this
app's users tend to already own define their response settings:

- **Korg OT-120 / AW-LT100 (orchestral tuners).** Two-way slow/fast toggle.
  Slow averages ~500 ms – 1 s for steady-state long tones; Fast is a
  ~100–200 ms snap. Our **Slow** sits at the responsive end of their Slow;
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

Open the Diagnostics panel (Settings → "Show spectrum analyzer &
diagnostics", or the equivalent in your language). The current preset name
appears in the diagnostics field alongside RMS, YIN aperiodicity, and the
running detection counters. The preset is also saved into your config file
between runs, so the next launch starts in whichever mode you last selected.
