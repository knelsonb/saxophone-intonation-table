# Test layout

    tests/
      acceptance/    scenario tests: the whole app driven through the harness
      harness/       the headless app, fake audio system and golden-file helpers
      golden/        input fixtures and reviewed expected outputs
      conftest.py    fixtures: app, fresh_app, make_app, host, clock
    test_*.py        the original unit and module-level suite (repo root)

## The rule that keeps the suite alive through a rewrite

Scenario tests talk to the app only through `tests.harness`
(`HeadlessApp` and the objects hanging off it: `table`, `tuner`, `export`,
`metronome`, `drone`, `pipes`, `deck`, `settings`, `audio`, `welcome`,
`autotune`, `dialogs`, `host`). They never import an app module (`intonation.*`)
and never touch a widget.

`tests/harness/driver.py` is the one module that knows how the app is built. Its
docstring lists every internal it relies on. When the app is restructured, that
file is the only thing to update; the scenarios keep running unchanged.

## How an app is run

`HeadlessApp(home, lang, host)` imports a private copy of the app, builds the
main window on Qt's offscreen platform and takes over its timers:

* **Audio**: `FakeAudioHost` stands in for `sounddevice`. Nothing runs on a
  thread; `app.play(Tone(...))` hands microphone blocks to the app and
  `app.capture_output(seconds)` collects what the app rendered for the speaker.
  Devices can be added, unplugged, replugged, made busy, made to hang, or made to
  refuse rates (`app.host`).
* **Time**: `ManualClock`. Nothing sleeps. `app.advance(2.0)` ticks the app's
  timers and pulls two seconds of output audio.
* **Dialogs**: every modal is intercepted. Script answers with
  `app.dialogs.expect(...)`, `save_path(...)`, `open_path(...)`. A dialog nobody
  scripted is recorded and raised from the test (it never blocks).
* **Isolation**: each app gets its own HOME and its own copy of every app module;
  closing it restores `sys.modules`, the environment and Qt exactly.

## Fixtures

* `app` is launched: English UI, welcome dialog already answered, list layout,
  one microphone (44.1 kHz) and one speaker (48 kHz).
* `fresh_app` is *not* launched and has no saved settings. Script the welcome
  dialog, then `launch()`.
* `make_app(**kw)` builds an app with its own seed files (`config=`, `customs=`,
  `ranges=`, `history=`), language or devices; call `launch()` yourself.

## Golden files

See `golden/README.md`. `UPDATE_GOLDEN=1 pytest tests/acceptance` rewrites the
expected outputs; read the diff before committing. Regenerate on the reference
platform (the BT2 venv) since measured values depend on the numeric libraries.
