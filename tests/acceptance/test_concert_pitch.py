"""Concert pitch (A4) and autotune, told from the player's side.

A band tunes to A = 442, a choir to 440, a baroque group somewhere else. These
scenarios change the reference in the middle of a session and check that the
table, the history log, the exports, the drone and the pitch pipes all agree
on the new pitch, and that "Detect Concert Pitch" does what its dialog says.
"""
from __future__ import annotations

import csv
import math

import pytest

from tests.harness import Tone, midi_to_freq

ALTO = "eb_alto"

# Six notes inside the alto's range, all played 12 cents sharp: the player
# clearly tunes to a higher A than the app is set to.
SHARP_SIX = ("C4", "D4", "E4", "G4", "A4", "C5")

# A sine whose period is a whole number of samples at the 44.1 kHz microphone
# rate (133 samples, E4 about 10 cents sharp of A = 440). Two takes of it play
# back to back without a phase jump, so the detector hears one unbroken note.
HELD_E4_HZ = 44100 / 133
HELD_TAKE_S = 133 * 300 / 44100


def cents_between(hz: float, reference: float) -> float:
    return 1200.0 * math.log2(hz / reference)


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cents_against_logged_a4(row: dict) -> float:
    """What a raw CSV row's cents should be, worked out from its own frequency
    and the concert pitch the row says it was measured against."""
    target = midi_to_freq(int(row["midi_sounding"]), float(row["a4_hz"]))
    return cents_between(float(row["freq_hz"]), target)


def assert_rows_measured_against_their_a4(rows: list[dict]) -> None:
    wrong = [(r["sounding_note"], r["a4_hz"], r["cents"], round(cents_against_logged_a4(r), 2))
             for r in rows if abs(float(r["cents"]) - cents_against_logged_a4(r)) > 0.5]
    assert not wrong, f"rows whose cents do not match their own a4_hz: {wrong}"


def play_sharp_six(app, cents: float = 12.0, a4: float = 440.0) -> None:
    app.pick_instrument(ALTO)
    for note in SHARP_SIX:
        app.play(Tone(note=note, cents=cents, a4=a4))


def heard_pitch(app, near: float, seconds: float = 1.0) -> float:
    return app.capture_output(seconds).pitch(near=near)


# ------------------------------------------------------------ scenario 5: A4 mid-session
def test_changing_a4_mid_session_re_reads_every_played_note_against_the_new_pitch(app):
    """Switching to 442 keeps every note I played and shows how it sits against 442."""
    app.pick_instrument(ALTO)
    for note, cents in (("C4", 0), ("E4", -15), ("A4", 0)):
        app.play(Tone(note=note, cents=cents))
    before = {r.sounding: r for r in app.table.rows()}
    assert len(before) == 3

    app.set_a4(442)

    assert (app.a4, app.engine_a4) == (442, 442.0)
    after = {r.sounding: r for r in app.table.rows()}
    assert set(after) == set(before)
    shift = cents_between(440.0, 442.0)                     # about -7.85 cents
    for name, row in before.items():
        assert after[name].n == row.n
        assert abs(after[name].mean - (row.mean + shift)) <= 1, name
    assert after["A4"].band == "amber"                      # was green at 440


def test_a_discarded_note_stays_gone_when_the_concert_pitch_changes(app):
    """A squeak I threw away must not come back because I touched the A4 selector."""
    app.pick_instrument(ALTO)
    app.play(Tone(note="A4"))
    app.play(Tone(note="C5", cents=-30))
    app.table.discard("C5", confirm=True)
    assert app.table.get("C5", by="sounding") is None

    app.set_a4(442)
    assert app.table.get("C5", by="sounding") is None
    app.set_a4(440)
    assert app.table.get("C5", by="sounding") is None
    assert app.table.row("A4", by="sounding").mean == 0


def test_an_a4_change_starts_a_new_run_and_exports_show_the_new_pitch(app):
    """Playing on after switching to 442 is logged as a separate run stamped 442."""
    app.pick_instrument(ALTO)
    app.play(Tone(note="A4"))
    app.set_a4(442)
    app.play(Tone(note="A4"))                               # still a true 440 Hz A

    runs, measurements = app.history_counts
    assert runs == 2
    rows = read_csv(app.export.csv("raw"))
    assert len(rows) == measurements
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    stamps = sorted(sorted({r["a4_hz"] for r in run}) for run in by_run.values())
    assert stamps == [["440.00"], ["442.00"]]               # one pitch per run
    assert_rows_measured_against_their_a4(rows)

    report = app.export.txt().read_text(encoding="utf-8")
    assert app.text("txt_a4", hz=442.0) in report


def test_the_drone_retunes_while_it_sounds_when_a4_changes(app):
    """If the drone is on and I switch to 442, the drone I am tuning to moves too."""
    app.drone.on("A4", voice="sine")
    assert abs(cents_between(heard_pitch(app, 440.0), 440.0)) < 2.0

    app.set_a4(442)

    assert app.drone.state.enabled
    assert abs(cents_between(heard_pitch(app, 442.0), 442.0)) < 2.0


def test_a_pitch_pipe_pressed_after_an_a4_change_sounds_at_the_new_pitch(app):
    """The pitch pipes give me A = 442 once the app is set to 442."""
    app.set_a4(442)

    app.pipes.press("A4")

    assert abs(cents_between(heard_pitch(app, 442.0), 442.0)) < 2.0
    app.pipes.release("A4")


@pytest.mark.xfail(strict=True, reason="Sauron-5: a held pitch-pipe pad keeps the old A4 "
                                       "after the concert pitch changes")
def test_a_held_pitch_pipe_follows_an_a4_change(app):
    """Holding the A pipe while switching to 442 should move the pipe to 442."""
    app.pipes.press("A4")
    assert abs(cents_between(heard_pitch(app, 440.0), 440.0)) < 2.0

    app.set_a4(442)

    assert app.pipes.active == [69]
    assert abs(cents_between(heard_pitch(app, 442.0), 442.0)) < 2.0


def test_going_to_442_and_back_to_440_gives_back_the_same_table(app):
    """Trying 442 and changing my mind must not cost me anything."""
    app.pick_instrument(ALTO)
    for note, cents in (("C4", 0), ("E4", -15), ("G4", 8), ("A4", 30), ("C5", -3)):
        app.play(Tone(note=note, cents=cents))
    before = app.table.rows()

    app.set_a4(442)
    assert app.table.rows() != before
    app.set_a4(440)

    assert app.table.rows() == before


def test_dropping_to_430_moves_a_sharp_a_onto_b_flat_and_440_brings_it_back(app):
    """At A = 430 a very sharp A really is closer to Bb; going back to 440 restores it."""
    app.pick_instrument(ALTO)
    app.play(Tone(note="A4", cents=15))
    a4_row = app.table.row("A4", by="sounding")

    app.set_a4(430)

    assert app.table.get("A4", by="sounding") is None
    bb = app.table.row("Bb4", by="sounding")
    assert bb.n == a4_row.n
    assert abs(bb.mean - (15 + cents_between(440.0, 430.0) - 100)) <= 1    # about -45

    app.set_a4(440)

    assert app.table.get("Bb4", by="sounding") is None
    assert app.table.row("A4", by="sounding") == a4_row


# ------------------------------------------------------------- the 430 and 450 limits
def test_a_440_hz_a_reads_40_cents_sharp_at_430_and_39_flat_at_450(app):
    """The ends of the selector still grade a plain 440 Hz A correctly."""
    app.pick_instrument(ALTO)
    app.set_a4(430)
    app.play(Tone(note="A4"))
    assert abs(app.table.row("A4", by="sounding").mean - cents_between(440.0, 430.0)) <= 1

    app.set_a4(450)

    assert app.engine_a4 == 450.0
    assert abs(app.table.row("A4", by="sounding").mean - cents_between(440.0, 450.0)) <= 1


@pytest.mark.parametrize("hz", [430, 450])
def test_the_drone_sounds_the_selected_pitch_at_the_ends_of_the_range(app, hz):
    """A drone at A = 430 or 450 sounds exactly that A."""
    app.set_a4(hz)

    app.drone.on("A4", voice="sine")

    assert abs(cents_between(heard_pitch(app, float(hz)), float(hz))) < 2.0


def test_notes_other_than_a_read_in_tune_at_443(app):
    """At A = 443 every note of an equal-tempered scale built on 443 reads as it was played."""
    app.pick_instrument(ALTO)
    app.set_a4(443)

    for note, cents in (("C5", 0), ("E4", -10), ("G4", 8), ("Bb3", 20)):
        app.play(Tone(note=note, cents=cents, a4=443.0))

    for note, cents in (("C5", 0), ("E4", -10), ("G4", 8), ("Bb3", 20)):
        assert abs(app.table.row(note, by="sounding").mean - cents) <= 1, note


# --------------------------------------------------------- saved A4 after a restart
def test_saved_concert_pitch_reaches_the_drone_pipes_and_first_run_after_restart(make_app):
    """Yesterday I left the app on 442; today the drone, the pipes and the log all start at 442."""
    app = make_app(config={"last_a4_hz": 442})
    app.launch()
    assert (app.a4, app.engine_a4) == (442, 442.0)

    app.pick_instrument(ALTO)
    app.play(Tone(note="A4"))                               # a true 440 Hz A
    rows = read_csv(app.export.csv("raw"))
    assert {r["a4_hz"] for r in rows} == {"442.00"}
    assert_rows_measured_against_their_a4(rows)
    assert app.table.row("A4", by="sounding").mean == -8

    app.drone.on("A4", voice="sine")
    assert abs(cents_between(heard_pitch(app, 442.0), 442.0)) < 2.0
    app.drone.off()

    app.pipes.press("A4")
    assert abs(cents_between(heard_pitch(app, 442.0), 442.0)) < 2.0
    app.pipes.release("A4")


# ------------------------------------------------------------- scenario 6: autotune
def test_autotune_suggests_a_higher_a4_for_a_consistently_sharp_player(app):
    """Six notes all 12 cents sharp: autotune offers 443, and accepting moves everything to 443."""
    play_sharp_six(app, cents=12)

    result = app.autotune.run(accept=True)

    assert result.status == "applied"
    assert (result.a4_before, result.a4_after) == (440, 443)
    assert app.text("autotune_confirm", clamped=443) in result.message
    assert (app.a4, app.engine_a4) == (443, 443.0)

    app.play(Tone(note="A4", a4=443.0))
    assert app.table.row("A4", by="sounding").mean == 0
    assert app.history_counts[0] == 2                       # a new run at 443
    rows = read_csv(app.export.csv("raw"))
    assert {r["a4_hz"] for r in rows} == {"440.00", "443.00"}
    assert_rows_measured_against_their_a4(rows)

    app.drone.on("A4", voice="sine")
    assert abs(cents_between(heard_pitch(app, 443.0), 443.0)) < 2.0


def test_autotune_with_fewer_than_three_notes_says_how_far_along_i_am(app):
    """With two notes played, autotune tells me 2 of 3 are done and changes nothing."""
    app.pick_instrument(ALTO)
    app.play(Tone(note="C4", cents=12))
    app.play(Tone(note="E4", cents=12))

    result = app.autotune.run(accept=True)

    assert result.status == "insufficient_data"
    assert app.text("autotune_nodata_progress", qualified=2, touched=2) in result.message
    assert (app.a4, app.engine_a4) == (440, 440.0)
    assert len(app.table.rows()) == 2


@pytest.mark.parametrize("hz, cents", [(450, +30), (430, -30)], ids=["sharp-at-450", "flat-at-430"])
def test_autotune_beyond_the_selector_range_says_it_stops_at_the_limit(app, hz, cents):
    """A player far outside 430..450 is told the suggestion was capped at the limit."""
    app.set_a4(hz)
    play_sharp_six(app, cents=cents, a4=float(hz))
    wanted = round(hz * 2.0 ** (cents / 1200.0))            # 458 or 423

    result = app.autotune.run(accept=True)

    assert result.status == "applied"
    assert app.text("autotune_clamp", rounded=wanted, clamped=hz) in result.message
    assert (app.a4, app.engine_a4) == (hz, float(hz))


@pytest.mark.xfail(strict=True, reason="Frodo-21: accepting autotune wipes the table, while a "
                                       "manual A4 change re-derives it")
def test_accepting_autotune_keeps_my_notes_like_a_manual_a4_change_does(app):
    """After autotune moves A4 to 443 my six notes should still be there, now reading about 0."""
    play_sharp_six(app, cents=12)

    app.autotune.run(accept=True)

    rows = app.table.rows()
    assert sorted(r.sounding for r in rows) == sorted(SHARP_SIX)
    assert all(abs(r.mean) <= 1 for r in rows), [(r.sounding, r.mean) for r in rows]


# ----------------------------------------------- readings measured before the change
@pytest.mark.xfail(strict=True, reason="Sauron-3: readings measured at the old A4 and still "
                                       "queued when autotune is accepted land in the new run")
def test_a_note_held_through_autotune_is_logged_against_the_new_pitch(app):
    """I keep sustaining E4 while clicking Yes: the 443 run is graded against 443 only."""
    app.pick_instrument(ALTO)
    for note in ("C4", "D4", "G4", "A4", "C5"):
        app.play(Tone(note=note, cents=12))
    app.play(Tone(freq=HELD_E4_HZ, seconds=HELD_TAKE_S), release=0)    # still sounding

    result = app.autotune.run(accept=True)
    assert result.a4_after == 443
    app.play(Tone(freq=HELD_E4_HZ, seconds=HELD_TAKE_S))                # same note, no break

    rows = read_csv(app.export.csv("raw"))
    new_run = [r for r in rows if r["a4_hz"] == "443.00"]
    assert new_run
    assert_rows_measured_against_their_a4(new_run)


def test_a_note_held_through_a_manual_a4_change_is_logged_against_the_new_pitch(app):
    """I keep sustaining E4 while picking 442 by hand: the 442 run is graded against 442 only."""
    app.pick_instrument(ALTO)
    app.play(Tone(freq=HELD_E4_HZ, seconds=HELD_TAKE_S), release=0)

    app.set_a4(442)
    app.play(Tone(freq=HELD_E4_HZ, seconds=HELD_TAKE_S))

    rows = read_csv(app.export.csv("raw"))
    assert {r["a4_hz"] for r in rows} == {"440.00", "442.00"}
    assert_rows_measured_against_their_a4(rows)


# ------------------------------------------------------------- exports across pitches
@pytest.mark.xfail(strict=True, reason="Sauron-12: aggregate CSV modes average cents measured "
                                       "against different A4 values into one number")
@pytest.mark.parametrize("mode", ["per_instrument_note", "per_nickname_note",
                                  "instrument_avg", "overall_per_note"])
def test_summary_csv_keeps_sessions_at_different_concert_pitches_apart(app, mode):
    """The same 440 Hz A is 0 cents at 440 and -8 at 442; a summary must not report about -4,
    a number that describes neither session."""
    app.pick_instrument(ALTO)
    app.play(Tone(note="A4"))
    app.set_a4(442)
    app.play(Tone(note="A4"))

    kw = {"instrument": ALTO} if mode == "instrument_avg" else {}
    rows = read_csv(app.export.csv(mode, **kw))

    means = [float(r["mean_cents"]) for r in rows if r["sounding_note"] == "A4"]
    assert means
    session_means = (0.0, cents_between(440.0, 442.0))
    blended = [m for m in means if min(abs(m - s) for s in session_means) > 1.5]
    assert not blended, f"A4 summary rows mixing both pitches: {blended}"


@pytest.mark.xfail(strict=True, reason="Sauron-12: the TXT header drops the octave for a "
                                       "+/-12 transposition ('fingered C sounds as C')")
def test_txt_header_for_a_piccolo_says_it_sounds_an_octave_up(app):
    """A piccolo sounds an octave above written; the header must not read like no transposition."""
    app.pick_instrument("flute_piccolo")

    report = app.export.txt().read_text(encoding="utf-8")

    assert app.text("txt_no_transp") not in report
    assert app.text("txt_transp", note="C") not in report.splitlines()
