# Golden files

## Inputs (loaded with `fixture_path(name)`)

| File | What it is |
| --- | --- |
| `config_v0.6.json` | `config.json` as written by v0.6.2 (24 fields). Built from that tag's real `AppConfig`. |
| `config_v1.0.json`, `config_v1.1.json`, `config_v1.2.json` | The same for v1.0.0, v1.1.0 and v1.2.0 (37 fields each). The field set did not change across these three; the files differ in the user choices they hold, including a `matrix_extra_octaves` of 9 that v1.0/v1.1 accepted and v1.2 clamps to 4. |
| `custom_instruments.json` | A customs file: one full entry, one legacy single-name entry, two malformed entries the loader must skip. |
| `instrument_ranges.json` | Range overrides: two valid, three invalid (low above high, out of 0..127, wrong type). |
| `log.jsonl` | A deterministic measurement history: 3 runs, 96 measurements, two instruments, one maker/model containing a comma and quotes, one nickname with an umlaut. |

`make_config_fixtures.py` and `make_log_fixture.py` regenerate them
(`python tests/golden/make_config_fixtures.py` needs the repo's git history).

## Expected outputs (`expected/`, compared with `assert_golden`)

* `csv_<mode>.csv`: the six CSV export modes of `log.jsonl`.
* `txt_known_table_en.txt`, `txt_known_table_de_markup.txt`: TXT export of the
  five-note table in `tests/acceptance/_scenarios.py`; the second is German with
  `&` and `<` in the maker and model.
* `pdf_model_known_table_*.json`: what the PDF export hands to the PDF renderer
  (paragraph markup and plain text, table cells, cell text colours). The venv has
  no PDF text extractor and a PDF embeds timestamps, so the renderer's input is
  the golden, not the file.
* `chart_model_known_table_en.json`: the arguments of the chart renderer (rows,
  title, subtitle, footer). Never pixels.
* `config_after_v*.json`, `config_keys.json`: the settings file after the current
  version has loaded and re-saved each release's config (window geometry, window
  state and splitter sizes are blanked: they depend on the display).
* `i18n_snapshot.json`: every UI string key per language and its format
  placeholders.

Wall-clock text (`13.10.2026 16:44`, `2026-10-13 16:44`) is replaced by
`<DD.MM.YYYY HH:MM>` / `<YYYY-MM-DD HH:MM>` before comparing.

Regenerate with `UPDATE_GOLDEN=1 pytest tests/acceptance/test_goldens.py` on the
BT2 venv, then review the diff.
