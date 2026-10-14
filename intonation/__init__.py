"""Intonation Analyzer: a real-time intonation table for wind instruments.

    core/         pitch math, YIN, the note detector, statistics, the session
    catalog/      the instrument catalogue (data and range overrides)
    persistence/  data directory, settings, history log, atomic writes
    i18n/         the UI strings and the Translator
    audio/        the audio engine, the output mixer and its sound sources
    export/       TXT / PDF / CSV / chart renderers and the export flows
    ui/           the main window, its tabs, widgets and dialogs
    crash.py      the last-resort exception handler, installed first
    assets.py     bundled-file paths that work from source and when frozen

``ui/`` sits on top and may import anything. Two edges still point upwards:
``persistence.config`` and the export renderers read the cents bands and
theme names from ``ui.theme``.

``python -m intonation`` starts the app (see ``__main__``).
"""

APP_NAME = 'Intonation Analyzer'
APP_VERSION = '1.4.0'
