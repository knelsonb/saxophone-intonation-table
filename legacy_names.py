"""The module names of releases before 2.0, and where each one lives now.

Until 2.0 the app was a flat set of ``sax_*`` modules in this directory. Each
old name stays importable for one release through a ``sax_*.py`` shim here
that puts the package module itself under the old name in ``sys.modules``:
``import sax_config`` and ``intonation.persistence.config`` are then the same
object, so a patch applied through either name is seen by both. (A
``from new import *`` re-export would copy the names into a second module
object; a test that patches ``sax_config.X`` would then change a copy the app
never reads.) The test
harness reads this table to point the old names at the private copy of the
app it loads, and test_legacy_shims.py checks every shim against it.

Source checkouts only: the frozen binary has neither the shims nor this file.
Delete them together.
"""

LEGACY_MODULES = {
    'sax_assets': 'intonation.assets',
    'sax_atomic': 'intonation.persistence.atomic',
    'sax_audio_control': 'intonation.audio.control',
    'sax_audio_engine': 'intonation.audio.engine',
    'sax_config': 'intonation.persistence.config',
    'sax_controls': 'intonation.ui.controls',
    'sax_coordination': 'intonation.core.coordination',
    'sax_crash': 'intonation.crash',
    'sax_deck': 'intonation.audio.sources.deck',
    'sax_detector': 'intonation.core.detector',
    'sax_device': 'intonation.ui.device_controller',
    'sax_dialogs': 'intonation.ui.dialogs',
    'sax_drone': 'intonation.audio.sources.drone',
    'sax_drone_bar': 'intonation.ui.drone_bar',
    'sax_export': 'intonation.export.controller',
    'sax_flow_layout': 'intonation.ui.flow_layout',
    'sax_i18n': 'intonation.i18n.strings',
    'sax_instrument_controller': 'intonation.ui.instrument_controller',
    'sax_instruments': 'intonation.catalog.instruments',
    'sax_intonation_chart': 'intonation.export.chart',
    'sax_intonation_gui': 'intonation.ui.main_window',
    'sax_intonation_log': 'intonation.persistence.history',
    'sax_metronome': 'intonation.audio.sources.metronome',
    'sax_mixer': 'intonation.audio.mixer',
    'sax_output_status': 'intonation.ui.output_status',
    'sax_output_users': 'intonation.audio.output_users',
    'sax_paths': 'intonation.persistence.paths',
    'sax_pitch': 'intonation.core.pitch',
    'sax_pitch_pipes': 'intonation.audio.sources.pitch_pipes',
    'sax_report': 'intonation.export.report',
    'sax_session': 'intonation.core.session',
    'sax_session_state': 'intonation.ui.session_state',
    'sax_stats': 'intonation.core.stats',
    'sax_tab_deck': 'intonation.ui.tabs.deck',
    'sax_tab_metro': 'intonation.ui.tabs.metro',
    'sax_tab_setup': 'intonation.ui.tabs.setup',
    'sax_tab_tuner': 'intonation.ui.tabs.tuner',
    'sax_table': 'intonation.ui.table',
    'sax_theme': 'intonation.ui.theme',
    'sax_toolbar': 'intonation.ui.toolbar',
    'sax_translator': 'intonation.i18n.translator',
    'sax_widgets': 'intonation.ui.widgets',
    'sax_yin': 'intonation.core.yin',
}
