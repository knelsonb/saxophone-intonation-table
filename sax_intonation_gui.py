#!/usr/bin/env python3
"""
Saxophon-Intonationsanalysator – GUI
=====================================
Abhaengigkeiten (Ubuntu / venv):
    pip install PyQt6 numpy sounddevice reportlab

Starten:
    python3 sax_intonation_gui.py
"""

import os
import sys
import math
import threading
import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QSizePolicy, QFileDialog, QMessageBox,
    QAbstractItemView, QGroupBox, QSplitter,
    QDialog, QLineEdit, QDialogButtonBox, QFormLayout,
    QStyledItemDelegate, QMenu, QCheckBox, QSpinBox, QToolButton,
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QRectF, QPointF, QLocale, QByteArray,
)
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QIcon, QGuiApplication

from sax_intonation_log import MeasurementLog
from sax_intonation_chart import render_intonation_chart
import sax_instruments
from sax_instruments import (
    families as instrument_families,
    instruments_in,
    transp_map as build_transp_map,
    display_name as instrument_display_name,
    family_of as instrument_family_of,
    register_custom,
)
import sax_config

APP_NAME = 'Intonation Analyzer'
APP_VERSION = '0.5.7.8'

# v0.5.4: AudioEngine + pitch detection + filter presets live in their own
# module so the engine has a state machine, host-API fallback chain, and
# hot-plug poller without dragging the GUI through every test path. The
# names we re-export below keep the rest of this file's imports stable.
from sax_audio_engine import (
    AudioEngine,
    AudioEngineState,
    AudioEngineError,
    AudioEngineDiagnostics,
    DeviceInfo,
    DeviceSelection,
    FILTER_PRESETS as _FILTER_PRESETS_EXT,
    FILTER_MODE_DEFAULT as _FILTER_MODE_DEFAULT_EXT,
    HOP_MS as _HOP_MS_EXT,
    BLOCK_MS as _BLOCK_MS_EXT,
    MIN_FREQ as _MIN_FREQ_EXT,
    MAX_FREQ as _MAX_FREQ_EXT,
    A4_DEFAULT as _A4_DEFAULT_EXT,
    DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE_EXT,
    DEFAULT_HOP_SIZE as _DEFAULT_HOP_SIZE_EXT,
    DEFAULT_BLOCK_SIZE as _DEFAULT_BLOCK_SIZE_EXT,
    SAMPLERATE_PREF_VALUES as _SAMPLERATE_PREF_VALUES_EXT,
    SAMPLERATE_CANDIDATES as _SAMPLERATE_CANDIDATES_EXT,
    cents_dev as _cents_dev_ext,
    query_input_devices,
    VENDOR_REGEX,
)


# =============================================================================
# I18N – alle UI-Strings an einem Ort
# =============================================================================
STRINGS = {
    'de': {
        # Fenster
        'window_title': 'Intonations-Analysator',
        # Gruppen
        'grp_instrument': 'Instrument',
        'grp_family':       'Familie',
        'grp_subinstrument':'Modell',
        'grp_nickname':     'Spitzname',
        'allow_oor':        'Auf Instrumentenumfang beschränken',
        'grp_layout':       'Layout',
        'layout_auto':      'Auto',
        'layout_single':    'Liste',
        'layout_matrix':    'Raster',
        'allow_oor_tip':    'Aktiviert: Töne au\u00dferhalb des Instrumentenumfangs werden ignoriert.\nDeaktiviert (Standard): Jeder gespielte Ton wird aufgezeichnet; der Umfang dient nur als Anzeigehilfe.',
        'nickname_tip':     'Spitzname (z.B. "Tenor #1")',
        'grp_filter':       'Reaktion',
        'filter_fast':      'Schnell',
        'filter_normal':    'Normal',
        'filter_slow':      'Langsam',
        'filter_tip':       'Tonhöhen-Filter.\nSchnell: minimale Glättung, reaktionsfreudige Anzeige.\nNormal (Standard): ausgewogen.\nLangsam: starke Glättung, ideal für lange Töne und Stimmanalyse.',
        'min_n_label':      'Mindest-Messungen pro Ton:',
        'min_n_tip':        'Töne mit weniger als dieser Anzahl Messungen werden ausgeblendet.\nVerhindert, dass kurze Versehen die Tabelle füllen.',
        # Diagnose-Panel (Spektrogramm + Live-Werte)
        'show_diagnostics':     'Spektrumanalysator & Diagnose anzeigen',
        'show_diagnostics_tip': 'Blendet einen Live-Spektrumanalysator und ein Diagnose-Feld unter dem Stimmgerät ein.\nNützlich zum Beobachten von Obertönen und zur Fehlersuche.',
        'spectro_title':        'Spektrumanalysator',
        'data_panel_title':     'Diagnose',
        'data_device':          'Audio-Eingang',
        'data_samplerate':      'Abtastrate',
        'data_blocksize':       'Blockgröße',
        'data_hopsize':         'Hop-Größe',
        'data_freqrange':       'Frequenzbereich',
        'data_filter_mode':     'Filtermodus',
        'data_filter_params':   'Filterparameter',
        'data_a4':              'Kammerton A',
        'data_rms':             'Pegel (dBFS)',
        'data_aperiodicity':    'YIN-Aperiodizität',
        'data_freq':            'Frequenz',
        'data_midi':            'MIDI (gehalten)',
        'data_notes_count':     'Töne seit Start',
        'data_hotplug_poller':  'Hotplug-Prüfung',
        'data_hotplug_value':       'alle {interval}s (zuletzt: {when})',
        'data_hotplug_value_never': 'alle {interval}s (—)',
        'audio_disabled':       'Audio deaktiviert',
        'custom_label':     '+  Eigenes …',
        'custom_dlg_title': 'Eigenes Instrument',
        'custom_dlg_info':  ('Ein eigenes Instrument hinzufügen. Die Transposition '
                              'ist die Differenz in Halbtönen vom gegriffenen C '
                              'zum klingenden Ton (Bb-Tenor = -2, Eb-Alt = +3).'),
        'custom_lbl_name':  'Anzeigename:',
        'custom_lbl_transp':'Transposition (Halbtöne):',
        'custom_err_name':  'Bitte einen Namen angeben.',
        'welcome_title':    'Willkommen beim Intonations-Analysator',
        'welcome_info':     ('Beim Spielen werden Cent-Abweichungen erkannt und '
                              'pro Ton statistisch ausgewertet.\n\n'
                              'Möchtest du den Verlauf zwischen Sitzungen speichern, '
                              'damit du später daraus CSVs und Diagramme erzeugen kannst?'),
        'welcome_persist':  'Messdatenverlauf dauerhaft speichern',
        'welcome_path':     'Die Datei landet unter:\n{path}',
        'welcome_continue': 'Loslegen',
        'welcome_skip':     'Später entscheiden',
        'csv_mode_per_nickname_note': 'Pro Spitzname und Ton (z.B. zwei Tenöre vergleichen)',
        'csv_nick_label':   'Spitzname:',
        'csv_all_nicks':    'Alle Spitznamen',
        'grp_display':    'Tondarstellung',
        'grp_a4':         'Kammerton A',
        'grp_language':   'Sprache',
        # Instrument-Namen
        'instr_eb_alto':    'Eb-Sax · Alt',
        'instr_eb_bari':    'Eb-Sax · Bariton',
        'instr_bb_tenor':   'Bb-Sax · Tenor',
        'instr_bb_soprano': 'Bb-Sax · Sopran',
        'instr_bb_bass':    'Bb-Sax · Bass',
        'instr_c':          'C-Instrument',
        # Anzeige-Modus
        'disp_griff':    'Gegriffene Töne',
        'disp_klingend': 'Klingende Töne',
        # Buttons
        'btn_autotune': '\U0001f3af  Kammerton ermitteln',
        'btn_reset':    '\u21ba  Reset',
        'btn_stop':     '\u23f8  Aufnahme pausieren',
        'btn_start':    '\u23fa  Aufnahme starten',
        'btn_txt':      '\u2b07  Export TXT',
        'btn_pdf':      '\u2b07  Export PDF',
        'btn_csv':      '\u2b07  Export CSV',
        'btn_chart':    '\ud83d\uddbc  Diagramm (PNG)',
        'btn_import':   '\u2b06  CSV importieren',
        # CSV-Import
        'csv_import_title':   'CSV importieren',
        'csv_import_saved':   '{runs} L\u00e4ufe und {meas} Messungen importiert.',
        'csv_import_empty':   'Keine neuen Datens\u00e4tze importiert (vielleicht bereits geladen?).',
        'csv_import_badhdr':  'Diese CSV stammt nicht aus dem Rohdaten-Export.',
        # Diagramm
        'chart_save_title':   'Diagramm speichern',
        'chart_filter':       'PNG-Bilder (*.png)',
        'chart_saved':        'Diagramm gespeichert:\n{path}',
        'chart_no_data':      'Keine Messdaten zum Darstellen.',
        'chart_title':        'Intonationsanalyse',
        'chart_subtitle':     '{instr}  \u00b7  A = {a4:.0f} Hz  \u00b7  {dt}',
        'chart_subtitle_id':  '{instr}  \u00b7  {maker} {model}  \u00b7  A = {a4:.0f} Hz  \u00b7  {dt}',
        'chart_footer':       'Aktuelle Sitzung: {notes} T\u00f6ne, {total} Messungen  \u00b7  Balken: Mittelwert, Whisker: \u00b11\u03c3',
        # CSV-Export
        'csv_dialog_title':  'CSV exportieren',
        'csv_dialog_info':   ('W\u00e4hle, wie die geloggten Messungen in der '
                              'CSV-Datei zusammengefasst werden sollen.'),
        'csv_mode_label':    'Aufteilung:',
        'csv_run_label':     'Lauf:',
        'csv_instr_label':   'Instrument:',
        'csv_all_runs':      'Alle L\u00e4ufe',
        'csv_all_instruments': 'Alle Instrumente',
        'csv_mode_raw':                 'Rohdaten (eine Zeile pro Messung)',
        'csv_mode_per_run_note':        'Pro Lauf und Ton (Mittel/Streuung)',
        'csv_mode_per_instrument_note': 'Pro Instrument und Ton (\u00fcber L\u00e4ufe)',
        'csv_mode_instrument_avg':      'Ein Instrument, je Ton gemittelt',
        'csv_mode_overall_per_note':    'Gesamtmittel je Ton',
        'csv_summary':       'Log: {n} Messungen in {runs} L\u00e4ufen',
        'csv_save_title':    'CSV speichern',
        'csv_filter':        'CSV-Dateien (*.csv)',
        'csv_saved':         'CSV gespeichert ({rows} Zeilen):\n{path}',
        'csv_no_data':       'Keine Messdaten im Log.',
        'csv_need_instr':    'F\u00fcr diese Aufteilung muss ein Instrument gew\u00e4hlt werden.',
        # Tabellen-Header
        'col_fingered':  'Gegriffener Ton',
        'col_sounding':  'Klingender Ton',
        'col_mean':      '\u00d8 Abw. (ct)',
        'col_std':       '\u03c3 (ct)',
        'col_n':         'N',
        'col_tendency':  'Tendenz',
        # Kontextmenü Tabelle
        'ctx_discard':   'Messungen für diesen Ton löschen',
        'ctx_discard_confirm': 'Alle Messungen für {note} löschen?',
        # Status
        'no_signal':     'Kein Signal',
        'status_fmt':    'Gegriffen: {fingered}   Klingend: {sounding}   {freq:.1f} Hz   {cents_str} ct   (A={a4:.0f} Hz)',
        # Tabellen-Label
        'table_title':   'Intonationstabelle',
        'table_summary': 'Intonationstabelle  \u2013  {notes} Töne  |  {total} Messungen',
        'table_empty_hint': 'Intonationstabelle  \u2013  spiel einen Ton, dann erscheinen hier Mittelwert und Standardabweichung pro Ton.',
        'table_matrix_title': 'Intonationsmatrix  \u2013  {played} von {total} Z\u00e4hlen gespielt',
        'matrix_oct_label':   'Okt {n}',
        'matrix_oct_rel_label': 'Okt {n:+d}',
        'matrix_oct_center':    'Okt 0 (Mitte)',
        # Reset-Dialog
        'reset_title':   'Reset',
        'reset_msg':     'Alle Messungen zurücksetzen?',
        # Audio-Fehler
        'audio_error_title': 'Keine Audio-Eingabe',
        'audio_error':   ('Die Bibliothek \u00bbsounddevice\u00ab ist nicht verf\u00fcgbar \u2014 '
                          'die Live-Tonh\u00f6henerkennung ist deaktiviert.\n'
                          'Gespeicherte CSVs lassen sich weiterhin \u00f6ffnen und anzeigen.\n\n'
                          'Audio aktivieren:\n'
                          '  Windows / macOS:   pip install sounddevice\n'
                          '  Linux (Debian/Ubuntu):  sudo apt install portaudio19-dev && pip install sounddevice\n\n'
                          'Danach das Programm neu starten.'),
        # Autotune
        'autotune_title':      'Kammerton ermitteln',
        'autotune_nodata':     'Bitte mindestens 3 Töne mit je \u2265 5 Messungen spielen,\nbevor der optimale Kammerton berechnet werden kann.',
        'autotune_result':     ('<b>Ergebnis der Kammertonanalyse</b><br><br>'
                                'Töne ausgewertet: <b>{notes}</b> (je \u2265 5 Messungen)<br>'
                                'Mittlere Abweichung (gewichtet): <b>{sign}{offset:.1f} ct</b><br>'
                                'Arithm. Mittel (zur Kontrolle): {sign2}{mean:.1f} ct<br><br>'
                                'Aktueller Kammerton: <b>{current:.0f} Hz</b><br>'
                                'Optimaler Kammerton: <b>{optimal:.2f} Hz</b>  \u2192  gerundet <b>{rounded} Hz</b><br><br>'),
        'autotune_clamp':      ('<i>Hinweis: {rounded} Hz liegt au\u00dferhalb des einstellbaren Bereichs '
                                '(430\u2013450 Hz). Es wird {clamped} Hz gesetzt.</i><br><br>'),
        'autotune_confirm':    'Soll der Kammerton auf <b>{clamped} Hz</b> gesetzt werden?<br>'
                               '<small>(Alle Messungen werden dabei zurückgesetzt)</small>',
        # Export
        'txt_save_title':  'TXT speichern',
        'txt_filter':      'Textdateien (*.txt)',
        'txt_saved':       'TXT gespeichert:\n{path}',
        'txt_header':      'INTONATIONS-ANALYSATOR',
        'txt_instr':       'Instrument : {name}',
        'txt_transp':      'Transpos.  : gegriffenes C klingt {note}',
        'txt_no_transp':   'Keine Transposition',
        'txt_a4':          'Kammerton   : A = {hz:.0f} Hz',
        'txt_date':        'Datum      : {dt}',
        'txt_col_header':  '{fingered:<12} {sounding:<12} {mean:>8} {std:>8} {n:>6}  Tendenz',
        'txt_col_mean':    '\u00d8 (ct)',
        'txt_col_std':     '\u03c3 (ct)',
        'txt_total':       'Gesamt: {total} Messungen  |  {notes} Töne',
        'pdf_save_title':  'PDF speichern',
        'pdf_filter':      'PDF-Dateien (*.pdf)',
        'pdf_saved':       'PDF gespeichert:\n{path}',
        'pdf_title':       'Intonations-Analysator',
        'pdf_transp':      'Transposition: gegriffenes C klingt {note} (+{n} Halbtöne)',
        'pdf_no_transp':   'Keine Transposition (C-Instrument)',
        'pdf_a4':          'Kammerton: A = {hz:.0f} Hz',
        'pdf_created':     'Erstellt: {dt}',
        'pdf_summary':     '{notes} Töne  |  {total} Messungen gesamt',
        'pdf_no_data':     'Keine Messdaten vorhanden.',
        'pdf_col_finger':  'Gegriffen',
        'pdf_col_sound':   'Klingend',
        'pdf_col_mean':    '\u00d8 Abw. (ct)',
        'pdf_col_std':     '\u03c3 (ct)',
        'pdf_col_n':       'N',
        'pdf_col_tend':    'Tendenz',
        'err_title':       'Fehler',
        'export_title':    'Export',
        'reportlab_err':   'reportlab nicht installiert.\npip install reportlab',
        # Instrument-Modell-Dialog
        'model_dialog_title':  'Instrumentangabe',
        'model_dialog_info':   'Diese Angaben erscheinen im Export (optional).',
        'model_label_maker':   'Hersteller:',
        'model_label_model':   'Modell:',
        'model_placeholder_maker': 'z.B. Yamaha, Selmer, Yanagisawa …',
        'model_placeholder_model': 'z.B. YAS-280, Mark VI, A-901 …',
        'txt_maker':       'Hersteller  : {maker}',
        'txt_model':       'Modell      : {model}',
        'pdf_maker':       'Hersteller: {maker}',
        'pdf_model':       'Modell: {model}',
        # Instr long names (für Export)
        'instr_long_eb_alto':    'Eb-Saxophon  (Alt)',
        'instr_long_eb_bari':    'Eb-Saxophon  (Bariton)',
        'instr_long_bb_tenor':   'Bb-Saxophon  (Tenor)',
        'instr_long_bb_soprano': 'Bb-Saxophon  (Sopran)',
        'instr_long_bb_bass':    'Bb-Saxophon  (Bass)',
        'instr_long_c':          'C-Instrument',
        # Transposition info chip
        'transp_info_eb': 'gegriffenes C  \u2192  klingt Eb',
        'transp_info_bb': 'gegriffenes C  \u2192  klingt Bb',
        'transp_info_c':  'keine Transposition',
        # v0.5.4 audio chip / picker / recovery banners
        'audio_chip_label':        'AUDIO-EINGANG',
        'audio_chip_none':         'Kein Ger\u00e4t',
        'audio_chip_opening':      '\u00d6ffne \u2026',
        'audio_chip_tip':          'Audioeingang ausw\u00e4hlen oder \u00fcberpr\u00fcfen',
        'audio_picker_title':      'Audioeingang',
        'audio_picker_rescan':     '\u21bb  Erneut suchen',
        'audio_picker_use':        'Ger\u00e4t verwenden',
        'audio_picker_cancel':     'Abbrechen',
        'audio_picker_show_all':   'Alle Host-APIs anzeigen',
        'audio_picker_prefer_ks':  'WDM-KS (niedrige Latenz, kann blockieren) bevorzugen',
        'audio_picker_current':    'aktuell',
        'audio_picker_apis_more':  '{n} APIs',
        'audio_picker_no_devices': 'Keine Eingabeger\u00e4te gefunden.',
        'audio_picker_row_meta':   '{api} \u00b7 {ch} Ch \u00b7 {sr:.0f} Hz',
        'audio_banner_no_device':  'Kein Audioeingang gefunden. Mikrofon oder Interface anschlie\u00dfen und \u00bbErneut versuchen\u00ab dr\u00fccken.',
        'audio_banner_disconnect': 'Audioger\u00e4t getrennt: "{name}". Anderes Ger\u00e4t w\u00e4hlen oder wieder anschlie\u00dfen und \u00bbErneut versuchen\u00ab dr\u00fccken.',
        'audio_banner_busy':       '"{name}" ist belegt \u2014 eine andere App nutzt das Ger\u00e4t. App schlie\u00dfen und \u00bbErneut versuchen\u00ab dr\u00fccken, oder anderes Ger\u00e4t w\u00e4hlen.',
        'audio_banner_unsupported_rate': 'Abtastrate wird vom Ger\u00e4t nicht unterst\u00fctzt. Anderes Ger\u00e4t w\u00e4hlen oder \u00bbErneut versuchen\u00ab dr\u00fccken.',
        'audio_banner_unknown':    'Audio-Fehler: {msg}',
        'audio_banner_retry':      'Erneut versuchen',
        'audio_banner_pick':       'Anderes Ger\u00e4t w\u00e4hlen',
        'audio_toast_interface':   'Neues Interface gefunden: {name}',
        'audio_toast_switch':      'Wechseln',
        'audio_toast_dismiss':     'Ignorieren',
        'audio_sr_notice':         'Audio l\u00e4uft mit {sr} Hz \u2014 das Ger\u00e4t unterst\u00fctzt 44100 Hz nicht.',
        # v0.5.5 \u2014 per-instrument range editor
        'gear_tip':                'Erlaubte T\u00f6ne f\u00fcr dieses Instrument bearbeiten',
        'range_editor_title':      'Tonumfang bearbeiten \u2014 {name}',
        'range_lo_label':          'Tiefster Griff-Ton (MIDI)',
        'range_hi_label':          'H\u00f6chster Griff-Ton (MIDI)',
        'range_lo_label_sounding': 'Tiefster Klingend-Ton (MIDI)',
        'range_hi_label_sounding': 'H\u00f6chster Klingend-Ton (MIDI)',
        'range_mode_note_griff':   'Werte als Griff-Tonh\u00f6hen (im Speicher kanonisch).',
        'range_mode_note_sound':   'Werte als klingende Tonh\u00f6hen; gespeichert wird kanonisch in Griffnotation.',
        'range_preview_fmt':       'Bereich: {lo_name} \u2013 {hi_name} ({semis} Halbt\u00f6ne, {octs:.1f} Oktaven)',
        'range_invalid':           'Ung\u00fcltig: tief darf nicht \u00fcber hoch liegen, beide m\u00fcssen zwischen 0 und 127 sein.',
        'range_save':              'Speichern',
        'range_restore_default':   'Standard wiederherstellen',
        'range_restore_fallback':  'Auf ({lo}, {hi}) zur\u00fccksetzen',
        'range_cancel':            'Abbrechen',
        # v0.5.6 — sample rate picker + adaptive cent precision diagnostics
        'samplerate_label':        'Abtastrate',
        'samplerate_auto':         'Auto (empfohlen)',
        'samplerate_unsupported':  'Dieses Gerät unterstützt {rate} Hz nicht. Auf Auto zurückgesetzt.',
        'data_rate_requested':     'Angeforderte Rate',
        'data_rate_negotiated':    'Ausgehandelte Rate',
        'data_block_hop':          'Block / Hop',
        'data_halfcent_floor':     '½-¢-Grenze bei A4',
    },
    'en': {
        'window_title': 'Intonation Analyzer',
        'grp_instrument': 'Instrument',
        'grp_family':       'Family',
        'grp_subinstrument':'Model',
        'grp_nickname':     'Nickname',
        'allow_oor':        'Filter to instrument range',
        'grp_layout':       'Layout',
        'layout_auto':      'Auto',
        'layout_single':    'List',
        'layout_matrix':    'Grid',
        'allow_oor_tip':    'Off (default): every played note is recorded; the instrument range is shown only as a display guide.\nOn: notes outside the instrument range are ignored.',
        'nickname_tip':     'Nickname (e.g. "Tenor #1")',
        'grp_filter':       'Response',
        'filter_fast':      'Fast',
        'filter_normal':    'Normal',
        'filter_slow':      'Slow',
        'filter_tip':       'Pitch-detection smoothing.\nFast: minimal smoothing, snappy tuner.\nNormal (default): balanced.\nSlow: heavy smoothing, ideal for long tones and tuning analysis.\nAll modes truncate attack and release transients.',
        'min_n_label':      'Min measurements per note:',
        'min_n_tip':        'Notes with fewer than this many measurements are hidden.\nKeeps brief slips out of the analysis.',
        # Diagnostics panel (spectrogram + live values)
        'show_diagnostics':     'Show spectrum analyzer & diagnostics',
        'show_diagnostics_tip': 'Reveals a live spectrum analyzer and a diagnostics readout below the tuner.\nUseful for inspecting overtones and troubleshooting.',
        'spectro_title':        'Spectrum analyzer',
        'data_panel_title':     'Diagnostics',
        'data_device':          'Audio input',
        'data_samplerate':      'Sample rate',
        'data_blocksize':       'Block size',
        'data_hopsize':         'Hop size',
        'data_freqrange':       'Frequency range',
        'data_filter_mode':     'Filter mode',
        'data_filter_params':   'Filter params',
        'data_a4':              'Concert pitch A',
        'data_rms':             'Level (dBFS)',
        'data_aperiodicity':    'YIN aperiodicity',
        'data_freq':            'Frequency',
        'data_midi':            'MIDI (locked)',
        'data_notes_count':     'Notes since start',
        'data_hotplug_poller':  'Hot-plug check',
        'data_hotplug_value':       'every {interval}s (last: {when})',
        'data_hotplug_value_never': 'every {interval}s (—)',
        'audio_disabled':       'Audio disabled',
        'custom_label':     '+  Custom …',
        'custom_dlg_title': 'Custom instrument',
        'custom_dlg_info':  ('Add a custom instrument. Transposition is the '
                              'number of semitones from written C to sounding '
                              'pitch (Bb tenor = -2, Eb alto = +3).'),
        'custom_lbl_name':  'Display name:',
        'custom_lbl_transp':'Transposition (semitones):',
        'custom_err_name':  'Please enter a name.',
        'welcome_title':    'Welcome to Intonation Analyzer',
        'welcome_info':     ('As you play, cent deviations are detected and '
                              'aggregated per note.\n\n'
                              'Would you like to save your measurement history '
                              'between sessions, so you can export CSVs and '
                              'charts from past data?'),
        'welcome_persist':  'Save measurement history to disk',
        'welcome_path':     'The file will live at:\n{path}',
        'welcome_continue': 'Get started',
        'welcome_skip':     'Decide later',
        'csv_mode_per_nickname_note': 'Per nickname and note (e.g. compare two tenors)',
        'csv_nick_label':   'Nickname:',
        'csv_all_nicks':    'All nicknames',
        'grp_display':    'Note Display',
        'grp_a4':         'Concert Pitch A',
        'grp_language':   'Language',
        'instr_eb_alto':    'Eb Sax · Alto',
        'instr_eb_bari':    'Eb Sax · Baritone',
        'instr_bb_tenor':   'Bb Sax · Tenor',
        'instr_bb_soprano': 'Bb Sax · Soprano',
        'instr_bb_bass':    'Bb Sax · Bass',
        'instr_c':          'C Instrument',
        'disp_griff':    'Fingered Notes',
        'disp_klingend': 'Sounding Notes',
        'btn_autotune': '\U0001f3af  Detect Concert Pitch',
        'btn_reset':    '\u21ba  Reset',
        'btn_stop':     '\u23f8  Pause Recording',
        'btn_start':    '\u23fa  Start Recording',
        'btn_txt':      '\u2b07  Export TXT',
        'btn_pdf':      '\u2b07  Export PDF',
        'btn_csv':      '\u2b07  Export CSV',
        'btn_chart':    '\ud83d\uddbc  Chart (PNG)',
        'btn_import':   '\u2b06  Import CSV',
        'csv_import_title':   'Import CSV',
        'csv_import_saved':   'Imported {runs} runs and {meas} measurements.',
        'csv_import_empty':   'No new records imported (already loaded?).',
        'csv_import_badhdr':  'This CSV is not a raw-mode export.',
        'chart_save_title':   'Save chart',
        'chart_filter':       'PNG images (*.png)',
        'chart_saved':        'Chart saved:\n{path}',
        'chart_no_data':      'No measurement data to chart.',
        'chart_title':        'Intonation Analysis',
        'chart_subtitle':     '{instr}  \u00b7  A = {a4:.0f} Hz  \u00b7  {dt}',
        'chart_subtitle_id':  '{instr}  \u00b7  {maker} {model}  \u00b7  A = {a4:.0f} Hz  \u00b7  {dt}',
        'chart_footer':       'Current session: {notes} notes, {total} measurements  \u00b7  Bars: mean, whiskers: \u00b11\u03c3',
        'csv_dialog_title':  'Export CSV',
        'csv_dialog_info':   ('Choose how the logged measurements should be '
                              'summarised in the CSV file.'),
        'csv_mode_label':    'Slice by:',
        'csv_run_label':     'Run:',
        'csv_instr_label':   'Instrument:',
        'csv_all_runs':      'All runs',
        'csv_all_instruments': 'All instruments',
        'csv_mode_raw':                 'Raw (one row per measurement)',
        'csv_mode_per_run_note':        'Per run and note (mean/std)',
        'csv_mode_per_instrument_note': 'Per instrument and note (across runs)',
        'csv_mode_instrument_avg':      'One instrument, per-note average',
        'csv_mode_overall_per_note':    'Overall mean per note',
        'csv_summary':       'Log: {n} measurements across {runs} runs',
        'csv_save_title':    'Save CSV',
        'csv_filter':        'CSV files (*.csv)',
        'csv_saved':         'CSV saved ({rows} rows):\n{path}',
        'csv_no_data':       'No measurement data in the log.',
        'csv_need_instr':    'This slice mode requires choosing one instrument.',
        'col_fingered':  'Fingered Note',
        'col_sounding':  'Sounding Note',
        'col_mean':      '\u00d8 Dev. (ct)',
        'col_std':       '\u03c3 (ct)',
        'col_n':         'N',
        'col_tendency':  'Tendency',
        # Table context menu
        'ctx_discard':   'Delete measurements for this note',
        'ctx_discard_confirm': 'Delete all measurements for {note}?',
        'no_signal':     'No signal',
        'status_fmt':    'Fingered: {fingered}   Sounding: {sounding}   {freq:.1f} Hz   {cents_str} ct   (A={a4:.0f} Hz)',
        'table_title':   'Intonation Table',
        'table_summary': 'Intonation Table  \u2013  {notes} notes  |  {total} measurements',
        'table_empty_hint': 'Intonation Table  \u2013  play a note to begin; per-note mean and standard deviation appear here as you play.',
        'table_matrix_title': 'Intonation matrix  \u2013  {played} of {total} cells played',
        'matrix_oct_label':   'Oct {n}',
        'matrix_oct_rel_label': 'Oct {n:+d}',
        'matrix_oct_center':    'Oct 0 (middle)',
        'reset_title':   'Reset',
        'reset_msg':     'Reset all measurements?',
        'audio_error_title': 'No audio input',
        'audio_error':   ('The "sounddevice" library is not available \u2014 live '
                          'pitch detection is disabled.\n'
                          'You can still open and view saved CSVs.\n\n'
                          'To enable audio:\n'
                          '  Windows / macOS:   pip install sounddevice\n'
                          '  Linux (Debian/Ubuntu):  sudo apt install portaudio19-dev && pip install sounddevice\n\n'
                          'Then restart the program.'),
        'autotune_title':      'Detect Concert Pitch',
        'autotune_nodata':     'Please play at least 3 notes with \u2265 5 measurements each\nbefore the optimal concert pitch can be calculated.',
        'autotune_result':     ('<b>Concert Pitch Analysis Result</b><br><br>'
                                'Notes evaluated: <b>{notes}</b> (each \u2265 5 measurements)<br>'
                                'Weighted mean deviation: <b>{sign}{offset:.1f} ct</b><br>'
                                'Arithmetic mean (reference): {sign2}{mean:.1f} ct<br><br>'
                                'Current concert pitch: <b>{current:.0f} Hz</b><br>'
                                'Optimal concert pitch: <b>{optimal:.2f} Hz</b>  \u2192  rounded <b>{rounded} Hz</b><br><br>'),
        'autotune_clamp':      ('<i>Note: {rounded} Hz is outside the selectable range '
                                '(430\u2013450 Hz). {clamped} Hz will be used instead.</i><br><br>'),
        'autotune_confirm':    'Set concert pitch to <b>{clamped} Hz</b>?<br>'
                               '<small>(All measurements will be reset)</small>',
        'txt_save_title':  'Save TXT',
        'txt_filter':      'Text files (*.txt)',
        'txt_saved':       'TXT saved:\n{path}',
        'txt_header':      'INTONATION ANALYZER',
        'txt_instr':       'Instrument : {name}',
        'txt_transp':      'Transpos.  : fingered C sounds as {note}',
        'txt_no_transp':   'No transposition',
        'txt_a4':          'Concert pitch: A = {hz:.0f} Hz',
        'txt_date':        'Date       : {dt}',
        'txt_col_header':  '{fingered:<12} {sounding:<12} {mean:>8} {std:>8} {n:>6}  Tendency',
        'txt_col_mean':    '\u00d8 (ct)',
        'txt_col_std':     '\u03c3 (ct)',
        'txt_total':       'Total: {total} measurements  |  {notes} notes',
        'pdf_save_title':  'Save PDF',
        'pdf_filter':      'PDF files (*.pdf)',
        'pdf_saved':       'PDF saved:\n{path}',
        'pdf_title':       'Intonation Analyzer',
        'pdf_transp':      'Transposition: fingered C sounds as {note} (+{n} semitones)',
        'pdf_no_transp':   'No transposition (C instrument)',
        'pdf_a4':          'Concert pitch: A = {hz:.0f} Hz',
        'pdf_created':     'Created: {dt}',
        'pdf_summary':     '{notes} notes  |  {total} measurements total',
        'pdf_no_data':     'No measurement data available.',
        'pdf_col_finger':  'Fingered',
        'pdf_col_sound':   'Sounding',
        'pdf_col_mean':    '\u00d8 Dev. (ct)',
        'pdf_col_std':     '\u03c3 (ct)',
        'pdf_col_n':       'N',
        'pdf_col_tend':    'Tendency',
        'err_title':       'Error',
        'export_title':    'Export',
        'reportlab_err':   'reportlab not installed.\npip install reportlab',
        # Instrument model dialog
        'model_dialog_title':  'Instrument Details',
        'model_dialog_info':   'These details will appear in the export (optional).',
        'model_label_maker':   'Manufacturer:',
        'model_label_model':   'Model:',
        'model_placeholder_maker': 'e.g. Yamaha, Selmer, Yanagisawa …',
        'model_placeholder_model': 'e.g. YAS-280, Mark VI, A-901 …',
        'txt_maker':       'Manufacturer: {maker}',
        'txt_model':       'Model       : {model}',
        'pdf_maker':       'Manufacturer: {maker}',
        'pdf_model':       'Model: {model}',
        'instr_long_eb_alto':    'Eb Saxophone  (Alto)',
        'instr_long_eb_bari':    'Eb Saxophone  (Baritone)',
        'instr_long_bb_tenor':   'Bb Saxophone  (Tenor)',
        'instr_long_bb_soprano': 'Bb Saxophone  (Soprano)',
        'instr_long_bb_bass':    'Bb Saxophone  (Bass)',
        'instr_long_c':          'C Instrument',
        'transp_info_eb': 'fingered C  \u2192  sounds Eb',
        'transp_info_bb': 'fingered C  \u2192  sounds Bb',
        'transp_info_c':  'no transposition',
        # v0.5.4 audio chip / picker / recovery banners
        'audio_chip_label':        'AUDIO IN',
        'audio_chip_none':         'No device',
        'audio_chip_opening':      'Opening \u2026',
        'audio_chip_tip':          'Pick or inspect the audio input device',
        'audio_picker_title':      'Audio Input',
        'audio_picker_rescan':     '\u21bb  Rescan',
        'audio_picker_use':        'Use selected device',
        'audio_picker_cancel':     'Cancel',
        'audio_picker_show_all':   'Show all host APIs',
        'audio_picker_prefer_ks':  'Prefer WDM-KS (low latency, may lock device)',
        'audio_picker_current':    'current',
        'audio_picker_apis_more':  '{n} APIs',
        'audio_picker_no_devices': 'No input devices found.',
        'audio_picker_row_meta':   '{api} \u00b7 {ch} ch \u00b7 {sr:.0f} Hz',
        'audio_banner_no_device':  'No audio input found. Plug in a microphone or interface and tap Retry.',
        'audio_banner_disconnect': 'Audio device disconnected: "{name}". Pick a different device, or plug it back in and tap Retry.',
        'audio_banner_busy':       '"{name}" is busy \u2014 another app is using it. Close that app and tap Retry, or pick a different device.',
        'audio_banner_unsupported_rate': 'Sample rate not supported by this device. Pick a different device or tap Retry.',
        'audio_banner_unknown':    'Audio error: {msg}',
        'audio_banner_retry':      'Retry',
        'audio_banner_pick':       'Pick different device',
        'audio_toast_interface':   'New interface detected: {name}',
        'audio_toast_switch':      'Switch',
        'audio_toast_dismiss':     'Dismiss',
        'audio_sr_notice':         'Running at {sr} Hz \u2014 this device does not support 44100 Hz.',
        # v0.5.5 \u2014 per-instrument range editor
        'gear_tip':                'Edit allowed notes for this instrument',
        'range_editor_title':      'Edit range \u2014 {name}',
        'range_lo_label':          'Lowest fingered note (MIDI)',
        'range_hi_label':          'Highest fingered note (MIDI)',
        'range_lo_label_sounding': 'Lowest sounding note (MIDI)',
        'range_hi_label_sounding': 'Highest sounding note (MIDI)',
        'range_mode_note_griff':   'Values are fingered pitches (canonical on disk).',
        'range_mode_note_sound':   'Values are sounding pitches; saved canonically as fingered.',
        'range_preview_fmt':       'Range: {lo_name} \u2013 {hi_name} ({semis} semitones, {octs:.1f} octaves)',
        'range_invalid':           'Invalid: low must not exceed high, and both must be 0\u2013127.',
        'range_save':              'Save',
        'range_restore_default':   'Restore Default',
        'range_restore_fallback':  'Restore to ({lo}, {hi})',
        'range_cancel':            'Cancel',
        # v0.5.6 — sample rate picker + adaptive cent precision diagnostics
        'samplerate_label':        'Sample rate',
        'samplerate_auto':         'Auto (recommended)',
        'samplerate_unsupported':  'This device does not support {rate} Hz. Reverted to Auto.',
        'data_rate_requested':     'Requested rate',
        'data_rate_negotiated':    'Negotiated rate',
        'data_block_hop':          'Block / Hop',
        'data_halfcent_floor':     '½-¢ floor @ A4',
    },
}


# =============================================================================
# Konstanten & Musik-Logik
# =============================================================================
# Audio constants live in sax_audio_engine. The aliases below preserve
# the existing call sites in this file (spectrum widget, diagnostics
# panel, etc.) without rewriting every line. SAMPLE_RATE / HOP_SIZE /
# BLOCK_SIZE are the *defaults* — the live engine may negotiate a
# different sample rate at startup and rescale block sizes accordingly.
SAMPLE_RATE   = _DEFAULT_SAMPLE_RATE_EXT
HOP_SIZE      = _DEFAULT_HOP_SIZE_EXT
BLOCK_SIZE    = _DEFAULT_BLOCK_SIZE_EXT
MIN_FREQ      = _MIN_FREQ_EXT
MAX_FREQ      = _MAX_FREQ_EXT
A4_DEFAULT    = _A4_DEFAULT_EXT
HOP_MS        = _HOP_MS_EXT
BLOCK_MS      = _BLOCK_MS_EXT
_FILTER_PRESETS = _FILTER_PRESETS_EXT
FILTER_MODE_DEFAULT = _FILTER_MODE_DEFAULT_EXT
SAMPLERATE_PREF_VALUES = _SAMPLERATE_PREF_VALUES_EXT
SAMPLERATE_CANDIDATES = _SAMPLERATE_CANDIDATES_EXT

# v0.5.6: frequency-adaptive cent display precision.
# Parabolic interpolation in YIN pins tau to about 0.1 samples; one cent
# corresponds to tau * (ln(2)/1200) ~= tau * 5.78e-4 samples. The minimum
# resolvable cent step at frequency f and sample rate sr is therefore
# approximately 0.1 / (tau * 5.78e-4) = 173 * f / sr. We snap the
# displayed precision to one of {tenths, halves, wholes} so the readout
# never claims more resolution than the measurement actually delivers.
CENT_PREC_TENTHS_MAX = 0.3
CENT_PREC_HALVES_MAX = 0.7


def cent_precision_floor(freq_hz: float, sample_rate: int) -> float:
    """Return the minimum resolvable cent step at ``freq_hz`` given
    ``sample_rate``. See the module-top derivation."""
    sr = float(sample_rate) if sample_rate else float(_DEFAULT_SAMPLE_RATE_EXT)
    if sr <= 0 or freq_hz <= 0:
        return CENT_PREC_TENTHS_MAX
    return 173.0 * float(freq_hz) / sr


def format_cents(value_cents: float, freq_hz: float,
                 sample_rate: int) -> str:
    """Format a cent value at the precision the measurement supports.

    Always emits a sign so neutral readouts read "+0" / "+0.0" rather
    than the visually-jumpy bare "0". Negative-zero floats coerce to
    "+0..." via the explicit ``>= 0`` test.

    v0.5.7.2: guard against non-finite inputs. NoteStats.mean can be
    NaN after a reset race (empty vals → np.mean), and int(round(NaN))
    raises ValueError. Returning the canonical "–" placeholder matches
    the other no-data cells in the matrix paint path.
    """
    if (not math.isfinite(value_cents) or not math.isfinite(freq_hz)
            or freq_hz <= 0):
        return '–'
    floor_ct = cent_precision_floor(freq_hz, sample_rate)
    if floor_ct <= CENT_PREC_TENTHS_MAX:
        snapped = float(value_cents)
        sign = '+' if snapped >= 0 else '-'
        return f"{sign}{abs(snapped):.1f}"
    if floor_ct <= CENT_PREC_HALVES_MAX:
        snapped = round(float(value_cents) * 2.0) / 2.0
        sign = '+' if snapped >= 0 else '-'
        return f"{sign}{abs(snapped):.1f}"
    snapped = round(float(value_cents))
    sign = '+' if snapped >= 0 else '-'
    return f"{sign}{abs(int(snapped))}"

CHROMA = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F',
          'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

TRANSP     = {'eb': 3, 'bb': 2, 'c': 0}
# MIDI-Bereiche pro Saxophon-Typ (gegriffene Töne, klingende Noten)
# Bass-Sax (Bb):  klingt Bb0 (22) – F#3 (54)   → gegriffen A1–E4  (21–52)
# Bariton (Eb):   klingt Db2 (37) – Ab4 (68)   → gegriffen Bb2–F5 (46–65 + Höhe)
# Tenor (Bb):     klingt Ab2 (44) – Eb5 (75)   → gegriffen G3–D6  (55–74)
# Alt (Eb):       klingt Db3 (49) – Ab5 (80)   → gegriffen Bb3–F6 (58–77)
# Sopran (Bb):    klingt Ab3 (56) – Eb6 (87)   → gegriffen G4–D7  (67–86)
SAX_MIDI   = range(21, 92)   # E0–G6: deckt alle Saxophon-Typen inkl. Bass ab

# Transposition map is now derived from sax_instruments.transp_map(), which
# includes the original six saxophone/C-instrument keys plus everything else
# in the catalog. Rebuilt on demand whenever a custom instrument is added.
TRANSP_MAP: dict[str, int] = build_transp_map()


def _rebuild_transp_map() -> None:
    """Refresh TRANSP_MAP after a custom instrument is registered."""
    global TRANSP_MAP
    TRANSP_MAP = build_transp_map()


def freq_to_midi(f, a4=None):
    ref = a4 if a4 is not None else A4_DEFAULT
    return 69.0 + 12.0 * math.log2(f / ref)

def midi_note_name(m):
    return f"{CHROMA[m % 12]}{m // 12 - 1}"

def cents_dev(f, a4=None):
    mf = freq_to_midi(f, a4)
    mr = round(mf)
    return mr, (mf - mr) * 100.0


# =============================================================================
# YIN Pitch-Detektion — moved to sax_audio_engine.yin_pitch in v0.5.4.
# Local thin wrapper retained for any in-process call sites; the engine
# uses its own implementation directly.
# =============================================================================
from sax_audio_engine import yin_pitch as _yin_pitch_ext


def yin_pitch(sig, sr=SAMPLE_RATE, fmin=MIN_FREQ, fmax=MAX_FREQ,
              thr=None):
    return _yin_pitch_ext(sig, sr, fmin, fmax,
                          thr if thr is not None else 0.12)


# (legacy YIN body removed in v0.5.4 — see sax_audio_engine.yin_pitch)


# =============================================================================
# Messdaten
# =============================================================================
class NoteStats:
    def __init__(self):
        self.vals: list[float] = []
    def add(self, c): self.vals.append(c)
    @property
    def mean(self):
        # v0.5.7.3: snapshot vals before evaluating. A concurrent
        # clear/reassign could shrink the list between the truthiness
        # check and np.mean, returning NaN that then propagates into the
        # matrix delegate's paint path (drawRoundedRect with NaN width
        # is undefined Qt behaviour). The list is bounded (a few dozen
        # entries per note in practice), so the copy is cheap.
        vals = list(self.vals)
        return float(np.mean(vals)) if vals else 0.0
    @property
    def std(self):
        # Same snapshot rationale as mean — np.std on a list that gets
        # cleared mid-call returns NaN.
        vals = list(self.vals)
        return float(np.std(vals)) if len(vals) > 1 else 0.0
    @property
    def n(self):    return len(self.vals)


# =============================================================================
# Audio-Engine — moved to sax_audio_engine.AudioEngine in v0.5.4.
# =============================================================================


# =============================================================================
# Tuner-Widget
# =============================================================================
class TunerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.cents  = 0.0
        self.freq   = 0.0
        self.note   = ''
        self.active = False
        self._decay = 0.0
        # v0.5.6: live engine sample rate, set by MainWindow whenever the
        # engine successfully opens. Used to render cents at a precision
        # the measurement can actually support.
        self.sample_rate = _DEFAULT_SAMPLE_RATE_EXT
        t = QTimer(self)
        t.timeout.connect(self._fade)
        t.start(80)
        # Restored to 260 in v0.5.3 after feedback that the 180px tuner
        # cramped the needle / note / cents readout once the diagnostics
        # panel sat beneath it. The spectrum analyzer below has its own
        # min height and the vertical layout pushes diagnostics down the
        # left pane rather than compressing the tuner.
        self.setMinimumHeight(260)

    def set_note(self, note, freq, cents):
        self.note, self.freq, self.cents = note, freq, cents
        self.active = True
        self._decay = 1.0
        self.update()

    def _fade(self):
        if self.active:
            self._decay = max(0.0, self._decay - 0.04)
            if self._decay == 0.0:
                self.active = False
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(18, 18, 24))
        alpha = int(255 * max(0.15, self._decay))

        # Tonname
        if self.note:
            p.setFont(QFont('Monospace', 72, QFont.Weight.Bold))
            p.setPen(QColor(220, 220, 255, alpha))
            p.drawText(QRectF(0, 8, W, H * 0.45),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       self.note)

        # Frequenz
        if self.freq > 0:
            p.setFont(QFont('Monospace', 16))
            p.setPen(QColor(140, 140, 180, alpha))
            p.drawText(QRectF(0, H * 0.44, W, 30),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{self.freq:.1f} Hz")

        # Skala
        sy, sh = H * 0.60, 28
        sw, sx = W * 0.78, (W - W * 0.78) / 2
        mx = 50.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(40, 40, 55))
        p.drawRoundedRect(QRectF(sx, sy, sw, sh), 8, 8)

        zw = sw * (10.0 / mx) / 2
        p.setBrush(QColor(30, 90, 50, 140))
        p.drawRoundedRect(QRectF(sx + sw/2 - zw, sy, zw*2, sh), 6, 6)

        if self.active:
            norm = max(-1.0, min(1.0, self.cents / mx))
            nx   = sx + sw/2 + norm * sw/2
            if   abs(self.cents) <= 5:  nc = QColor(60,  220, 100, alpha)
            elif abs(self.cents) <= 15: nc = QColor(255, 200, 40,  alpha)
            else:                       nc = QColor(240, 70,  70,  alpha)
            p.setBrush(nc)
            p.drawRoundedRect(QRectF(nx - 4, sy - 6, 8, sh + 12), 4, 4)

        p.setPen(QColor(90, 90, 110))
        p.setFont(QFont('Monospace', 9))
        for ct in [-50, -25, 0, 25, 50]:
            nx = sx + sw/2 + (ct / mx) * sw/2
            p.drawLine(QPointF(nx, sy - 2), QPointF(nx, sy + sh + 2))
            p.drawText(QRectF(nx - 18, sy + sh + 4, 36, 16),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{'+' if ct>0 else ''}{ct}")

        if self.active:
            p.setFont(QFont('Monospace', 32, QFont.Weight.Bold))
            if   abs(self.cents) <= 5:  cc = QColor(60,  220, 100, alpha)
            elif abs(self.cents) <= 15: cc = QColor(255, 200, 40,  alpha)
            else:                       cc = QColor(240, 70,  70,  alpha)
            p.setPen(cc)
            p.drawText(QRectF(0, sy + sh + 26, W, 55),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{format_cents(self.cents, self.freq, self.sample_rate)} ct")
        p.end()


# =============================================================================
# Spektrumanalysator-Widget
# =============================================================================
class SpectrumAnalyzerWidget(QWidget):
    """Live spectrum analyzer (FFT magnitude vs. log frequency).

    Pulls slices of `AudioEngine._buf` on its own QTimer tick (~30 fps),
    runs a Hann-windowed rfft, bins magnitudes into log-spaced frequency
    buckets between 27 Hz and 4000 Hz, and paints them as a filled curve
    in dBFS. A peak-hold envelope decays at a fixed rate so transient
    spikes stay readable for a few hundred ms before sliding back down.

    The audio callback is never touched here — the timer reads `_buf`
    cooperatively and tolerates a hop of staleness. Display only, never
    gates or feeds back into pitch detection.
    """

    F_LO = 27.0
    F_HI = 4000.0
    DB_FLOOR = -80.0
    DB_CEIL  = -10.0
    N_BINS = 192          # horizontal resolution (frequency buckets)
    REFRESH_MS = 33       # ~30 fps
    # Peak hold decay in dB per second. 0.3 s "hang" then decay; we
    # approximate the hang implicitly by capping decay to roughly 100
    # dB/s, which means a 70 dB spike takes ~0.7 s to drop fully back
    # to the live curve. Plenty visible without smearing.
    PEAK_DECAY_DB_PER_S = 60.0

    def __init__(self, engine: 'AudioEngine | None'):
        super().__init__()
        self._engine = engine
        # Log-spaced bin edges used to bucket the rfft magnitudes.
        self._edges = np.logspace(math.log10(self.F_LO), math.log10(self.F_HI),
                                   self.N_BINS + 1)
        # Bin center frequencies, used for the X axis mapping.
        self._centers = np.sqrt(self._edges[:-1] * self._edges[1:])
        # Cached Hann window + bucket map. The engine may rebind _buf
        # to a different size when it negotiates a non-44100 sample
        # rate; _rebuild_for() regenerates the cache on the fly.
        self._window = np.hanning(BLOCK_SIZE).astype(np.float32)
        fft_freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)
        self._bucket = np.searchsorted(self._edges, fft_freqs) - 1
        self._bucket = np.clip(self._bucket, -1, self.N_BINS - 1)
        self._win_norm = float(np.sum(self._window) * 0.5)
        # Live curve and peak-hold envelope, both in dBFS.
        self._levels = np.full(self.N_BINS, self.DB_FLOOR, dtype=np.float32)
        self._peaks  = np.full(self.N_BINS, self.DB_FLOOR, dtype=np.float32)
        # Powers-of-two grid lines in Hz, labeled along the bottom.
        self._grid_hz = [32, 64, 128, 256, 512, 1024, 2048, 4096]

        self.setMinimumHeight(150)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.REFRESH_MS)

    def _rebuild_for(self, block_size: int, samplerate: int) -> None:
        """Regenerate window + bucket map for a new buffer size / rate.

        Called by _tick when the engine's buffer size shifts (sample-
        rate renegotiation at startup or device change)."""
        block_size = max(8, int(block_size))
        samplerate = max(1, int(samplerate))
        self._window = np.hanning(block_size).astype(np.float32)
        fft_freqs = np.fft.rfftfreq(block_size, d=1.0 / samplerate)
        self._bucket = np.searchsorted(self._edges, fft_freqs) - 1
        self._bucket = np.clip(self._bucket, -1, self.N_BINS - 1)
        self._win_norm = float(np.sum(self._window) * 0.5)

    def _tick(self) -> None:
        if not AUDIO_OK or self._engine is None:
            return
        # Pull a snapshot copy under the engine's lock so the audio
        # callback can't half-roll the buffer while we FFT it. Fixes
        # the v0.5.3 spectrum-widget data race documented in wave 1.
        try:
            buf = self._engine.get_buf_snapshot()
        except Exception:
            return
        if buf is None or len(buf) < 8:
            return
        # Engine may have reallocated _buf at a different sample rate;
        # if our cached window/bucket maps no longer match, rebuild.
        if buf.size != self._window.size:
            self._rebuild_for(buf.size,
                              int(getattr(self._engine, 'samplerate',
                                           SAMPLE_RATE)))
        # Bucket the rfft magnitudes into log-spaced bins, take the max
        # per bucket so narrow spikes survive the downsampling.
        windowed = buf * self._window
        mag = np.abs(np.fft.rfft(windowed)) / self._win_norm
        col = np.full(self.N_BINS, 1e-12, dtype=np.float32)
        valid = self._bucket >= 0
        # np.maximum.at handles repeated indices; per-bucket peak instead
        # of sum keeps the curve from drifting upward at the low end
        # where many fft bins fall into a single log bucket.
        np.maximum.at(col, self._bucket[valid], mag[valid])
        db = 20.0 * np.log10(col + 1e-12)
        db = np.clip(db, self.DB_FLOOR, 0.0)
        # Light temporal smoothing on the live curve — single-pole IIR
        # with a coefficient picked to settle in ~50 ms at 30 fps. The
        # peak-hold envelope is what makes spikes legible; the live
        # curve just needs to not strobe.
        self._levels = 0.6 * self._levels + 0.4 * db
        # Peak-hold: instantaneous capture, linear decay between ticks.
        decay = self.PEAK_DECAY_DB_PER_S * (self.REFRESH_MS / 1000.0)
        self._peaks = np.maximum(self._peaks - decay, self._levels)
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(10, 10, 16))
        if not AUDIO_OK or self._engine is None:
            p.setPen(QColor(140, 140, 160))
            p.setFont(QFont('Monospace', 11))
            p.drawText(QRectF(0, 0, W, H),
                       Qt.AlignmentFlag.AlignCenter,
                       'audio disabled')
            p.end()
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Plot area leaves room for axis labels along the bottom.
        margin_l = 4
        margin_r = 4
        margin_t = 4
        margin_b = 14
        x0 = margin_l
        y0 = margin_t
        x1 = max(W - margin_r, x0 + 1)
        y1 = max(H - margin_b, y0 + 1)
        plot_w = x1 - x0
        plot_h = y1 - y0

        log_lo = math.log10(self.F_LO)
        log_hi = math.log10(self.F_HI)
        log_span = log_hi - log_lo

        def x_of(freq: float) -> float:
            return x0 + (math.log10(freq) - log_lo) / log_span * plot_w

        def y_of(db: float) -> float:
            # DB_CEIL maps to top (y0), DB_FLOOR to bottom (y1).
            t = (db - self.DB_FLOOR) / (self.DB_CEIL - self.DB_FLOOR)
            t = max(0.0, min(1.0, t))
            return y1 - t * plot_h

        # Vertical gridlines at powers of two, with thin labels.
        p.setFont(QFont('Monospace', 8))
        grid_pen = QPen(QColor(40, 44, 60))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for hz in self._grid_hz:
            if hz < self.F_LO or hz > self.F_HI:
                continue
            gx = x_of(hz)
            p.drawLine(QPointF(gx, y0), QPointF(gx, y1))
        p.setPen(QColor(120, 130, 150))
        for hz in self._grid_hz:
            if hz < self.F_LO or hz > self.F_HI:
                continue
            gx = x_of(hz)
            label = f'{hz}' if hz < 1000 else f'{hz // 1000}k'
            p.drawText(QRectF(gx - 20, y1 + 1, 40, margin_b),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       label)

        # Horizontal dB gridline at -40 dBFS for visual reference.
        p.setPen(QPen(QColor(40, 44, 60), 1, Qt.PenStyle.DashLine))
        y_mid = y_of(-40.0)
        p.drawLine(QPointF(x0, y_mid), QPointF(x1, y_mid))

        # Live curve as a filled polygon under the line.
        from PyQt6.QtGui import QPolygonF
        poly = QPolygonF()
        poly.append(QPointF(x_of(self._centers[0]), y1))
        for i, f in enumerate(self._centers):
            poly.append(QPointF(x_of(float(f)), y_of(float(self._levels[i]))))
        poly.append(QPointF(x_of(self._centers[-1]), y1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(58, 156, 220, 110))
        p.drawPolygon(poly)

        # Bright line on top of the fill.
        line_pen = QPen(QColor(120, 200, 255))
        line_pen.setWidth(2)
        p.setPen(line_pen)
        prev = None
        for i, f in enumerate(self._centers):
            pt = QPointF(x_of(float(f)), y_of(float(self._levels[i])))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        # Peak-hold envelope as a thinner, paler line on top.
        peak_pen = QPen(QColor(240, 220, 140))
        peak_pen.setWidth(1)
        p.setPen(peak_pen)
        prev = None
        for i, f in enumerate(self._centers):
            pt = QPointF(x_of(float(f)), y_of(float(self._peaks[i])))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        p.end()


# Backwards-compatible alias. v0.5.2 referenced SpectrogramWidget; the
# rest of the file still uses the old name where harmless, but new
# constructions in the splitter use the analyzer class directly.
SpectrogramWidget = SpectrumAnalyzerWidget


# =============================================================================
# Diagnose-Panel
# =============================================================================
class DataPanelWidget(QWidget):
    """Read-only key:value readout of audio + engine state.

    The values come straight from constants and from the AudioEngine's
    last_* attributes, refreshed every 250 ms. Diagnostic only; never
    influences playback or detection."""

    REFRESH_MS = 250

    def __init__(self, engine: 'AudioEngine | None',
                 t_func, get_notes_count, get_cfg):
        super().__init__()
        self._engine = engine
        self._t = t_func
        self._get_notes_count = get_notes_count
        self._get_cfg = get_cfg
        self._device_label = self._resolve_device_label()

        from PyQt6.QtWidgets import QGridLayout
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 6, 8, 6)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(2)
        self._rows: dict[str, QLabel] = {}
        # Order matters — top-to-bottom layout of the panel.
        self._row_keys = [
            'data_device', 'data_rate_requested', 'data_rate_negotiated',
            'data_samplerate', 'data_block_hop',
            'data_blocksize', 'data_hopsize',
            'data_halfcent_floor',
            'data_freqrange', 'data_filter_mode', 'data_filter_params',
            'data_a4', 'data_rms', 'data_aperiodicity', 'data_freq',
            'data_midi', 'data_notes_count',
            'data_hotplug_poller',
        ]
        for r, key in enumerate(self._row_keys):
            k_lbl = QLabel(self._t(key) + ':')
            k_lbl.setStyleSheet('color:#8a8aa0;font-family:Monospace;font-size:11px;')
            v_lbl = QLabel('—')
            v_lbl.setStyleSheet('color:#d8d8ee;font-family:Monospace;font-size:11px;')
            v_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(k_lbl, r, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(v_lbl, r, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self._rows[key] = v_lbl
        self._grid.setColumnStretch(1, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self.REFRESH_MS)
        self._refresh_static()
        self._refresh()

    def retranslate(self, t_func) -> None:
        self._t = t_func
        # Re-label each row's key column. The grid stores key labels at
        # column 0 in the same order as self._row_keys.
        for r, key in enumerate(self._row_keys):
            item = self._grid.itemAtPosition(r, 0)
            if item is not None and item.widget() is not None:
                item.widget().setText(self._t(key) + ':')
        self._refresh_static()
        self._refresh()

    def _resolve_device_label(self) -> str:
        """Mirror what the engine reports if it's running. Falls back to
        the PortAudio default *only* through the safe probe — never the
        bare query_devices(kind='input') path that crashes when no
        input device exists (the v0.5.3 silent-crash bug)."""
        if not AUDIO_OK:
            return '—'
        eng = self._engine
        if eng is not None and getattr(eng, 'active_device', None) is not None:
            d = eng.active_device
            return f'{d.name}  [{d.host_api}]' if d.host_api else d.name
        return '—'

    def _refresh_static(self) -> None:
        """Update fields that only change on instrument/config changes."""
        self._rows['data_device'].setText(self._resolve_device_label())
        self._rows['data_freqrange'].setText(
            f'{MIN_FREQ:.1f} – {MAX_FREQ:.1f} Hz')

    def _refresh(self) -> None:
        if not AUDIO_OK or self._engine is None:
            for key in ('data_a4', 'data_rms', 'data_aperiodicity',
                        'data_freq', 'data_midi', 'data_filter_mode',
                        'data_filter_params'):
                self._rows[key].setText('—')
            for key in ('data_samplerate', 'data_blocksize', 'data_hopsize',
                        'data_rate_requested', 'data_rate_negotiated',
                        'data_block_hop', 'data_halfcent_floor'):
                self._rows[key].setText('—')
            self._rows['data_notes_count'].setText(
                str(int(self._get_notes_count())))
            self._rows['data_hotplug_poller'].setText('—')
            return
        # Pull a single atomic snapshot of the engine's diagnostic
        # scalars; avoids reading half-updated last_* values mid-callback.
        diag = self._engine.get_diagnostics()
        mode = self._engine.filter_mode
        params = _FILTER_PRESETS.get(mode, {})
        self._rows['data_device'].setText(self._resolve_device_label())
        sr = max(1, diag.samplerate)
        hop = max(1, diag.hop_size)
        hop_ms = 1000.0 * hop / sr
        self._rows['data_samplerate'].setText(f'{sr} Hz')
        self._rows['data_blocksize'].setText(f'{diag.block_size} samples')
        self._rows['data_hopsize'].setText(
            f'{hop} samples ({hop_ms:.1f} ms)')
        # v0.5.6 audio negotiation + precision rows.
        cfg = self._get_cfg() if self._get_cfg else None
        pref = (str(getattr(cfg, 'audio_samplerate_pref', 'auto'))
                if cfg else 'auto')
        if pref == 'auto':
            self._rows['data_rate_requested'].setText('Auto')
        else:
            self._rows['data_rate_requested'].setText(f'{pref} Hz')
        self._rows['data_rate_negotiated'].setText(f'{sr} Hz')
        block_ms = 1000.0 * diag.block_size / sr
        self._rows['data_block_hop'].setText(
            f'{diag.block_size} / {hop} ({block_ms:.0f} ms / {hop_ms:.0f} ms)')
        floor_a4 = cent_precision_floor(440.0, sr)
        self._rows['data_halfcent_floor'].setText(f'{floor_a4:.2f} ¢')
        self._rows['data_filter_mode'].setText(mode)
        if params:
            self._rows['data_filter_params'].setText(
                'win={window}  conf={confirm}  yin_thr={yin_thr}  '
                'rms_floor={rms_floor:g}  edge={edge_hops}'.format(**params))
        else:
            self._rows['data_filter_params'].setText('—')
        self._rows['data_a4'].setText(f'{self._engine.a4:.2f} Hz')
        self._rows['data_rms'].setText(f'{diag.rms_db:+.1f} dBFS')
        self._rows['data_aperiodicity'].setText(f'{diag.aperiodicity:.3f}')
        if diag.freq > 0:
            self._rows['data_freq'].setText(f'{diag.freq:.2f} Hz')
        else:
            self._rows['data_freq'].setText('—')
        m = diag.locked_midi
        if m is not None:
            self._rows['data_midi'].setText(
                f'{m}  ({midi_note_name(int(m))})')
        else:
            self._rows['data_midi'].setText('—')
        self._rows['data_notes_count'].setText(
            str(int(self._get_notes_count())))
        # Hot-plug poller status. The poller is a 1 Hz QTimer in
        # MainWindow that calls engine.refresh_devices(); the engine
        # stamps last_devices_refresh_at on every tick. Surfacing the
        # last-refresh time here lets the user confirm at a glance
        # that hot-plug detection is alive when a new interface fails
        # to recover.
        last = getattr(self._engine, 'last_devices_refresh_at', None)
        if last is None:
            self._rows['data_hotplug_poller'].setText(
                self._t('data_hotplug_value_never', interval=1))
        else:
            try:
                hhmmss = last.strftime('%H:%M:%S')
            except Exception:
                hhmmss = '—'
            self._rows['data_hotplug_poller'].setText(
                self._t('data_hotplug_value', interval=1, when=hhmmss))


# =============================================================================
# Delegate: grafischer Intonationsbalken in der Tabelle
# =============================================================================
class CentBarDelegate(QStyledItemDelegate):
    """Zeichnet einen zentrierten, farbcodierten Balken für Cent-Abweichungen.
    Cell payload is a dict {'cents': float, 'freq': float}; legacy bare
    floats are still accepted so older callers keep working."""

    MAX_CENT = 50.0   # ±50 ct = volle Balkenhälfte

    def __init__(self, parent=None, sample_rate_getter=None):
        super().__init__(parent)
        # v0.5.6: live sample rate for adaptive cent precision in the
        # printed value beside each bar. The bar geometry itself uses
        # the raw float so the visual position stays smooth.
        self._sr_get = sample_rate_getter or (
            lambda: _DEFAULT_SAMPLE_RATE_EXT)

    def paint(self, painter, option, index):
        raw = index.data(Qt.ItemDataRole.UserRole)
        freq = 0.0
        if isinstance(raw, dict):
            try:
                cents = float(raw.get('cents'))
                freq = float(raw.get('freq') or 0.0)
            except (TypeError, ValueError):
                super().paint(painter, option, index)
                return
        else:
            try:
                cents = float(raw)
            except (TypeError, ValueError):
                super().paint(painter, option, index)
                return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = option.rect
        # Hintergrund (Auswahl berücksichtigen)
        from PyQt6.QtWidgets import QStyle
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(r, QColor('#2d4a7a'))
        else:
            painter.fillRect(r, QColor(0, 0, 0, 0))

        # Farbe nach Betrag
        abw = abs(cents)
        if abw < 10:
            bar_col = QColor('#3a9e5f')   # grün
        elif abw < 20:
            bar_col = QColor('#c8a020')   # gelb
        else:
            bar_col = QColor('#c03030')   # rot

        # Dimensionen
        pad_x, pad_y = 8, 5
        w = r.width() - 2 * pad_x
        h = r.height() - 2 * pad_y
        cx = r.left() + pad_x + w // 2   # Mittellinie x

        # Hintergrundleiste
        bg_h = max(4, h // 3)
        bg_y = r.top() + pad_y + (h - bg_h) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(45, 45, 60))
        painter.drawRoundedRect(r.left() + pad_x, bg_y, w, bg_h, 3, 3)

        # Grüne Mitte-Zone (±5 ct)
        zone_w = max(2, int(w / 2 * 5.0 / self.MAX_CENT))
        painter.setBrush(QColor(30, 80, 40, 160))
        painter.drawRect(cx - zone_w, bg_y, zone_w * 2, bg_h)

        # Füllbalken
        norm = max(-1.0, min(1.0, cents / self.MAX_CENT))
        fill_w = max(2, int(abs(norm) * w / 2))
        bar_h = bg_h + 2
        bar_y = bg_y - 1
        painter.setBrush(bar_col)
        if cents >= 0:
            painter.drawRoundedRect(cx, bar_y, fill_w, bar_h, 2, 2)
        else:
            painter.drawRoundedRect(cx - fill_w, bar_y, fill_w, bar_h, 2, 2)

        # Mittellinie
        painter.setPen(QPen(QColor(180, 180, 200), 1))
        painter.drawLine(cx, bg_y - 2, cx, bg_y + bg_h + 2)

        # Cent-Wert als Text rechts — adaptive precision (v0.5.6)
        sr = self._sr_get() or _DEFAULT_SAMPLE_RATE_EXT
        txt = f"{format_cents(cents, freq, sr)} ct"
        painter.setPen(bar_col)
        painter.setFont(QFont('Monospace', 9))
        txt_rect = option.rect.adjusted(0, 0, -4, 0)
        painter.drawText(txt_rect,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         txt)

        painter.restore()

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        return sh.__class__(max(sh.width(), 160), max(sh.height(), 28))


# =============================================================================
# Delegate: matrix-mode cell with mean number + bar + std whiskers + live arrow
# =============================================================================
class MatrixCellDelegate(QStyledItemDelegate):
    """Paints each piano-roll cell with the same data the single-column
    table exposes per row, just stacked vertically:

    * Top strip: fingered note name on the left, sounding note name on
      the right (always shown for in-range cells so the user can read
      the cell's identity without consulting the row/column headers).
    * Mid strip: mean cents (color-coded by magnitude) and ±std as a
      smaller adjacent value.
    * Lower strip: horizontal scale centered on 0 ct with the filled
      bar from center to mean, plus ±1σ whiskers below.
    * Bottom-right corner: small N counter.

    Active cell (currently-played note within the last 1.5s) gets the
    blue background tint. Out-of-range cells render only the dimmed
    note label (so the player still knows what note that row+column
    represents) on a darker background.

    Data layout per cell (set on the QTableWidgetItem):
        ItemDataRole.UserRole → dict {
            'mean':          float | None,
            'std':           float | None,
            'n':             int,
            'in_range':      bool,
            'active':        bool,
            'fingered_name': str,
            'sounding_name': str,
        }
    """

    MAX_CENT = 50.0   # ±50 ct = bar saturated

    def __init__(self, parent=None, sample_rate_getter=None):
        super().__init__(parent)
        # v0.5.6: live sample rate for adaptive cent precision.
        self._sr_get = sample_rate_getter or (
            lambda: _DEFAULT_SAMPLE_RATE_EXT)

    def paint(self, painter, option, index):
        data = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            super().paint(painter, option, index)
            return

        r = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        in_range = data.get('in_range', True)
        active   = data.get('active', False)
        mean     = data.get('mean')
        std      = data.get('std') or 0.0
        n        = int(data.get('n') or 0)
        fingered = data.get('fingered_name', '')
        sounding = data.get('sounding_name', '')
        freq     = float(data.get('freq') or 0.0)
        sr       = self._sr_get() or _DEFAULT_SAMPLE_RATE_EXT

        if not in_range:
            # Out-of-range cells render with NO border and NO text — they
            # blend into the table background so the user only sees the
            # cells that correspond to physically playable notes. The
            # surrounding grid structure stays so column alignment is
            # preserved.
            painter.fillRect(r, QColor('#12121a'))
            painter.restore()
            return

        # In-range background — active blue tint or default panel color.
        if active:
            painter.fillRect(r, QColor('#2c5a8a'))
        else:
            painter.fillRect(r, QColor('#1a1a24'))

        # Subtle cell border — only on in-range cells, so the playable
        # area visually pops as a grid against the unbordered out-of-
        # range background.
        painter.setPen(QPen(QColor(55, 55, 75), 1))
        painter.drawRect(r.adjusted(0, 0, -1, -1))

        pad_x = 4
        # ----- Top strip: note names ----------------------------------
        top_h = 13
        top_y = r.top() + 1
        painter.setFont(QFont('Monospace', 7))
        painter.setPen(QColor(150, 150, 175))
        if fingered:
            painter.drawText(
                QRectF(r.left() + pad_x, top_y, r.width() / 2 - pad_x, top_h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                fingered)
        if sounding and sounding != fingered:
            painter.setPen(QColor(120, 130, 160))
            painter.drawText(
                QRectF(r.left() + r.width() / 2, top_y,
                       r.width() / 2 - pad_x, top_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                sounding)

        # v0.5.7.3: belt-and-suspenders against NaN sneaking through
        # from NoteStats (despite the source-side snapshot). NaN in
        # `mean` makes the bar width / fill width NaN and Qt's
        # drawRoundedRect with a NaN width is undefined. Treat NaN/inf
        # the same as "no measurement yet" — show the seeded dot.
        if mean is None or not math.isfinite(float(mean)):
            # In-range but no measurement yet — show a centered dot to
            # acknowledge the seeded slot.
            painter.setPen(QColor(80, 80, 100))
            painter.setFont(QFont('Monospace', 10))
            painter.drawText(
                QRectF(r.left(), r.top() + top_h, r.width(), r.height() - top_h),
                Qt.AlignmentFlag.AlignCenter, '·')
            painter.restore()
            return

        # ----- Mid strip: mean (color) + ±std ------------------------
        col = (QColor('#3a9e5f') if abs(mean) <= 5 else
               QColor('#c8a020') if abs(mean) <= 12 else QColor('#c03030'))
        mid_h = 18
        mid_y = r.top() + top_h
        painter.setPen(col)
        painter.setFont(QFont('Monospace', 11, QFont.Weight.Bold))
        painter.drawText(
            QRectF(r.left() + pad_x, mid_y,
                   r.width() * 0.62 - pad_x, mid_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            format_cents(mean, freq, sr))
        if n > 1 and std > 0:
            painter.setPen(QColor(170, 170, 200))
            painter.setFont(QFont('Monospace', 8))
            # Strip sign from format_cents and prepend "±".
            std_txt = format_cents(std, freq, sr).lstrip('+-')
            painter.drawText(
                QRectF(r.left() + r.width() * 0.55, mid_y,
                       r.width() * 0.45 - pad_x, mid_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"±{std_txt}")

        # ----- Bar with whiskers --------------------------------------
        scale_w = r.width() - 2 * pad_x
        scale_h = 4
        cx = r.left() + r.width() / 2
        scale_y = r.top() + top_h + mid_h + 3
        # Background trough.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(45, 45, 60))
        painter.drawRoundedRect(QRectF(r.left() + pad_x, scale_y,
                                        scale_w, scale_h), 2, 2)
        # ±5 ct in-tune band.
        zone_w = max(2.0, scale_w / 2 * 5.0 / self.MAX_CENT)
        painter.setBrush(QColor(40, 110, 60, 160))
        painter.drawRect(QRectF(cx - zone_w, scale_y, zone_w * 2, scale_h))
        # Filled bar.
        norm = max(-1.0, min(1.0, mean / self.MAX_CENT))
        fill_w = abs(norm) * scale_w / 2
        painter.setBrush(col)
        if mean >= 0:
            painter.drawRoundedRect(QRectF(cx, scale_y - 1,
                                            fill_w, scale_h + 2), 2, 2)
        else:
            painter.drawRoundedRect(QRectF(cx - fill_w, scale_y - 1,
                                            fill_w, scale_h + 2), 2, 2)
        painter.setPen(QPen(QColor(200, 200, 220), 1))
        painter.drawLine(QPointF(cx, scale_y - 2),
                          QPointF(cx, scale_y + scale_h + 2))

        # Whiskers.
        if n > 1 and std > 0:
            std_lo = max(-1.0, min(1.0, (mean - std) / self.MAX_CENT))
            std_hi = max(-1.0, min(1.0, (mean + std) / self.MAX_CENT))
            x_lo = cx + std_lo * scale_w / 2
            x_hi = cx + std_hi * scale_w / 2
            w_y = scale_y + scale_h + 4
            painter.setPen(QPen(QColor(220, 220, 235, 200), 1.4))
            painter.drawLine(QPointF(x_lo, w_y), QPointF(x_hi, w_y))
            painter.drawLine(QPointF(x_lo, w_y - 2),
                              QPointF(x_lo, w_y + 2))
            painter.drawLine(QPointF(x_hi, w_y - 2),
                              QPointF(x_hi, w_y + 2))

        # N counter, bottom-right.
        painter.setPen(QColor(140, 140, 165))
        painter.setFont(QFont('Monospace', 7))
        painter.drawText(
            QRectF(r.right() - 28, r.bottom() - 12, 26, 10),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"n={n}")

        painter.restore()

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        # Need vertical room: top (13) + mid (18) + bar (4) + whiskers (8)
        # + N counter (8) = ~55px minimum.
        return sh.__class__(max(sh.width(), 110), max(sh.height(), 60))


# =============================================================================
# Audio-Eingang Chip + Banner + Picker (v0.5.4)
# =============================================================================
# Status-dot colour palette — matches the existing dark theme.
_AUDIO_DOT_COLORS = {
    AudioEngineState.RUNNING: '#2ecc71',
    AudioEngineState.OPENING: '#b7770d',
    AudioEngineState.ENUMERATING: '#b7770d',
    AudioEngineState.FAILED:  '#c0392b',
    AudioEngineState.STOPPED: '#666',
    AudioEngineState.INIT:    '#666',
}


class _StatusDot(QWidget):
    """8 px solid circle. Recoloured by the chip whenever engine state
    changes — gives the user a glanceable health indicator without
    needing to read text."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor('#666')
        self.setFixedSize(10, 10)

    def set_color(self, hex_color: str) -> None:
        c = QColor(hex_color)
        if c != self._color:
            self._color = c
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(0, 0, self.width(), self.height())
        p.end()


class AudioChip(QPushButton):
    """Toolbar chip that shows current audio-input device + state.

    Clicking opens the picker. State updates are driven by the engine's
    ``state_changed`` signal — the chip never queries PortAudio
    directly, which keeps the GUI thread off the cold-init path
    documented in Legolas's perf memo."""

    def __init__(self, t_func):
        super().__init__()
        self._t = t_func
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet("""
            QPushButton{background:#34495e;color:#eee;border:none;
                         border-radius:5px;padding:4px 10px;font-size:12px;
                         text-align:left;}
            QPushButton:hover{background:#3d566e;}
            QPushButton:pressed{background:#2c3e50;}
        """)
        # Layout: dot · label · device name · chevron.
        from PyQt6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)
        self._dot = _StatusDot(self)
        lay.addWidget(self._dot)
        self._label = QLabel(self._t('audio_chip_label'))
        self._label.setStyleSheet('color:#bdc3c7;font-size:10px;font-weight:bold;')
        lay.addWidget(self._label)
        self._name = QLabel(self._t('audio_chip_none'))
        self._name.setStyleSheet('color:#eee;font-size:12px;')
        lay.addWidget(self._name, 1)
        self._chevron = QLabel('▾')
        self._chevron.setStyleSheet('color:#888;font-size:12px;')
        lay.addWidget(self._chevron)

    def retranslate(self, t_func) -> None:
        self._t = t_func
        self._label.setText(self._t('audio_chip_label'))
        self.setToolTip(self._t('audio_chip_tip'))

    def update_from_state(self, state: AudioEngineState,
                          device_name: str, host_api: str,
                          samplerate: int) -> None:
        color = _AUDIO_DOT_COLORS.get(state, '#666')
        self._dot.set_color(color)
        if state == AudioEngineState.RUNNING and device_name:
            short = device_name if len(device_name) <= 24 else device_name[:23] + '…'
            suffix = ''
            # Only surface non-44.1k rates in the chip — keeps the chip
            # clean when nothing unusual is happening.
            if samplerate and samplerate != 44100:
                suffix = f' · {samplerate / 1000:g} kHz'
            self._name.setText(f'{short}{suffix}')
            tip = device_name
            if host_api:
                tip += f'  [{host_api}]'
            if samplerate:
                tip += f'  · {samplerate} Hz'
            self.setToolTip(tip)
        elif state in (AudioEngineState.OPENING,
                        AudioEngineState.ENUMERATING):
            self._name.setText(self._t('audio_chip_opening'))
            self.setToolTip(self._t('audio_chip_opening'))
        else:
            self._name.setText(self._t('audio_chip_none'))
            self.setToolTip(self._t('audio_chip_tip'))


class AudioRecoveryBanner(QWidget):
    """Inline banner that appears when the engine is in FAILED state.

    Sits above the tuner; carries two buttons (Retry, pick a different
    device). The copy is selected by ``AudioEngineError`` per Frodo's
    UX memo — never displays raw PortAudio error codes to the user."""

    def __init__(self, t_func, on_retry, on_pick):
        super().__init__()
        self._t = t_func
        self._on_retry = on_retry
        self._on_pick = on_pick
        self.setStyleSheet("""
            QWidget{background:#1e1e2e;border:1px solid #444;border-left:4px solid #c0392b;border-radius:5px;}
            QLabel{color:#eee;font-size:12px;}
            QPushButton{background:#34495e;color:#eee;border:none;border-radius:4px;
                         padding:5px 12px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
        """)
        from PyQt6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        self._msg = QLabel('')
        self._msg.setWordWrap(True)
        lay.addWidget(self._msg, 1)
        self._btn_retry = QPushButton(self._t('audio_banner_retry'))
        self._btn_retry.clicked.connect(lambda: self._on_retry())
        lay.addWidget(self._btn_retry)
        self._btn_pick = QPushButton(self._t('audio_banner_pick'))
        self._btn_pick.clicked.connect(lambda: self._on_pick())
        lay.addWidget(self._btn_pick)
        self.hide()

    def retranslate(self, t_func) -> None:
        self._t = t_func
        self._btn_retry.setText(self._t('audio_banner_retry'))
        self._btn_pick.setText(self._t('audio_banner_pick'))

    def show_for(self, err: AudioEngineError, device_name: str,
                 raw_msg: str) -> None:
        if err == AudioEngineError.NO_DEVICE:
            self._msg.setText('⚠  ' + self._t('audio_banner_no_device'))
        elif err == AudioEngineError.DEVICE_DISCONNECTED:
            self._msg.setText('⚠  ' + self._t('audio_banner_disconnect',
                                              name=device_name or '?'))
        elif err == AudioEngineError.DEVICE_BUSY:
            self._msg.setText('⚠  ' + self._t('audio_banner_busy',
                                              name=device_name or '?'))
        elif err == AudioEngineError.UNSUPPORTED_RATE:
            self._msg.setText('⚠  '
                              + self._t('audio_banner_unsupported_rate'))
        else:
            # HOSTAPI_FAILURE / UNKNOWN — surface a generic line. The
            # raw PortAudio text goes to diagnostics, not here.
            self._msg.setText('⚠  '
                              + self._t('audio_banner_unknown', msg=raw_msg or '—'))
        self.show()


def _promote_vendor_prefix(name: str) -> str:
    """If ``name`` contains a known vendor brand mid-string (typical of
    Windows naming like "Headset (FIIO DSP Audio)" or "Microphone
    (2- Scarlett Solo)"), return a ``"{VENDOR} · {rest}"`` form with
    the matched vendor lifted to the front and the parenthesised vendor
    fragment stripped from the body.

    Returns the original ``name`` unchanged if no vendor regex matches
    or if the match already sits at position 0 (no promotion needed)."""
    import re
    if not name:
        return name
    m = re.search(VENDOR_REGEX, name, re.IGNORECASE)
    if m is None:
        return name
    vendor = m.group(0).upper()
    # Step 1: drop any parenthesised fragment whose interior matches the
    # vendor regex. Replace each such paren run with a single space (the
    # v0.5.7.2 fix: previously we used r'\s*\([^)]*\)\s*' which ate the
    # adjacent space and collapsed "Line In (FIIO) - ASUS" to
    # "Line In- ASUS"). Non-vendor parens are preserved verbatim.
    paren_re = re.compile(r'\([^)]*\)')
    cleaned_parts: list[str] = []
    pos = 0
    for pm in paren_re.finditer(name):
        cleaned_parts.append(name[pos:pm.start()])
        inner = pm.group(0)
        if re.search(VENDOR_REGEX, inner, re.IGNORECASE) is None:
            cleaned_parts.append(inner)
        else:
            cleaned_parts.append(' ')
        pos = pm.end()
    cleaned_parts.append(name[pos:])
    body = ''.join(cleaned_parts)
    # Step 2 (v0.5.7.4): strip bare occurrences of the matched vendor
    # token still left in the body. Without this, "FIIO Q3" produced
    # "FIIO · FIIO Q3" because step 1 only touches paren-wrapped vendor
    # text. We only strip the SAME vendor that was hoisted to the prefix
    # — other vendor-list tokens in the body are part of the device
    # name proper (e.g. "Universal Audio Apollo Twin X" should keep
    # "Apollo" in the body even though Apollo is a known brand).
    body = re.sub(
        rf'\b{re.escape(m.group(0))}\b', ' ', body, count=1, flags=re.IGNORECASE
    )
    # Step 3: clean up whitespace and orphan separators left by removal.
    body = re.sub(r'\s*\(\s*\)\s*', ' ', body)  # empty parens
    body = re.sub(r'\s+', ' ', body).strip(' -·:|')
    body = re.sub(r'\s*-\s*', ' - ', body).strip(' -')
    if not body:
        # Input was effectively just the vendor token (possibly wrapped
        # in parens) — return it bare rather than echoing back "VENDOR · ".
        return vendor
    return f'{vendor} · {body}'


class AudioPickerDialog(QDialog):
    """Modal device picker. Two-line rows, dedup-by-name with an
    expandable host-API sublist. Vendor regex ranks external interfaces
    to the top. The dialog never opens a stream itself — accepting the
    selection just calls ``engine.open_device(spec)`` and lets the
    state machine do the rest."""

    # v0.5.7.2: sample-rate re-open runs on a worker thread (open_device
    # internally walks candidates with t.join(timeout=0.8) per attempt,
    # which would freeze the UI for multiple seconds on a slow device).
    # The worker emits this signal carrying (prev_pref, new_pref) and a
    # QueuedConnection on the receiving slot marshals back to the GUI
    # thread for the UI updates.
    _sr_reopen_done = pyqtSignal(str, str)

    def __init__(self, parent, t_func, engine: AudioEngine,
                 cfg: sax_config.AppConfig, current: 'DeviceInfo | None'):
        super().__init__(parent)
        self._t = t_func
        self._engine = engine
        self._cfg = cfg
        self._current = current
        self._chosen: 'DeviceInfo | None' = None
        # v0.5.7.2: debounce flag — ignore further combo changes while
        # an open is in flight. The combo is also disabled, but the
        # flag guards against programmatic edits / queued signals that
        # could slip in around the worker boundary.
        self._sr_switch_in_flight: bool = False
        self._sr_prev_pref: str = 'auto'
        # v0.5.7.3: close-during-worker UAF guard. If the user closes the
        # dialog while an sr-reopen worker is still running, the worker's
        # finally block used to emit `_sr_reopen_done` on a QObject that
        # Qt had already destroyed (segfault / undefined behaviour). The
        # threading.Event lets the worker bail out of the emit, and we
        # also disconnect the signal in closeEvent as a second line of
        # defence.
        self._closing_event: threading.Event = threading.Event()
        self._sr_reopen_done.connect(
            self._on_sr_reopen_done, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle(self._t('audio_picker_title'))
        self.setModal(True)
        self.setMinimumWidth(560)
        # v0.5.7.1: dialog used to open at Qt's minimum-content height,
        # which fit ~3 rows and forced the user to scroll past the most
        # interesting devices (vendor interfaces below the system mics).
        # Open at a size where the full typical list fits unscrolled.
        self.setMinimumSize(480, 480)
        self.setStyleSheet("""
            QDialog{background:#1e1e2e;color:#eee;}
            QLabel{color:#eee;font-size:12px;}
            QListWidget{background:#15151f;border:1px solid #444;
                         color:#eee;font-size:12px;}
            QListWidget::item{padding:6px;}
            QListWidget::item:selected{background:#34495e;}
            QPushButton{background:#34495e;color:#eee;border:none;
                         border-radius:5px;padding:6px 14px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
            QCheckBox{color:#bbb;font-size:11px;}
        """)
        self._build()
        self._refill()
        # v0.5.7.1: explicit default size so the picker opens at a
        # comfortable height for the typical Windows device count.
        self.resize(560, 600)

    def _build(self) -> None:
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        # v0.5.6: sample-rate preference row. Sits ABOVE the device list
        # so the user reads the policy first and the chosen device second.
        sr_row = QHBoxLayout()
        sr_lbl = QLabel(self._t('samplerate_label') + ':')
        sr_lbl.setStyleSheet('color:#bbb;font-size:12px;')
        sr_row.addWidget(sr_lbl)
        self._sr_combo = QComboBox()
        self._sr_combo.addItem(self._t('samplerate_auto'), 'auto')
        for hz in (192000, 96000, 88200, 48000, 44100):
            # Number-with-thin-space formatting reads as "192 000 Hz" in
            # both DE and EN locales without dragging in locale plumbing.
            label = f'{hz:,}'.replace(',', ' ') + ' Hz'
            self._sr_combo.addItem(label, str(hz))
        cur_pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                'auto') or 'auto')
        idx = self._sr_combo.findData(cur_pref)
        if idx < 0:
            idx = 0
        self._sr_combo.setCurrentIndex(idx)
        # v0.5.7.2: snapshot for revert-on-failure path.
        self._sr_prev_pref = cur_pref
        self._sr_combo.currentIndexChanged.connect(self._on_sr_changed)
        self._sr_combo.setStyleSheet(
            'QComboBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:4px;padding:3px 8px;font-size:12px;}'
            'QComboBox::drop-down{border:none;width:18px;background:#1e1e2e;}'
            'QComboBox QAbstractItemView{background:#1e1e2e;color:#ddd;'
            'border:1px solid #444;outline:0;'
            'selection-background-color:#34495e;selection-color:#fff;}'
            'QComboBox QAbstractItemView::item{background:#1e1e2e;color:#ddd;'
            'padding:4px 8px;border:none;}'
            'QComboBox QAbstractItemView::item:selected{background:#34495e;color:#fff;}')
        sr_row.addWidget(self._sr_combo, 1)
        root.addLayout(sr_row)

        self._sr_error_lbl = QLabel('')
        self._sr_error_lbl.setStyleSheet(
            'color:#e07070;font-size:11px;padding:0 2px;')
        self._sr_error_lbl.setVisible(False)
        self._sr_error_lbl.setWordWrap(True)
        root.addWidget(self._sr_error_lbl)

        top = QHBoxLayout()
        self._btn_rescan = QPushButton(self._t('audio_picker_rescan'))
        self._btn_rescan.clicked.connect(self._refill)
        top.addWidget(self._btn_rescan)
        top.addStretch()
        root.addLayout(top)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _i: self._accept())
        root.addWidget(self._list, 1)

        opts = QHBoxLayout()
        self._cb_show_all = QCheckBox(self._t('audio_picker_show_all'))
        self._cb_show_all.setChecked(bool(getattr(self._cfg,
                                                    'show_all_host_apis',
                                                    False)))
        self._cb_show_all.toggled.connect(self._on_toggle_show_all)
        opts.addWidget(self._cb_show_all)
        self._cb_ks = QCheckBox(self._t('audio_picker_prefer_ks'))
        self._cb_ks.setChecked(bool(getattr(self._cfg, 'prefer_wdmks', False)))
        self._cb_ks.toggled.connect(self._on_toggle_ks)
        opts.addWidget(self._cb_ks)
        opts.addStretch()
        root.addLayout(opts)

        btns = QHBoxLayout()
        self._btn_use = QPushButton(self._t('audio_picker_use'))
        self._btn_use.clicked.connect(self._accept)
        self._btn_cancel = QPushButton(self._t('audio_picker_cancel'))
        self._btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self._btn_use)
        btns.addWidget(self._btn_cancel)
        root.addLayout(btns)

    def _on_sr_changed(self, _idx: int) -> None:
        """User picked a new sample-rate policy. Persist immediately,
        then reopen the active device with the new pref on a worker
        thread.

        v0.5.7.2: open_device walks candidates and runs
        ``t.join(timeout=0.8)`` per attempt, which on Qt's main thread
        froze the dialog for multiple seconds. We now spawn a worker,
        disable the combo while it runs, and marshal the UI updates
        back through ``_sr_reopen_done`` (QueuedConnection) so all
        widget mutation happens on the GUI thread again. The engine's
        internal lock (v0.5.4) makes the off-thread call safe.
        """
        if self._sr_switch_in_flight:
            # Debounce: ignore further changes until the current worker
            # finishes. The combo is disabled below, so this only fires
            # for queued signals that slipped past the disable.
            return
        new_pref = str(self._sr_combo.currentData() or 'auto')
        if new_pref not in SAMPLERATE_PREF_VALUES:
            new_pref = 'auto'
        prev_pref = self._sr_prev_pref
        if new_pref == prev_pref:
            return
        self._cfg.audio_samplerate_pref = new_pref
        sax_config.save_config(self._cfg)
        self._sr_error_lbl.setVisible(False)
        dev = self._engine.get_active_device()
        if dev is None:
            self._sr_prev_pref = new_pref
            return
        sel = DeviceSelection(name=dev.name, host_api=dev.host_api,
                              samplerate=0)

        # Block the GUI from issuing more re-opens until this one lands.
        self._sr_switch_in_flight = True
        self._sr_combo.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # v0.5.7.3: capture the event by closure (not self.) so the
        # worker can check it even if the dialog is mid-teardown. The
        # event lives independently of the QObject's lifetime.
        closing_event = self._closing_event

        def _worker() -> None:
            try:
                self._engine.stop()
                self._engine.open_device(sel, samplerate_pref=new_pref)
            finally:
                # Even if the engine raises, hand control back to the
                # GUI thread so the combo gets re-enabled — UNLESS the
                # dialog is being closed, in which case the QObject
                # may already be destroyed and emitting on a dead
                # receiver is undefined behaviour.
                if not closing_event.is_set():
                    try:
                        self._sr_reopen_done.emit(prev_pref, new_pref)
                    except RuntimeError:
                        # "wrapped C/C++ object has been deleted" —
                        # raced past our event check. Swallow; there's
                        # no receiver to hand off to anyway.
                        pass

        t = threading.Thread(target=_worker, name='sr-reopen',
                             daemon=True)
        t.start()

    def _on_sr_reopen_done(self, prev_pref: str, new_pref: str) -> None:
        """Runs on the GUI thread (QueuedConnection). Re-enables the
        combo, surfaces the inline error on UNSUPPORTED_RATE, and
        reverts the selection if the new pref was refused."""
        try:
            QGuiApplication.restoreOverrideCursor()
        except Exception:
            pass
        failed_rate = (
            self._engine.state == AudioEngineState.FAILED
            and self._engine.last_error == AudioEngineError.UNSUPPORTED_RATE
        )
        if failed_rate:
            shown_rate = new_pref if new_pref != 'auto' else '–'
            self._sr_error_lbl.setText(
                self._t('samplerate_unsupported', rate=shown_rate))
            self._sr_error_lbl.setVisible(True)
            # Revert to previous pref in combo + cfg, then retry the
            # open with the previous pref so the user is left in the
            # same audio state they started with.
            self._cfg.audio_samplerate_pref = prev_pref
            sax_config.save_config(self._cfg)
            self._sr_combo.blockSignals(True)
            revert_idx = self._sr_combo.findData(prev_pref)
            if revert_idx < 0:
                revert_idx = 0
            self._sr_combo.setCurrentIndex(revert_idx)
            self._sr_combo.blockSignals(False)
            dev = self._engine.get_active_device()
            if dev is not None:
                sel = DeviceSelection(name=dev.name, host_api=dev.host_api,
                                      samplerate=0)
                # Retry on a worker too so we don't re-introduce the
                # main-thread block we just removed. Best-effort: fire
                # and forget — failure here just leaves the engine in
                # FAILED, which the banner handles separately.
                def _retry() -> None:
                    try:
                        self._engine.open_device(sel,
                                                 samplerate_pref=prev_pref)
                    except Exception:
                        pass
                threading.Thread(target=_retry, name='sr-revert',
                                 daemon=True).start()
            self._sr_prev_pref = prev_pref
        else:
            self._sr_prev_pref = new_pref
        self._sr_combo.setEnabled(True)
        self._sr_switch_in_flight = False

    def closeEvent(self, event) -> None:
        """v0.5.7.3: signal any in-flight sr-reopen worker that we're
        going away so it skips its emit (which would otherwise land on a
        deleted QObject). Also disconnect the signal as belt-and-
        suspenders — if a queued emission is already sitting in the
        event loop, the disconnect makes it a no-op."""
        self._closing_event.set()
        try:
            self._sr_reopen_done.disconnect(self._on_sr_reopen_done)
        except (TypeError, RuntimeError):
            # Already disconnected or signal/slot already torn down.
            pass
        super().closeEvent(event)

    def done(self, result: int) -> None:
        """``accept()`` / ``reject()`` route through here, so set the
        closing flag here too. closeEvent isn't always called on
        programmatic accept/reject paths."""
        self._closing_event.set()
        super().done(result)

    def _on_toggle_show_all(self, checked: bool) -> None:
        self._cfg.show_all_host_apis = bool(checked)
        sax_config.save_config(self._cfg)
        self._refill()

    def _on_toggle_ks(self, checked: bool) -> None:
        self._cfg.prefer_wdmks = bool(checked)
        sax_config.save_config(self._cfg)
        self._engine.set_prefer_wdmks(bool(checked))

    def _rank(self, d: DeviceInfo) -> int:
        """Higher = sorts earlier. Vendor regex dominates."""
        import re
        score = 0
        if re.search(VENDOR_REGEX, d.name, re.IGNORECASE):
            score += 100
        api = d.host_api.lower()
        if 'wasapi' in api:
            score += 20
        elif 'wdm-ks' in api:
            score += 10
        if d.default_samplerate >= 48000:
            score += 5
        low = d.name.lower()
        if any(x in low for x in ('webcam', 'hdmi', 'nvidia', 'amd')):
            score -= 50
        if any(x in low for x in ('microphone array', 'stereo mix')):
            score -= 20
        return score

    def _refill(self) -> None:
        from PyQt6.QtWidgets import QListWidgetItem
        self._list.clear()
        devices = self._engine.refresh_devices()
        if not devices:
            it = QListWidgetItem(self._t('audio_picker_no_devices'))
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(it)
            return
        show_all = self._cb_show_all.isChecked()
        # Group by canonical name unless show-all is on.
        groups: dict[str, list[DeviceInfo]] = {}
        order: list[str] = []
        for d in devices:
            key = d.name if not show_all else f'{d.name}\0{d.host_api}'
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(d)
        # Pick a representative for each group (WASAPI > WDM-KS > MME > rest).
        def api_pref(d: DeviceInfo) -> int:
            api = d.host_api.lower()
            if 'wasapi' in api:
                return 0
            if 'wdm-ks' in api:
                return 1
            if api == 'mme':
                return 2
            if 'directsound' in api:
                return 3
            return 9
        rows: list[tuple[DeviceInfo, list[DeviceInfo]]] = []
        for key in order:
            members = sorted(groups[key], key=api_pref)
            rows.append((members[0], members))
        # Sort rows by rank, current device pinned to the top.
        def sort_key(row):
            primary = row[0]
            is_current = (
                self._current is not None
                and primary.name == self._current.name
                and primary.host_api == self._current.host_api)
            return (0 if is_current else 1, -self._rank(primary),
                    primary.name.lower())
        rows.sort(key=sort_key)
        for primary, members in rows:
            self._add_row(primary, members)

    def _add_row(self, primary: DeviceInfo,
                 members: list[DeviceInfo]) -> None:
        from PyQt6.QtWidgets import QListWidgetItem
        is_current = (
            self._current is not None
            and primary.name == self._current.name
            and primary.host_api == self._current.host_api)
        meta = self._t('audio_picker_row_meta',
                       api=primary.host_api or '—',
                       ch=primary.max_input_channels,
                       sr=primary.default_samplerate)
        badge = ''
        if is_current:
            badge = f'  ◀ {self._t("audio_picker_current")}'
        elif len(members) > 1:
            badge = f'  ▸ {self._t("audio_picker_apis_more", n=len(members))}'
        # v0.5.7: if a vendor brand appears mid-string in the Windows
        # device name (e.g. "Headset (FIIO DSP Audio)"), promote the
        # brand to the front of the row label so saxophone players can
        # scan by brand instead of by Windows device-naming convention.
        # The full original name remains in the row's tooltip.
        display_name = _promote_vendor_prefix(primary.name)
        text = f'{display_name}{badge}\n    {meta}'
        it = QListWidgetItem(text)
        it.setData(Qt.ItemDataRole.UserRole, primary)
        it.setToolTip(primary.name)
        self._list.addItem(it)
        if is_current:
            self._list.setCurrentItem(it)

    def _accept(self) -> None:
        it = self._list.currentItem()
        if it is None:
            self.reject()
            return
        dev = it.data(Qt.ItemDataRole.UserRole)
        if not isinstance(dev, DeviceInfo):
            self.reject()
            return
        self._chosen = dev
        self.accept()

    def chosen(self) -> 'DeviceInfo | None':
        return self._chosen


# =============================================================================
# Per-instrument range editor (v0.5.5)
# =============================================================================
class RangeEditorDialog(QDialog):
    """Modal editor for the (lo, hi) fingered-MIDI range of one
    instrument. Persists to ~/.intonation_analyzer/instrument_ranges.json
    via sax_instruments.save_range_override / clear_range_override.

    The dialog owns no policy: the caller (MainWindow) refreshes its
    table after accept() so the new range takes effect immediately."""

    def __init__(self, parent, t, instrument_key: str,
                 display_name: str, current_lo: int, current_hi: int,
                 baked_lo: int, baked_hi: int, has_baked: bool,
                 display: str = 'griff', transp: int = 0):
        """``current_lo`` / ``current_hi`` / ``baked_lo`` / ``baked_hi``
        are ALWAYS fingered MIDI (canonical disk format). The dialog
        renders them as sounding MIDI when ``display == 'klingend'`` by
        adding ``transp`` semitones, and reverses the offset on save —
        the JSON on disk never changes format regardless of which user
        opens the editor.
        """
        super().__init__(parent)
        self._t = t
        self._key = instrument_key
        self._baked = (baked_lo, baked_hi)  # canonical fingered MIDI
        self._has_baked = has_baked
        self._display = 'klingend' if display == 'klingend' else 'griff'
        self._transp = int(transp) if self._display == 'klingend' else 0
        self.setWindowTitle(self._t('range_editor_title', name=display_name))
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setSpacing(8)
        self._lo_spin = QSpinBox()
        self._lo_spin.setRange(0, 127)
        self._lo_spin.setValue(int(current_lo) + self._transp)
        self._hi_spin = QSpinBox()
        self._hi_spin.setRange(0, 127)
        self._hi_spin.setValue(int(current_hi) + self._transp)
        spin_css = (
            'QSpinBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:3px 6px;font-size:13px;min-width:90px;}'
        )
        self._lo_spin.setStyleSheet(spin_css)
        self._hi_spin.setStyleSheet(spin_css)
        if self._display == 'klingend':
            lo_label = self._t('range_lo_label_sounding')
            hi_label = self._t('range_hi_label_sounding')
        else:
            lo_label = self._t('range_lo_label')
            hi_label = self._t('range_hi_label')
        form.addRow(lo_label, self._lo_spin)
        form.addRow(hi_label, self._hi_spin)
        layout.addLayout(form)

        # Mode hint so the user knows what the spinboxes mean and that
        # the file format is stable across users with different display
        # preferences.
        mode_note_key = ('range_mode_note_sound'
                        if self._display == 'klingend'
                        else 'range_mode_note_griff')
        self._mode_lbl = QLabel(self._t(mode_note_key))
        self._mode_lbl.setStyleSheet('color:#888;font-size:11px;'
                                     'font-style:italic;')
        self._mode_lbl.setWordWrap(True)
        layout.addWidget(self._mode_lbl)

        self._preview_lbl = QLabel('')
        self._preview_lbl.setStyleSheet('color:#bbb;font-size:12px;')
        self._preview_lbl.setWordWrap(True)
        layout.addWidget(self._preview_lbl)

        self._error_lbl = QLabel('')
        self._error_lbl.setStyleSheet('color:#e07070;font-size:12px;')
        self._error_lbl.setWordWrap(True)
        self._error_lbl.setVisible(False)
        layout.addWidget(self._error_lbl)

        # Buttons row.
        btn_row = QHBoxLayout()
        if has_baked:
            restore_label = self._t('range_restore_default')
        else:
            # Show baked numbers in the current display mode so they
            # match what the spinboxes display.
            disp_lo = baked_lo + self._transp
            disp_hi = baked_hi + self._transp
            restore_label = self._t('range_restore_fallback',
                                    lo=disp_lo, hi=disp_hi)
        self._btn_restore = QPushButton(restore_label)
        self._btn_restore.clicked.connect(self._on_restore)
        self._btn_cancel = QPushButton(self._t('range_cancel'))
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_save = QPushButton(self._t('range_save'))
        self._btn_save.clicked.connect(self._on_save)
        btn_css = (
            'QPushButton{background:#34495e;color:#eee;border:none;'
            'border-radius:5px;padding:6px 12px;font-size:12px;}'
            'QPushButton:hover{background:#3d566e;}'
            'QPushButton:disabled{background:#2a2a3a;color:#666;}'
        )
        self._btn_restore.setStyleSheet(btn_css)
        self._btn_cancel.setStyleSheet(btn_css)
        self._btn_save.setStyleSheet(btn_css)
        btn_row.addWidget(self._btn_restore)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_save)
        layout.addLayout(btn_row)

        # Live preview wiring.
        self._lo_spin.valueChanged.connect(self._update_preview)
        self._hi_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

        # Dark theme to match the rest of the app.
        self.setStyleSheet('QDialog{background:#12121a;color:#ddd;} '
                           'QLabel{color:#ccc;}')

    def _update_preview(self) -> None:
        lo = self._lo_spin.value()
        hi = self._hi_spin.value()
        invalid = lo > hi
        # Visual cue on the offending field.
        bad_css = (
            'QSpinBox{background:#3a1e1e;color:#fdd;border:1px solid #c0392b;'
            'border-radius:5px;padding:3px 6px;font-size:13px;min-width:90px;}'
        )
        ok_css = (
            'QSpinBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:3px 6px;font-size:13px;min-width:90px;}'
        )
        self._lo_spin.setStyleSheet(bad_css if invalid else ok_css)
        self._hi_spin.setStyleSheet(bad_css if invalid else ok_css)
        if invalid:
            self._error_lbl.setText(self._t('range_invalid'))
            self._error_lbl.setVisible(True)
            self._preview_lbl.setText('')
            self._btn_save.setEnabled(False)
            return
        self._error_lbl.setVisible(False)
        self._btn_save.setEnabled(True)
        semis = hi - lo
        octs = semis / 12.0
        self._preview_lbl.setText(self._t(
            'range_preview_fmt',
            lo_name=midi_note_name(lo), hi_name=midi_note_name(hi),
            semis=semis, octs=octs))

    def _on_restore(self) -> None:
        # Baked is canonical fingered; render in current display mode.
        lo, hi = self._baked
        self._lo_spin.setValue(lo + self._transp)
        self._hi_spin.setValue(hi + self._transp)

    def _on_save(self) -> None:
        # Spinbox values are in the current display mode. Convert back
        # to fingered (canonical) before persisting so the file format
        # is stable across users with different display preferences.
        disp_lo = self._lo_spin.value()
        disp_hi = self._hi_spin.value()
        if disp_lo > disp_hi:
            return
        lo = disp_lo - self._transp
        hi = disp_hi - self._transp
        # If the user dialed back to the baked default AND a baked entry
        # exists, clear the override instead of writing a redundant
        # entry. Keeps the overrides file lean and lets future baked
        # changes flow through.
        if self._has_baked and (lo, hi) == self._baked:
            sax_instruments.clear_range_override(self._key)
        else:
            sax_instruments.save_range_override(self._key, lo, hi)
        self.accept()


# =============================================================================
# Haupt-Fenster
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Pick the user's system language as the default. German-speaking
        # locales get the original DE strings; anyone else gets English.
        self.lang       = 'de' if QLocale.system().name().startswith('de') else 'en'
        # v0.5.7.1: default to Bb tenor — the most common community-band
        # sax. Existing users with a persisted ``last_instrument_key``
        # keep their own choice (restored a few lines below); this only
        # affects fresh installs and first-launches.
        self.instrument = 'bb_tenor'
        self.display    = 'griff'
        self.stats: dict[int, NoteStats] = {}
        self._lock = threading.Lock()
        self._recording = True   # Aufnahme läuft beim Start
        self._active_midi: int | None = None
        self._active_midi_at: datetime.datetime | None = None
        self._layout_mode: str = 'single'   # 'single' | 'matrix'
        # Cumulative count of distinct note emissions since launch (each
        # _on_note increments). Diagnostics readout consumes this; the
        # table-level note count comes from `len(self.stats)`.
        self._notes_count: int = 0

        # Load user config + previously-registered custom instruments before
        # building the UI so the catalog reflects them at first paint.
        self._cfg = sax_config.load_config()
        for c in sax_config.load_customs():
            register_custom(c.key, c.transp, c.name_de, c.name_en)
        _rebuild_transp_map()

        # v0.5.5: restore last-session preferences that influence the
        # initial UI build. Geometry, splitter sizes, and nickname text
        # are widget-bound and get applied after _build_ui returns.
        # First-launch (empty config) falls back to the locale-derived
        # default already set above.
        if getattr(self._cfg, 'last_lang', '') in ('de', 'en'):
            self.lang = self._cfg.last_lang
        if getattr(self._cfg, 'last_instrument_key', ''):
            # Trust the saved key only if the catalog still knows it; a
            # stale custom instrument that was deleted should fall back
            # rather than crash the combo population.
            if self._cfg.last_instrument_key in TRANSP_MAP:
                self.instrument = self._cfg.last_instrument_key
        if getattr(self._cfg, 'last_display_mode', '') in ('griff', 'klingend'):
            self.display = self._cfg.last_display_mode

        self._engine = AudioEngine()
        self._engine.set_filter_mode(
            getattr(self._cfg, 'filter_mode', FILTER_MODE_DEFAULT))
        self._engine.set_prefer_wdmks(
            bool(getattr(self._cfg, 'prefer_wdmks', False)))
        # Persistence comes from config (welcome dialog), with the env var
        # SAX_INTONATION_LOG_PATH as a power-user override that always wins.
        env_path = os.environ.get('SAX_INTONATION_LOG_PATH')
        log_path = env_path if env_path else self._cfg.effective_log_path()
        self._log = MeasurementLog(path=log_path or None)
        if AUDIO_OK:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=self._engine.a4)
            self._engine.signals.note_detected.connect(self._on_note)
            self._engine.signals.state_changed.connect(self._on_engine_state)
            self._engine.signals.devices_changed.connect(
                self._on_devices_changed)
            self._engine.signals.interface_appeared.connect(
                self._on_interface_appeared)

        self._build_ui()
        self._restore_session_state()
        self._seed_expected_notes()
        self._update_record_btn_style()

        # Start the engine AFTER the UI exists so a startup PortAudio
        # failure paints a banner instead of crashing __init__. The
        # engine's start() never raises — it sets state and emits
        # state_changed. This is the v0.5.4 headline fix.
        if AUDIO_OK:
            saved = self._device_selection_from_cfg()
            pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                'auto') or 'auto')
            # v0.5.7: the engine needs the saved (name, host_api) so its
            # hot-plug poller and retry_open path can re-resolve the
            # user's preferred device against a fresh device list
            # without round-tripping through MainWindow.
            self._engine.set_preferred_hint(saved)
            self._engine.start(preferred=saved, samplerate_pref=pref)

        # Hot-plug poller: 1 Hz per Legolas's measurements (cached
        # query_devices is ~0 ms; only re-init is expensive, and we
        # only re-init when the device set actually changed).
        if AUDIO_OK:
            self._device_poll = QTimer(self)
            self._device_poll.timeout.connect(self._poll_devices)
            self._device_poll.start(1000)

        if not AUDIO_OK:
            QMessageBox.information(self, self._t('audio_error_title'),
                                     self._t('audio_error'))

        # First-boot welcome dialog asks about persistence. Skipped on every
        # subsequent launch.
        if not self._cfg.welcome_shown:
            self._show_welcome_dialog()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_table)
        self._refresh_timer.start(300)

    # ── Übersetzungs-Helfer ──────────────────────────────────────────────────
    def _t(self, key, **kwargs):
        s = STRINGS[self.lang].get(key, STRINGS['de'].get(key, f'[{key}]'))
        return s.format(**kwargs) if kwargs else s

    # ── UI aufbauen ──────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 10, 14, 10)

        # ── Toolbar ───────────────────────────────────────────────────────────
        # Always two predictable rows: inputs on top (instrument selection,
        # display, A4, language, import), actions on bottom (autotune, record,
        # reset, exports). When the window is wide enough, both rows breathe
        # easily; when it's narrow, content stays grouped by intent instead
        # of wrapping at arbitrary positions.
        toolbar_container = QWidget()
        toolbar_v = QVBoxLayout(toolbar_container)
        toolbar_v.setContentsMargins(0, 0, 0, 0)
        toolbar_v.setSpacing(6)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar_actions = QHBoxLayout()
        toolbar_actions.setSpacing(8)
        toolbar_v.addLayout(toolbar)
        toolbar_v.addLayout(toolbar_actions)

        # Instrument: family combo + sub-instrument combo + Custom + nickname.
        self._grp_instr = QGroupBox(self._t('grp_instrument'))
        il = QHBoxLayout(self._grp_instr)
        il.setContentsMargins(8, 4, 8, 4)
        il.setSpacing(6)

        self._family_combo = QComboBox()
        for family_key, name_de, name_en in instrument_families():
            label = name_de if self.lang == 'de' else name_en
            self._family_combo.addItem(label, family_key)
        self._family_combo.setMinimumWidth(130)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        il.addWidget(self._family_combo)

        self._instr_combo = QComboBox()
        self._instr_combo.setMinimumWidth(180)
        self._instr_combo.currentIndexChanged.connect(self._on_instr_changed)
        il.addWidget(self._instr_combo)

        # v0.5.5: gear button opens the per-instrument range editor. Sits
        # immediately to the right of the instrument combo so the user
        # finds it without hunting through menus. Unicode glyph instead of
        # an SVG asset — the project ships no settings icon and we keep
        # the dependency surface minimal.
        self._btn_range = QToolButton()
        self._btn_range.setText('⚙')   # U+2699 GEAR
        self._btn_range.setToolTip(self._t('gear_tip'))
        self._btn_range.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_range.setStyleSheet("""
            QToolButton{background:#1e1e2e;color:#ddd;border:1px solid #444;
                         border-radius:5px;padding:2px 6px;font-size:15px;
                         min-height:28px;min-width:28px;}
            QToolButton:hover{background:#2a2a3a;border:1px solid #6699cc;}
            QToolButton:pressed{background:#16161e;}
        """)
        self._btn_range.clicked.connect(self._open_range_editor)
        il.addWidget(self._btn_range)

        self._btn_custom = QPushButton(self._t('custom_label'))
        self._btn_custom.setToolTip(self._t('custom_dlg_title'))
        self._btn_custom.setMaximumWidth(110)
        self._btn_custom.clicked.connect(self._on_add_custom)
        self._btn_custom.setStyleSheet("""
            QPushButton{background:#34495e;color:#eee;border:none;
                         border-radius:5px;padding:6px 10px;font-size:12px;}
            QPushButton:hover{background:#3d566e;}
            QPushButton:pressed{background:#2c3e50;}
        """)
        il.addWidget(self._btn_custom)

        self._nick_edit = QLineEdit()
        self._nick_edit.setPlaceholderText(self._t('nickname_tip'))
        # v0.5.7.1: 160px capped at ~12 characters, which truncated real
        # horn names like "Selmer Reference 54 Tenor #2" before the user
        # could see what they'd typed. Bump to 320 max / 280 min.
        self._nick_edit.setMinimumWidth(280)
        self._nick_edit.setMaximumWidth(320)
        self._nick_edit.editingFinished.connect(self._on_nickname_changed)
        self._nick_edit.setStyleSheet("""
            QLineEdit{background:#1e1e2e;border:1px solid #444;
                       border-radius:5px;color:#ddd;padding:4px 8px;font-size:12px;}
            QLineEdit:focus{border:1px solid #6699cc;}
        """)
        il.addWidget(self._nick_edit)

        # v0.5.7: the "Filter to instrument range" checkbox lives
        # alongside the min-N spinbox under the table (both gate what
        # the table shows). The "Show spectrum analyzer & diagnostics"
        # checkbox lives in a footer under the TunerWidget (next to
        # the widgets it controls). The checkboxes themselves are
        # constructed below in _build_ui after the splitter, with the
        # same handlers + persistence — the only thing that changed
        # is the parent layout.

        # Select the saxophone family + the default alto instrument.
        self._select_family_for_instrument(self.instrument)
        self._populate_instrument_combo(select_key=self.instrument)

        # Anzeige
        self._grp_disp = QGroupBox(self._t('grp_display'))
        dl = QHBoxLayout(self._grp_disp)
        dl.setContentsMargins(8, 4, 8, 4)
        self._disp_combo = QComboBox()
        self._disp_combo.addItems([self._t('disp_griff'), self._t('disp_klingend')])
        self._disp_combo.currentIndexChanged.connect(self._on_disp_changed)
        dl.addWidget(self._disp_combo)

        # Layout mode override: Auto / List / Grid. Auto picks based on
        # window width; the explicit choices let the user pin the layout
        # they want regardless of how wide the window is.
        self._layout_combo = QComboBox()
        self._layout_combo.addItem(self._t('layout_auto'), 'auto')
        self._layout_combo.addItem(self._t('layout_single'), 'single')
        self._layout_combo.addItem(self._t('layout_matrix'), 'matrix')
        pref = getattr(self._cfg, 'layout_mode_preference', 'auto')
        for i in range(self._layout_combo.count()):
            if self._layout_combo.itemData(i) == pref:
                self._layout_combo.setCurrentIndex(i)
                break
        self._layout_combo.currentIndexChanged.connect(
            self._on_layout_pref_changed)
        dl.addWidget(self._layout_combo)

        # Kammerton
        self._grp_a4 = QGroupBox(self._t('grp_a4'))
        al = QHBoxLayout(self._grp_a4)
        al.setContentsMargins(8, 4, 8, 4)
        self._a4_combo = QComboBox()
        for hz in range(430, 451):
            self._a4_combo.addItem(f'{hz} Hz', hz)
        self._a4_combo.setCurrentIndex(10)   # 440 Hz
        self._a4_combo.setMinimumWidth(100)
        self._a4_combo.currentIndexChanged.connect(self._on_a4_changed)
        al.addWidget(self._a4_combo)

        # Response: pitch-detection smoothing preset. Mirrors a guitar-
        # tuner "Fast/Normal/Slow" toggle. Routed straight into the
        # engine; persisted in cfg.filter_mode.
        self._grp_filter = QGroupBox(self._t('grp_filter'))
        fl = QHBoxLayout(self._grp_filter)
        fl.setContentsMargins(8, 4, 8, 4)
        self._filter_combo = QComboBox()
        self._filter_combo.addItem(self._t('filter_fast'),   'fast')
        self._filter_combo.addItem(self._t('filter_normal'), 'normal')
        self._filter_combo.addItem(self._t('filter_slow'),   'slow')
        cur_mode = getattr(self._cfg, 'filter_mode', FILTER_MODE_DEFAULT)
        for i in range(self._filter_combo.count()):
            if self._filter_combo.itemData(i) == cur_mode:
                self._filter_combo.setCurrentIndex(i)
                break
        self._filter_combo.setToolTip(self._t('filter_tip'))
        self._filter_combo.currentIndexChanged.connect(
            self._on_filter_mode_changed)
        self._filter_combo.setMinimumWidth(100)
        fl.addWidget(self._filter_combo)

        # Sprache
        self._grp_lang = QGroupBox(self._t('grp_language'))
        ll2 = QHBoxLayout(self._grp_lang)
        ll2.setContentsMargins(8, 4, 8, 4)
        self._lang_combo = QComboBox()
        self._lang_combo.addItem('Deutsch', 'de')
        self._lang_combo.addItem('English', 'en')
        self._lang_combo.setMinimumWidth(100)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        ll2.addWidget(self._lang_combo)

        # Buttons
        self._btn_autotune = self._make_btn(self._t('btn_autotune'), '#1a6b3a', self._on_autotune)
        self._btn_record   = self._make_btn(self._t('btn_stop'),     '#b7770d', self._on_record_toggle)
        self._btn_reset    = self._make_btn(self._t('btn_reset'),    '#c0392b', self._on_reset)
        self._btn_txt      = self._make_btn(self._t('btn_txt'),      '#2980b9', self._export_txt)
        self._btn_pdf      = self._make_btn(self._t('btn_pdf'),      '#8e44ad', self._export_pdf)
        self._btn_csv      = self._make_btn(self._t('btn_csv'),      '#16a085', self._export_csv)
        self._btn_chart    = self._make_btn(self._t('btn_chart'),    '#d35400', self._export_chart)
        self._btn_import   = self._make_btn(self._t('btn_import'),   '#7f8c8d', self._import_csv)

        # AUDIO IN chip — Frodo-UX memo: between Language and Import.
        # The chip is the user's at-a-glance health indicator when the
        # tuner goes silent. Clicking opens the picker modal.
        self._audio_chip = AudioChip(self._t)
        self._audio_chip.clicked.connect(self._open_audio_picker)
        self._audio_chip.setMinimumWidth(220)
        # Reflect whatever state the engine has at this point. The
        # engine may not have run start() yet (we wait for the UI to
        # finish building); the chip will repaint when state_changed
        # fires.
        self._audio_chip.update_from_state(
            self._engine.state, '', '', 0)

        # Inputs row: instrument config + audio chip + import.
        toolbar.addWidget(self._grp_instr)
        toolbar.addWidget(self._grp_disp)
        toolbar.addWidget(self._grp_a4)
        toolbar.addWidget(self._grp_filter)
        toolbar.addWidget(self._grp_lang)
        toolbar.addWidget(self._audio_chip)
        toolbar.addWidget(self._btn_import)
        toolbar.addStretch()
        # Actions row: autotune + recording controls + exports.
        toolbar_actions.addWidget(self._btn_autotune)
        toolbar_actions.addWidget(self._btn_record)
        toolbar_actions.addWidget(self._btn_reset)
        toolbar_actions.addStretch()
        toolbar_actions.addWidget(self._btn_txt)
        toolbar_actions.addWidget(self._btn_pdf)
        toolbar_actions.addWidget(self._btn_chart)
        toolbar_actions.addWidget(self._btn_csv)
        root.addWidget(toolbar_container)

        # ── Splitter ──────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # Links: Tuner (+ optional spectrogram + diagnostics panel)
        left = QWidget()
        ll3 = QVBoxLayout(left)
        ll3.setContentsMargins(0, 0, 6, 0)
        # Recovery banner — appears only when the engine is in FAILED.
        # Hidden by default; show_for() makes it visible with the right copy.
        self._audio_banner = AudioRecoveryBanner(
            self._t, self._retry_audio, self._open_audio_picker)
        ll3.addWidget(self._audio_banner)

        self._tuner = TunerWidget()
        # Tuner gets a fixed-ish vertical slot now that it shares the
        # pane with optional panels. Expanding horizontally but only
        # Preferred vertically lets the spectrogram claim leftover space
        # when the panels are visible.
        self._tuner.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Preferred)
        # Tuner takes its full minimum height (260px) at the top of the
        # left pane. The spectrum analyzer + diagnostics panel sit below
        # it and only consume space when diagnostics are enabled.
        ll3.addWidget(self._tuner)
        self._status_lbl = QLabel(self._t('no_signal'))
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet('color:#888;font-size:13px;padding:4px;')
        ll3.addWidget(self._status_lbl)

        # v0.5.7: "Show spectrum analyzer & diagnostics" footer row.
        # Lives directly below the tuner so the toggle is adjacent to
        # the widgets it controls. Same handler, same persistence as
        # the old toolbar-resident checkbox — only the parent moved.
        diag_footer = QHBoxLayout()
        diag_footer.setContentsMargins(4, 0, 4, 0)
        self._cb_diag = QCheckBox(self._t('show_diagnostics'))
        self._cb_diag.setChecked(
            bool(getattr(self._cfg, 'show_diagnostics', False)))
        self._cb_diag.setToolTip(self._t('show_diagnostics_tip'))
        self._cb_diag.setStyleSheet("""
            QCheckBox { color: #bbb; font-size: 12px; padding: 2px 4px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
        """)
        self._cb_diag.toggled.connect(self._on_diagnostics_toggled)
        diag_footer.addWidget(self._cb_diag)
        diag_footer.addStretch()
        ll3.addLayout(diag_footer)

        # Spectrum analyzer + diagnostics panels. Always constructed so
        # the toggle is just show/hide — keeps the timer threading
        # consistent and avoids reconstruction cost when flipped
        # repeatedly. Attribute name kept as _spectro to minimize churn
        # against the rest of the file's references.
        self._spectro_grp = QGroupBox(self._t('spectro_title'))
        sg_l = QVBoxLayout(self._spectro_grp)
        sg_l.setContentsMargins(6, 6, 6, 6)
        self._spectro = SpectrumAnalyzerWidget(self._engine if AUDIO_OK else None)
        sg_l.addWidget(self._spectro)
        ll3.addWidget(self._spectro_grp, 1)

        self._data_grp = QGroupBox(self._t('data_panel_title'))
        dg_l = QVBoxLayout(self._data_grp)
        dg_l.setContentsMargins(6, 6, 6, 6)
        self._data_panel = DataPanelWidget(
            self._engine if AUDIO_OK else None,
            self._t,
            lambda: self._notes_count,
            lambda: self._cfg,
        )
        dg_l.addWidget(self._data_panel)
        ll3.addWidget(self._data_grp)

        show_diag = bool(getattr(self._cfg, 'show_diagnostics', False))
        self._spectro_grp.setVisible(show_diag)
        self._data_grp.setVisible(show_diag)

        # Rechts: Tabelle
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        self._table_lbl = QLabel(self._t('table_empty_hint'))
        self._table_lbl.setStyleSheet('font-size:14px;font-weight:bold;color:#ccc;padding:2px 0 6px 0;')
        self._table_lbl.setWordWrap(True)
        rl.addWidget(self._table_lbl)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(self._table_headers())
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget{background:#1a1a24;color:#ddd;gridline-color:#333;
                         font-size:14px;border:none;}
            QTableWidget::item{padding:5px 8px;}
            QTableWidget::item:alternate{background:#1f1f2e;}
            QTableWidget::item:selected{background:#2d4a7a;color:#fff;}
            QHeaderView::section{background:#252535;color:#aaa;font-size:12px;
                                  padding:6px;border:none;border-bottom:1px solid #444;}
        """)
        rl.addWidget(self._table)
        self._bar_delegate = CentBarDelegate(
            self._table, sample_rate_getter=self._engine_sample_rate)
        self._table.setItemDelegateForColumn(5, self._bar_delegate)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)

        # Min-N filter under the table: hide notes with fewer measurements
        # than the threshold so single accidental blips don't pollute the
        # view. Default 5 = "actually held". Mirrors the autotune
        # min-n requirement.
        min_n_row = QHBoxLayout()
        min_n_row.setContentsMargins(0, 6, 0, 0)
        self._min_n_lbl = QLabel(self._t('min_n_label'))
        self._min_n_lbl.setStyleSheet('color:#aaa;font-size:12px;')
        self._min_n_spin = QSpinBox()
        self._min_n_spin.setRange(0, 999)
        self._min_n_spin.setValue(int(getattr(self._cfg, 'min_n_visible', 5)))
        self._min_n_spin.setToolTip(self._t('min_n_tip'))
        self._min_n_spin.setMinimumWidth(70)
        self._min_n_spin.setStyleSheet(
            'QSpinBox{background:#1e1e2e;color:#ddd;border:1px solid #444;'
            'border-radius:5px;padding:2px 6px;font-size:12px;}')
        self._min_n_spin.valueChanged.connect(self._on_min_n_changed)
        min_n_row.addWidget(self._min_n_lbl)
        min_n_row.addWidget(self._min_n_spin)
        # v0.5.7: "Filter to instrument range" checkbox lives in this
        # row alongside the min-N spinbox. Both controls gate what the
        # table shows; co-locating them lets the user adjust scope in
        # one place instead of hunting in the top toolbar. UI logic
        # preserved verbatim: checked = filter ON, unchecked = show all
        # (stored as the inverted cfg.allow_out_of_range).
        self._cb_oor = QCheckBox(self._t('allow_oor'))
        self._cb_oor.setChecked(not self._cfg.allow_out_of_range)
        self._cb_oor.setToolTip(self._t('allow_oor_tip'))
        self._cb_oor.setStyleSheet("""
            QCheckBox { color: #bbb; font-size: 12px; padding: 2px 4px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
        """)
        self._cb_oor.toggled.connect(self._on_oor_toggled)
        min_n_row.addSpacing(16)
        min_n_row.addWidget(self._cb_oor)
        min_n_row.addStretch()
        rl.addLayout(min_n_row)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 620])
        root.addWidget(splitter, 1)
        # Held for closeEvent persistence and restore-on-launch (v0.5.5).
        self._splitter = splitter

        # Fensterstil
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#12121a;color:#ddd;}
            QGroupBox{border:1px solid #333;border-radius:6px;font-size:12px;
                      color:#aaa;margin-top:6px;padding-top:4px;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;top:-2px;}
            QComboBox{background:#1e1e2e;border:1px solid #444;border-radius:5px;
                      color:#ddd;padding:4px 8px;font-size:13px;min-height:28px;}
            QComboBox:hover{background:#252535;border:1px solid #6699cc;}
            QComboBox::drop-down{border:none;width:20px;background:#1e1e2e;}
            QComboBox QAbstractItemView{background:#1e1e2e;color:#ddd;
                      border:1px solid #444;outline:0;
                      selection-background-color:#34495e;
                      selection-color:#fff;}
            QComboBox QAbstractItemView::item{background:#1e1e2e;color:#ddd;
                      padding:4px 8px;border:none;}
            QComboBox QAbstractItemView::item:selected{background:#34495e;
                      color:#fff;}
            QComboBox QAbstractItemView::item:hover{background:#2a2a3a;
                      color:#fff;}
            QSplitter::handle{background:#333;}
        """)

        self.setWindowTitle(self._t('window_title'))

    def _table_headers(self):
        return [self._t('col_fingered'), self._t('col_sounding'),
                self._t('col_mean'), self._t('col_std'),
                self._t('col_n'), self._t('col_tendency')]

    def _make_btn(self, text, color, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        b.setMinimumHeight(36)
        b.setMinimumWidth(150)
        b.setStyleSheet(f"""
            QPushButton{{background:{color};color:white;border:none;
                         border-radius:6px;font-size:13px;padding:0 12px;}}
            QPushButton:hover{{background:{color}cc;}}
            QPushButton:pressed{{background:{color}99;}}
        """)
        return b

    # ── Sprache wechseln ─────────────────────────────────────────────────────
    def _on_lang_changed(self, idx):
        self.lang = self._lang_combo.itemData(idx)
        self._retranslate()

    def _retranslate(self):
        """Aktualisiert alle beschrifteten Widgets ohne Neuaufbau."""
        self.setWindowTitle(self._t('window_title'))
        self._grp_instr.setTitle(self._t('grp_instrument'))
        self._grp_disp.setTitle(self._t('grp_display'))
        self._grp_a4.setTitle(self._t('grp_a4'))
        self._grp_filter.setTitle(self._t('grp_filter'))
        self._grp_lang.setTitle(self._t('grp_language'))

        # Family + sub-instrument combos re-populated in current language.
        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        for family_key, name_de, name_en in instrument_families():
            label = name_de if self.lang == 'de' else name_en
            self._family_combo.addItem(label, family_key)
        self._family_combo.blockSignals(False)
        self._select_family_for_instrument(self.instrument)
        self._populate_instrument_combo(select_key=self.instrument)
        self._btn_custom.setText(self._t('custom_label'))
        self._btn_custom.setToolTip(self._t('custom_dlg_title'))
        if hasattr(self, '_btn_range'):
            self._btn_range.setToolTip(self._t('gear_tip'))
        self._nick_edit.setPlaceholderText(self._t('nickname_tip'))

        # Display-Combo
        idx_d = self._disp_combo.currentIndex()
        self._disp_combo.blockSignals(True)
        self._disp_combo.clear()
        self._disp_combo.addItems([self._t('disp_griff'), self._t('disp_klingend')])
        self._disp_combo.setCurrentIndex(idx_d)
        self._disp_combo.blockSignals(False)

        # Layout-Combo — re-label the Auto/List/Grid items in the new lang.
        if hasattr(self, '_layout_combo'):
            cur = self._layout_combo.currentData()
            self._layout_combo.blockSignals(True)
            self._layout_combo.clear()
            self._layout_combo.addItem(self._t('layout_auto'), 'auto')
            self._layout_combo.addItem(self._t('layout_single'), 'single')
            self._layout_combo.addItem(self._t('layout_matrix'), 'matrix')
            for i in range(self._layout_combo.count()):
                if self._layout_combo.itemData(i) == cur:
                    self._layout_combo.setCurrentIndex(i)
                    break
            self._layout_combo.blockSignals(False)

        # Filter-mode combo — re-label Fast/Normal/Slow.
        if hasattr(self, '_filter_combo'):
            cur = self._filter_combo.currentData()
            self._filter_combo.blockSignals(True)
            self._filter_combo.clear()
            self._filter_combo.addItem(self._t('filter_fast'),   'fast')
            self._filter_combo.addItem(self._t('filter_normal'), 'normal')
            self._filter_combo.addItem(self._t('filter_slow'),   'slow')
            for i in range(self._filter_combo.count()):
                if self._filter_combo.itemData(i) == cur:
                    self._filter_combo.setCurrentIndex(i)
                    break
            self._filter_combo.setToolTip(self._t('filter_tip'))
            self._filter_combo.blockSignals(False)
        if hasattr(self, '_min_n_lbl'):
            self._min_n_lbl.setText(self._t('min_n_label'))
            self._min_n_spin.setToolTip(self._t('min_n_tip'))
        # v0.5.7: "Filter to instrument range" checkbox relocated next
        # to the min-N spinbox; same retranslate behaviour as before.
        if hasattr(self, '_cb_oor'):
            self._cb_oor.setText(self._t('allow_oor'))
            self._cb_oor.setToolTip(self._t('allow_oor_tip'))

        # Buttons
        self._btn_autotune.setText(self._t('btn_autotune'))
        self._btn_record.setText(self._t('btn_stop' if self._recording else 'btn_start'))
        self._btn_reset.setText(self._t('btn_reset'))
        self._btn_txt.setText(self._t('btn_txt'))
        self._btn_pdf.setText(self._t('btn_pdf'))
        self._btn_csv.setText(self._t('btn_csv'))
        self._btn_chart.setText(self._t('btn_chart'))
        self._btn_import.setText(self._t('btn_import'))

        # Tabellen-Header
        self._table.setHorizontalHeaderLabels(self._table_headers())

        # Status & Tabellenlabel
        self._status_lbl.setText(self._t('no_signal'))

        # Diagnose-Panel + Spektrogramm.
        if hasattr(self, '_cb_diag'):
            self._cb_diag.setText(self._t('show_diagnostics'))
            self._cb_diag.setToolTip(self._t('show_diagnostics_tip'))
        if hasattr(self, '_spectro_grp'):
            self._spectro_grp.setTitle(self._t('spectro_title'))
        if hasattr(self, '_data_grp'):
            self._data_grp.setTitle(self._t('data_panel_title'))
        if hasattr(self, '_data_panel'):
            self._data_panel.retranslate(self._t)
        if hasattr(self, '_audio_chip'):
            self._audio_chip.retranslate(self._t)
        if hasattr(self, '_audio_banner'):
            self._audio_banner.retranslate(self._t)

        self._refresh_table()

    # ── Audio-Callback ────────────────────────────────────────────────────────
    def _on_note(self, midi_kl: int, freq: float, cents: float):
        if not self._recording:
            return
        # Drop out-of-range notes when the toggle is off — keeps the table
        # bounded to the instrument's nominal range.
        if not self._cfg.allow_out_of_range:
            transp = TRANSP_MAP.get(self.instrument, 0)
            lo_f, hi_f = sax_instruments.fingered_range(self.instrument)
            midi_fingered = midi_kl - transp
            if not (lo_f <= midi_fingered <= hi_f):
                return
        with self._lock:
            if midi_kl not in self.stats:
                self.stats[midi_kl] = NoteStats()
            self.stats[midi_kl].add(cents)
        # Bump the diagnostics counter outside the lock — single thread
        # writes it, single thread reads via the timer; integer add is
        # atomic enough for a display-only counter.
        self._notes_count += 1
        # Per-measurement log. Instrument/A4 are read off the active run
        # inside the log, not from `self`, so a callback firing during a UI
        # change still attributes to the run that was active when it fired.
        midi_gr = midi_kl - TRANSP_MAP.get(self.instrument, 0)
        self._log.add_measurement(midi_sounding=midi_kl,
                                   midi_fingered=midi_gr,
                                   cents=cents, freq_hz=freq)

        transp     = TRANSP_MAP.get(self.instrument, 0)
        midi_gr    = midi_kl - transp
        kl_name    = midi_note_name(midi_kl)
        gr_name    = midi_note_name(midi_gr)
        disp_name  = gr_name if self.display == 'griff' else kl_name
        sr_now = self._engine_sample_rate()
        self._tuner.sample_rate = sr_now
        self._tuner.set_note(disp_name, freq, cents)
        cents_str = format_cents(cents, freq, sr_now)
        self._status_lbl.setText(self._t(
            'status_fmt', fingered=gr_name, sounding=kl_name,
            freq=freq, cents_str=cents_str, a4=self._engine.a4))
        # Highlight the row currently being played so the user can see
        # which entry in a long table just ticked. _refresh_table reads
        # this on its next tick (every 300ms via _refresh_timer).
        self._active_midi = midi_kl
        self._active_midi_at = datetime.datetime.now()

    # ── Tabelle ──────────────────────────────────────────────────────────────────────
    # Two layout modes:
    #   'single' — single-column-of-notes table (1 row per played note +
    #              1 row per seeded expected note).
    #   'matrix' — piano-roll: 12 chromatic rows × N octave columns, where
    #              N covers the instrument's range padded ±1 octave. Cells
    #              outside the instrument's range render greyed out.
    # The mode is chosen automatically based on the available width.
    _MATRIX_COL_WIDTH = 112    # min usable width per octave column (px)
    _MATRIX_HEADER_W  = 64     # row-header width (note names)
    _MATRIX_ROW_HEIGHT = 60    # tall enough for: names + mean+std + bar + N
    # Hysteresis: enter matrix at the calculated threshold, drop back to
    # single only when noticeably narrower. Prevents thrash near the edge.
    _MATRIX_HYSTERESIS = 48    # px

    def _refresh_table(self):
        if not hasattr(self, '_table'):
            return
        desired = self._desired_layout_mode()
        if desired != self._layout_mode:
            self._layout_mode = desired
            self._configure_table_for_mode(desired)
        if desired == 'matrix':
            self._refresh_table_matrix()
        else:
            self._refresh_table_single()

    def _desired_layout_mode(self) -> str:
        """User-preference first; falls back to a width-driven auto pick.
        In auto mode, matrix is used as long as the viewport can fit at
        least two octave columns; below that, single-column for
        readability. Matrix never truncates a playable note — wider
        instruments overflow into a horizontal scrollbar instead of
        dropping back to single."""
        pref = getattr(self._cfg, 'layout_mode_preference', 'auto')
        if pref == 'single':
            return 'single'
        if pref == 'matrix':
            return 'matrix'
        if not hasattr(self, '_table'):
            return 'single'
        w = self._table.viewport().width()
        if w <= 0:
            w = self._table.width()
        if w <= 0:
            return 'single'
        floor = self._MATRIX_HEADER_W + 2 * self._MATRIX_COL_WIDTH
        if self._layout_mode == 'matrix':
            return 'matrix' if w >= floor - self._MATRIX_HYSTERESIS else 'single'
        return 'matrix' if w >= floor else 'single'

    def _matrix_octave_range(self) -> tuple[int, int]:
        """(lo_octave, hi_octave) inclusive to display for the current
        instrument. Spans the instrument's nominal fingered range AND any
        actually-played notes outside it — overtones, altissimo, and
        accidentals get their own cells so nothing gets truncated.

        Half-step-beyond rule: if the low note is exactly a C (the start
        of its octave) we pad one column below so the B a half-step lower
        is visible; if the high note is exactly a B (the end of its
        octave) we pad one column above so the C a half-step higher is
        visible. Extra context octaves beyond that are configurable via
        cfg.matrix_extra_octaves."""
        transp = TRANSP_MAP.get(self.instrument, 0)
        lo_f, hi_f = sax_instruments.fingered_range(self.instrument)
        if self.display == 'griff':
            lo_midi, hi_midi = lo_f, hi_f
        else:
            lo_midi, hi_midi = lo_f + transp, hi_f + transp
        # Played-note expansion (only when OOR is allowed).
        with self._lock:
            played = [m for m, st in self.stats.items() if st.n > 0]
        if played and self._cfg.allow_out_of_range:
            if self.display == 'griff':
                played = [m - transp for m in played]
            lo_midi = min(lo_midi, min(played))
            hi_midi = max(hi_midi, max(played))
        lo_oct = lo_midi // 12 - 1
        hi_oct = hi_midi // 12 - 1
        # Half-step-beyond rule.
        if lo_midi % 12 == 0:      # low note is C → show B in the column below
            lo_oct -= 1
        if hi_midi % 12 == 11:     # high note is B → show C in the column above
            hi_oct += 1
        # Configurable extra context on each side.
        extra = max(0, int(getattr(self._cfg, 'matrix_extra_octaves', 0)))
        lo_oct -= extra
        hi_oct += extra
        # Clamp to non-negative octaves (MIDI octave -1 not useful for
        # any real instrument in this app).
        lo_oct = max(0, lo_oct)
        return (lo_oct, hi_oct)

    def _matrix_octave_count(self) -> int:
        lo, hi = self._matrix_octave_range()
        return hi - lo + 1

    def _configure_table_for_mode(self, mode: str) -> None:
        """Swap the table between single-column and matrix layouts.

        Reuses cached delegates rather than constructing per call. The
        single-mode delegate is the existing CentBarDelegate on column 5;
        matrix mode replaces it with a default delegate (PyQt6 won't
        accept None) and installs MatrixCellDelegate on every cell."""
        if not hasattr(self, '_default_delegate'):
            self._default_delegate = QStyledItemDelegate(self._table)
        if not hasattr(self, '_matrix_delegate'):
            self._matrix_delegate = MatrixCellDelegate(
                self._table, sample_rate_getter=self._engine_sample_rate)
        if mode == 'matrix':
            n_oct = self._matrix_octave_count()
            self._table.clear()
            self._table.setColumnCount(n_oct)
            self._table.setRowCount(12)
            self._table.verticalHeader().setVisible(True)
            self._table.verticalHeader().setDefaultSectionSize(self._MATRIX_ROW_HEIGHT)
            # Fixed column widths — playable notes always render at full
            # cell size. If the columns don't all fit, Qt's horizontal
            # scrollbar takes over instead of the cells getting squished.
            hh = self._table.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            hh.setDefaultSectionSize(self._MATRIX_COL_WIDTH)
            for c in range(n_oct):
                self._table.setColumnWidth(c, self._MATRIX_COL_WIDTH)
            self._table.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._table.setItemDelegateForColumn(5, self._default_delegate)
            self._table.setItemDelegate(self._matrix_delegate)
        else:
            self._table.clear()
            self._table.setColumnCount(6)
            self._table.verticalHeader().setVisible(False)
            self._table.verticalHeader().setDefaultSectionSize(28)
            self._table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch)
            self._table.setHorizontalHeaderLabels(self._table_headers())
            # Restore the default per-table delegate, then the bar delegate
            # for the tendency column.
            self._table.setItemDelegate(self._default_delegate)
            self._table.setItemDelegateForColumn(5, self._bar_delegate)

    def _active_midi_now(self):
        """Currently-played MIDI if the highlight is still fresh, else None."""
        if (self._active_midi_at is None
                or (datetime.datetime.now() - self._active_midi_at)
                .total_seconds() > 1.5):
            return None
        return self._active_midi

    def _refresh_table_single(self):
        transp     = TRANSP_MAP.get(self.instrument, 0)
        disp_griff = (self.display == 'griff')

        if disp_griff:
            hdrs = [self._t('col_fingered'), self._t('col_sounding')]
        else:
            hdrs = [self._t('col_sounding'), self._t('col_fingered')]
        self._table.setHorizontalHeaderLabels(
            hdrs + [self._t('col_mean'), self._t('col_std'),
                    self._t('col_n'), self._t('col_tendency')])

        with self._lock:
            raw_items = sorted(self.stats.items())

        # Min-N filter: rows with 1..min_n-1 measurements are below
        # threshold and hidden as noise. N=0 seeded blanks still show so
        # the instrument range stays visible as a guide.
        min_n = max(0, int(getattr(self._cfg, 'min_n_visible', 0)))
        items = [(m, s) for (m, s) in raw_items
                 if s.n == 0 or s.n >= min_n]

        self._table.setRowCount(len(items))
        played_n = 0
        active_midi = self._active_midi_now()
        sr_now = self._engine_sample_rate()
        a4 = self._engine.a4
        for row, (midi_kl, st) in enumerate(items):
            midi_gr = midi_kl - transp
            kl_name = midi_note_name(midi_kl)
            gr_name = midi_note_name(midi_gr)
            n1, n2  = (gr_name, kl_name) if disp_griff else (kl_name, gr_name)
            mean    = st.mean
            has_data = st.n > 0
            if has_data:
                played_n += 1

            col = (QColor('#3a9e5f') if abs(mean) <= 5 else
                   QColor('#c8a020') if abs(mean) <= 12 else QColor('#c03030'))
            dim_col = QColor('#555')

            # Nominal frequency for this MIDI: drives the precision floor.
            note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
            mean_str = format_cents(mean, note_freq, sr_now) if has_data else '–'
            if st.n > 1:
                std_str = '±' + format_cents(st.std, note_freq, sr_now).lstrip('+-')
            else:
                std_str = '–'

            is_active = (active_midi == midi_kl)
            for c, val in enumerate([
                n1, n2,
                mean_str,
                std_str,
                str(st.n) if has_data else '–',
                '',
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 2:
                    item.setForeground(col if has_data else dim_col)
                elif not has_data:
                    item.setForeground(dim_col)
                if c == 5 and has_data:
                    item.setData(Qt.ItemDataRole.UserRole,
                                 {'cents': mean, 'freq': note_freq})
                if is_active:
                    item.setBackground(QColor('#2c5a8a'))
                self._table.setItem(row, c, item)

        total = sum(s.n for _, s in items)
        if not items or played_n == 0:
            label = self._t('table_empty_hint')
        else:
            label = self._t('table_summary', notes=played_n, total=total)
        self._table_lbl.setText(label)

    def _refresh_table_matrix(self):
        transp     = TRANSP_MAP.get(self.instrument, 0)
        disp_griff = (self.display == 'griff')
        lo_f, hi_f = sax_instruments.fingered_range(self.instrument)
        lo_oct, hi_oct = self._matrix_octave_range()
        octaves = list(range(lo_oct, hi_oct + 1))

        # Fingered (griff) mode uses RELATIVE octave labels (-1, 0, +1)
        # centered on the instrument's middle octave — saxophone players
        # read "low Bb" as Bb3 in SPN, which feels mis-octaved against an
        # absolute scale. Concert (klingend) mode keeps absolute octave
        # numbers because sounding pitch IS absolute.
        if disp_griff:
            mid_oct = (lo_oct + hi_oct) // 2
            header_strings = [
                self._t('matrix_oct_rel_label', n=(o - mid_oct))
                for o in octaves
            ]
        else:
            header_strings = [
                self._t('matrix_oct_label', n=o) for o in octaves
            ]
        self._table.setHorizontalHeaderLabels(header_strings)
        # Row labels: pitch class only (octave lives in the column header).
        self._table.setVerticalHeaderLabels(
            [c.split('/')[0] for c in CHROMA])

        with self._lock:
            stats_by_midi = dict(self.stats)
        active_midi = self._active_midi_now()

        played_n = 0
        in_range_cells = 0
        for r in range(12):
            for c, oct_ in enumerate(octaves):
                midi_visible = (oct_ + 1) * 12 + r
                if disp_griff:
                    midi_fingered = midi_visible
                    midi_sounding = midi_visible + transp
                else:
                    midi_sounding = midi_visible
                    midi_fingered = midi_visible - transp
                in_range = lo_f <= midi_fingered <= hi_f
                if in_range:
                    in_range_cells += 1

                st = stats_by_midi.get(midi_sounding)
                # Apply min-N gate so single-blip cells don't render with
                # arbitrary cents readings. Same rule as single-column:
                # a cell with 1..min_n-1 hits is treated as if it has no
                # data yet.
                min_n = max(0, int(
                    getattr(self._cfg, 'min_n_visible', 0)))
                has_data = (st is not None and st.n > 0
                            and st.n >= min_n)
                if has_data:
                    played_n += 1
                # MatrixCellDelegate reads this dict and paints all six
                # data fields the single-column view shows per row:
                # fingered name, sounding name, mean, std, N, bar.
                # Nominal frequency at A4 drives the precision floor.
                note_freq = self._engine.a4 * (
                    2.0 ** ((midi_sounding - 69) / 12.0))
                payload = {
                    'mean':          st.mean if has_data else None,
                    'std':           st.std if has_data else None,
                    'n':             st.n if st is not None else 0,
                    'in_range':      in_range,
                    'active':        (active_midi == midi_sounding),
                    'fingered_name': midi_note_name(midi_fingered),
                    'sounding_name': midi_note_name(midi_sounding),
                    'freq':          note_freq,
                }
                item = QTableWidgetItem('')
                item.setData(Qt.ItemDataRole.UserRole, payload)
                self._table.setItem(r, c, item)

        if in_range_cells == 0:
            label = self._t('table_empty_hint')
        else:
            label = self._t('table_matrix_title',
                             played=played_n, total=in_range_cells)
        self._table_lbl.setText(label)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Defer the mode-decision one event-loop tick so the splitter has
        # time to propagate the new viewport size into the table widget
        # before _desired_layout_mode reads it. Without the singleShot,
        # rapid window shrinks sometimes leave the table in matrix mode
        # because viewport().width() still reflects the pre-resize size.
        if hasattr(self, '_table'):
            QTimer.singleShot(0, self._refresh_table)

    def _make_bar(self, cents, w=20):
        half = w // 2
        fill = int(min(1.0, abs(cents) / 40.0) * half)
        if cents > 1:
            return ' '*half + '\u2502' + '\u2588'*fill + '\u2591'*(half-fill)
        elif cents < -1:
            return '\u2591'*(half-fill) + '\u2588'*fill + '\u2502' + ' '*half
        return ' '*half + '\u2502' + ' '*half

    # ── Instrument / Anzeige / A4 ─────────────────────────────────────────────
    def _on_family_changed(self, _idx):
        """Family combo changed → repopulate the sub-instrument combo with
        the new family's entries and select the first one."""
        self._populate_instrument_combo(select_key=None)

    def _populate_instrument_combo(self, select_key: str | None) -> None:
        family_key = self._family_combo.currentData()
        if family_key is None:
            return
        self._instr_combo.blockSignals(True)
        self._instr_combo.clear()
        target_idx = 0
        for i, (key, name_de, name_en) in enumerate(instruments_in(family_key)):
            label = name_de if self.lang == 'de' else name_en
            self._instr_combo.addItem(label, key)
            if key == select_key:
                target_idx = i
        if self._instr_combo.count():
            self._instr_combo.setCurrentIndex(target_idx)
        self._instr_combo.blockSignals(False)
        if self._instr_combo.count():
            self._on_instr_changed(self._instr_combo.currentIndex())

    def _select_family_for_instrument(self, instrument_key: str) -> None:
        """Move the family combo to the family that contains the given key,
        without triggering a sub-instrument repopulate."""
        family_key = instrument_family_of(instrument_key)
        if family_key is None:
            return
        for i in range(self._family_combo.count()):
            if self._family_combo.itemData(i) == family_key:
                self._family_combo.blockSignals(True)
                self._family_combo.setCurrentIndex(i)
                self._family_combo.blockSignals(False)
                return

    def _on_instr_changed(self, idx):
        key = self._instr_combo.itemData(idx)
        if key is None:
            return
        # Family-combo scrubs re-populate the sub-instrument combo, which
        # auto-fires this handler even when the resolved key matches the
        # already-active instrument. Bail out so we don't spawn a fresh run
        # for every flicker through the family list.
        if key == self.instrument:
            return
        self.instrument = key
        self._engine.instr_key = self.instrument
        # Seed the stats with empty slots for every expected fingered note so
        # the table immediately shows the player what the instrument's range
        # looks like. Real measurements fill in as the player plays;
        # out-of-range notes (overtones, accidentals) appear automatically
        # via the existing _on_note path.
        self._seed_expected_notes()
        # Instrument switch ⇒ new run, so per-run aggregates stay coherent.
        # Empty predecessor runs are coalesced inside `start_run`.
        if AUDIO_OK and self._recording:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=self._engine.a4,
                                label=self._nick_edit.text().strip())
        self._refresh_table()

    def _open_range_editor(self) -> None:
        """Open the per-instrument range editor for the active instrument.
        Accepts → persist via sax_instruments override DB → refresh table
        so the new bounds take effect immediately."""
        key = self.instrument
        cur_lo, cur_hi = sax_instruments.fingered_range(key)
        baked_lo, baked_hi = sax_instruments.baked_fingered_range(key)
        has_baked = sax_instruments.has_baked_range(key)
        name = instrument_display_name(key, self.lang)
        # v0.5.7.1: pass display mode + instrument transposition so the
        # dialog can render values in whichever notation the user is
        # already reading on the matrix. File format stays fingered.
        transp = TRANSP_MAP.get(key, 0)
        dlg = RangeEditorDialog(
            self, self._t, key, name,
            cur_lo, cur_hi, baked_lo, baked_hi, has_baked,
            display=self.display, transp=transp)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Re-seed expected notes with the new bounds so the table
            # immediately reflects what changed.
            self._seed_expected_notes()
            self._refresh_table()

    def _seed_expected_notes(self) -> None:
        """Populate self.stats with empty NoteStats for the current
        instrument's expected fingered range. Existing measurements are
        preserved."""
        transp = TRANSP_MAP.get(self.instrument, 0)
        lo, hi = sax_instruments.fingered_range(self.instrument)
        with self._lock:
            for fingered in range(lo, hi + 1):
                sounding = fingered + transp
                if sounding in SAX_MIDI and sounding not in self.stats:
                    self.stats[sounding] = NoteStats()

    def _on_oor_toggled(self, checked: bool) -> None:
        """Persist the choice and refresh the table so the matrix can
        shrink back to nominal range when the filter goes on.

        UI semantic: `checked` means "filter to range ON" (restrict).
        Internally we store the inverse as `allow_out_of_range`."""
        self._cfg.allow_out_of_range = not checked
        sax_config.save_config(self._cfg)
        # Wave-1 bug #7: the docstring above promises a refresh; the
        # v0.5.3 implementation forgot to actually call _refresh_table,
        # so toggling "Filter to instrument range" only took effect on
        # the next other refresh.
        self._refresh_table()

    # ── Audio-Geräteverwaltung (v0.5.4) ───────────────────────────────────
    def _engine_sample_rate(self) -> int:
        """Best-known live sample rate for adaptive cent precision.
        Falls back to the constant default when the engine is FAILED or
        audio is disabled — defensive; nothing should call the formatter
        in those states, but a stale delegate paint can race teardown."""
        try:
            sr = int(getattr(self._engine, 'samplerate', 0) or 0)
        except Exception:
            sr = 0
        return sr if sr > 0 else int(_DEFAULT_SAMPLE_RATE_EXT)

    def _device_selection_from_cfg(self) -> DeviceSelection:
        return DeviceSelection(
            name=str(getattr(self._cfg, 'audio_device_name', '') or ''),
            host_api=str(getattr(self._cfg, 'audio_device_host_api', '') or ''),
            samplerate=int(getattr(self._cfg, 'audio_device_samplerate', 0) or 0),
        )

    def _persist_active_device(self) -> None:
        """Write the engine's currently-active device back to config.
        Index is deliberately NOT stored — only name + host API + rate,
        per Gandalf's persistence design."""
        dev = self._engine.get_active_device()
        if dev is None:
            return
        self._cfg.audio_device_name = dev.name
        self._cfg.audio_device_host_api = dev.host_api
        self._cfg.audio_device_samplerate = int(self._engine.samplerate or 0)
        sax_config.save_config(self._cfg)
        # v0.5.7: keep the engine's hot-plug auto-recovery hint in sync
        # with whatever's actually active. Otherwise the user picks a
        # new device, unplugs it, plugs it back in, and the engine
        # still resolves the stale launch-time hint.
        self._engine.set_preferred_hint(DeviceSelection(
            name=dev.name, host_api=dev.host_api,
            samplerate=int(self._engine.samplerate or 0)))

    def _open_audio_picker(self) -> None:
        if not AUDIO_OK:
            return
        dlg = AudioPickerDialog(self, self._t, self._engine, self._cfg,
                                self._engine.get_active_device())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chosen = dlg.chosen()
            if chosen is not None:
                pref = str(getattr(self._cfg, 'audio_samplerate_pref',
                                    'auto') or 'auto')
                self._engine.open_device(
                    DeviceSelection(name=chosen.name,
                                    host_api=chosen.host_api,
                                    samplerate=0),
                    samplerate_pref=pref)

    def _retry_audio(self) -> None:
        if not AUDIO_OK:
            return
        # v0.5.7: force a fresh PortAudio enumeration before re-opening
        # so a device plugged in after launch can be picked up. The old
        # ``engine.retry()`` reused the stale snapshot and never saw
        # the hot-plugged device.
        self._engine.retry_open()

    def _poll_devices(self) -> None:
        if not AUDIO_OK:
            return
        # refresh_devices() emits signals on diff — we just kick it.
        try:
            self._engine.refresh_devices()
        except Exception:
            pass

    def _on_engine_state(self, state, err, msg) -> None:
        """React to engine state transitions: update chip, banner, and
        diagnostics panel device label."""
        dev = self._engine.get_active_device()
        name = dev.name if dev else ''
        host = dev.host_api if dev else ''
        sr = int(getattr(self._engine, 'samplerate', 0) or 0)
        if hasattr(self, '_audio_chip'):
            self._audio_chip.update_from_state(state, name, host, sr)
        if hasattr(self, '_audio_banner'):
            if state == AudioEngineState.FAILED:
                self._audio_banner.show_for(err, name, msg)
            else:
                self._audio_banner.hide()
        if state == AudioEngineState.RUNNING:
            self._persist_active_device()
            if hasattr(self, '_tuner') and sr:
                self._tuner.sample_rate = sr
            # First-time-only notice if we wound up at a non-44.1k rate.
            if (sr and sr != 44100
                    and not bool(getattr(self._cfg,
                                          'audio_sr_notice_shown', False))):
                self._cfg.audio_sr_notice_shown = True
                sax_config.save_config(self._cfg)
                # Surface as a passive status-bar line, not a modal —
                # Frodo-UX memo: never block the user with a dialog
                # over sample-rate disclosure.
                if hasattr(self, '_status_lbl'):
                    self._status_lbl.setText(
                        self._t('audio_sr_notice', sr=sr))

    def _on_devices_changed(self, _devices) -> None:
        # Nothing to do at the MainWindow level — the picker reads
        # devices lazily on each open, and the chip is driven by state
        # changes, not the device list. Hook kept so unit tests / future
        # toasts can subscribe without rewiring the engine.
        pass

    def _on_interface_appeared(self, device: 'DeviceInfo') -> None:
        """Hot-plug toast for vendor-class interfaces. Polite, non-modal,
        Frodo-UX memo policy: only fires for matched vendor names so the
        user isn't trained to dismiss it on every wake-from-sleep."""
        # Use a QMessageBox with a short title — close enough to a toast
        # for v0.5.4. A custom non-blocking widget is v0.5.5 work.
        # v0.5.7.6: single-snapshot read. Reading active_device twice
        # opens a TOCTOU window — between the not-None check and the
        # .name access the audio worker can tear the stream down and
        # null the device out, raising AttributeError on .name.
        dev = self._engine.get_active_device()
        if dev is not None and dev.name == device.name:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(self._t('audio_chip_label'))
        box.setText(self._t('audio_toast_interface', name=device.name))
        btn_switch = box.addButton(self._t('audio_toast_switch'),
                                   QMessageBox.ButtonRole.AcceptRole)
        box.addButton(self._t('audio_toast_dismiss'),
                      QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is btn_switch:
            self._engine.open_device(DeviceSelection(
                name=device.name, host_api=device.host_api, samplerate=0))

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        """Show or hide the spectrogram + diagnostics panels and persist
        the choice. Widgets remain constructed either way so flipping
        the toggle has no allocation cost on the audio thread."""
        self._cfg.show_diagnostics = bool(checked)
        sax_config.save_config(self._cfg)
        if hasattr(self, '_spectro_grp'):
            self._spectro_grp.setVisible(bool(checked))
        if hasattr(self, '_data_grp'):
            self._data_grp.setVisible(bool(checked))
        self._refresh_table()

    def _on_layout_pref_changed(self, _idx: int) -> None:
        """User picked Auto / List / Grid. Persist and refresh the table."""
        pref = self._layout_combo.currentData()
        if pref in ('auto', 'single', 'matrix'):
            self._cfg.layout_mode_preference = pref
            sax_config.save_config(self._cfg)
            self._refresh_table()

    def _on_filter_mode_changed(self, _idx: int) -> None:
        """User picked Fast / Normal / Slow. Reroute live audio through
        the new preset and persist."""
        mode = self._filter_combo.currentData()
        if mode in _FILTER_PRESETS:
            self._cfg.filter_mode = mode
            sax_config.save_config(self._cfg)
            if AUDIO_OK:
                self._engine.set_filter_mode(mode)

    def _on_min_n_changed(self, value: int) -> None:
        """User adjusted the min-N filter. Persist + redraw the table.
        The filter is purely a display gate — measurements are still
        collected; rows just hide until they accumulate enough hits."""
        self._cfg.min_n_visible = max(0, int(value))
        sax_config.save_config(self._cfg)
        self._refresh_table()

    def _on_nickname_changed(self) -> None:
        """User finished editing the nickname. Stamp the new label onto the
        current run so the next CSV export and table summary pick it up."""
        nickname = self._nick_edit.text().strip()
        if AUDIO_OK and self._recording:
            self._log.set_current_run_metadata(label=nickname)

    def _on_add_custom(self) -> None:
        """Prompt the user for a custom instrument and register it."""
        result = self._ask_custom_instrument()
        if result is None:
            return
        name, transp = result
        # Build a stable key: 'custom_' + slugified name. If the slug collides
        # with an existing custom (different display name but same slug, e.g.
        # 'Mezzo' vs 'mezzo'), append a numeric suffix instead of silently
        # overwriting the older entry — old CSV exports may still reference
        # the original transposition for that key.
        base = 'custom_' + ''.join(
            c.lower() if c.isalnum() else '_' for c in name).strip('_')[:32]
        if not base or base == 'custom_':
            return
        existing_keys = set(build_transp_map().keys())
        key = base
        suffix = 2
        while key in existing_keys:
            # If the user re-typed the EXACT same display name, treat it as
            # an intentional re-add (let register_custom replace the row).
            existing = [c for c in sax_config.load_customs() if c.key == key]
            if existing and existing[0].name_en == name:
                break
            key = f"{base}_{suffix}"
            suffix += 1
        register_custom(key, transp, name, name)
        _rebuild_transp_map()
        # Persist for next session.
        customs = sax_config.load_customs()
        customs = [c for c in customs if c.key != key]
        customs.append(sax_config.CustomInstrument(
            key=key, transp=transp, name_de=name, name_en=name))
        sax_config.save_customs(customs)
        # Refresh combos and select the new instrument.
        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        for fk, de, en in instrument_families():
            self._family_combo.addItem(de if self.lang == 'de' else en, fk)
        self._family_combo.blockSignals(False)
        self._select_family_for_instrument(key)
        self._populate_instrument_combo(select_key=key)

    def _add_dialog_lang_toggle(self, dlg, layout, on_change=None):
        """Drop a tiny DE | EN toggle at the top of any modal dialog so the
        user can switch language without having to cancel out and find the
        main-window combo. `on_change`, if given, is called after the lang
        switch so the dialog can repaint its own labels."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addStretch()

        def make_btn(label, code):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(self.lang == code)
            btn.setFixedWidth(46)
            btn.setStyleSheet("""
                QPushButton{background:#2a2a3e;color:#bbb;border:1px solid #444;
                             border-radius:4px;padding:3px 8px;font-size:11px;}
                QPushButton:checked{background:#2c5282;color:white;border:1px solid #6699cc;}
                QPushButton:hover:!checked{background:#34344a;}
            """)
            return btn

        btn_de = make_btn('DE', 'de')
        btn_en = make_btn('EN', 'en')

        def switch(target):
            if self.lang == target:
                return
            # Flip via the main window's combo so all subscribers (the main
            # toolbar, labels, etc.) update too.
            for i in range(self._lang_combo.count()):
                if self._lang_combo.itemData(i) == target:
                    self._lang_combo.setCurrentIndex(i)
                    break
            btn_de.setChecked(self.lang == 'de')
            btn_en.setChecked(self.lang == 'en')
            # Update the dialog's own window title.
            dlg.setWindowTitle(self._t(dlg.property('_lang_title_key')
                                        or 'window_title'))
            if on_change is not None:
                on_change()

        btn_de.clicked.connect(lambda: switch('de'))
        btn_en.clicked.connect(lambda: switch('en'))
        row.addWidget(btn_de)
        row.addWidget(btn_en)
        layout.addLayout(row)

    def _show_welcome_dialog(self) -> None:
        """First-boot dialog. Offers opt-in for persistent JSONL log."""
        dlg = QDialog(self)
        dlg.setProperty('_lang_title_key', 'welcome_title')
        dlg.setWindowTitle(self._t('welcome_title'))
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #ccc; font-size: 13px; }
            QCheckBox { color: #ddd; font-size: 13px; padding: 6px 0; }
            QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 8px 18px; font-size: 13px;
            }
            QPushButton:hover  { background: #3a6da8; }
        """)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 14, 24, 18)

        # Language toggle at the top — load-bearing on first launch, since
        # this dialog fires before the user has touched the main combo.
        info = QLabel(self._t('welcome_info'))
        info.setWordWrap(True)
        cb = QCheckBox(self._t('welcome_persist'))
        cb.setChecked(False)
        default_path = sax_config.CONFIG_DIR / sax_config.DEFAULT_LOG_FILENAME
        path_lbl = QLabel(self._t('welcome_path', path=str(default_path)))
        path_lbl.setStyleSheet('color: #888; font-size: 11px;')
        path_lbl.setWordWrap(True)
        btns = QDialogButtonBox()
        btn_go = btns.addButton(self._t('welcome_continue'),
                                 QDialogButtonBox.ButtonRole.AcceptRole)
        btn_later = btns.addButton(self._t('welcome_skip'),
                                    QDialogButtonBox.ButtonRole.RejectRole)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        def relabel():
            info.setText(self._t('welcome_info'))
            cb.setText(self._t('welcome_persist'))
            path_lbl.setText(self._t('welcome_path', path=str(default_path)))
            btn_go.setText(self._t('welcome_continue'))
            btn_later.setText(self._t('welcome_skip'))

        self._add_dialog_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addWidget(cb)
        layout.addWidget(path_lbl)
        layout.addWidget(btns)
        accepted = (dlg.exec() == QDialog.DialogCode.Accepted)
        # Whether they accepted or skipped, mark welcome as shown so we don't
        # ask again. Persistence is only enabled if they ticked the box.
        self._cfg.welcome_shown = True
        if accepted and cb.isChecked():
            self._cfg.persistence_enabled = True
            # Rebuild the log on the chosen path. Existing in-memory entries
            # would be lost otherwise — but at this point the user has only
            # been in the app a few seconds at most, so that's fine.
            new_path = self._cfg.effective_log_path()
            if new_path:
                self._log = MeasurementLog(path=new_path)
                if AUDIO_OK:
                    self._log.start_run(instrument=self.instrument,
                                        a4_hz=self._engine.a4,
                                        label=self._nick_edit.text().strip())
        sax_config.save_config(self._cfg)

    def _ask_custom_instrument(self) -> tuple[str, int] | None:
        dlg = QDialog(self)
        dlg.setProperty('_lang_title_key', 'custom_dlg_title')
        dlg.setWindowTitle(self._t('custom_dlg_title'))
        dlg.setMinimumWidth(380)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #bbb; font-size: 13px; }
            QLineEdit, QSpinBox {
                background: #252540; border: 1px solid #444; border-radius: 5px;
                color: #eee; padding: 6px 10px; font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus { border: 1px solid #6699cc; }
            QDialogButtonBox QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 6px 18px; font-size: 13px;
            }
        """)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 14, 20, 16)
        info = QLabel(self._t('custom_dlg_info'))
        info.setStyleSheet('color: #888; font-size: 12px;')
        info.setWordWrap(True)
        form = QFormLayout()
        form.setSpacing(8)
        edit_name = QLineEdit()
        edit_name.setPlaceholderText('e.g. Eb Mezzo-Soprano')
        name_lbl = QLabel(self._t('custom_lbl_name'))
        form.addRow(name_lbl, edit_name)
        spin_transp = QSpinBox()
        spin_transp.setRange(-36, 36)
        spin_transp.setValue(0)
        transp_lbl = QLabel(self._t('custom_lbl_transp'))
        form.addRow(transp_lbl, spin_transp)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        def relabel():
            info.setText(self._t('custom_dlg_info'))
            name_lbl.setText(self._t('custom_lbl_name'))
            transp_lbl.setText(self._t('custom_lbl_transp'))

        self._add_dialog_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        name = edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, self._t('err_title'),
                                self._t('custom_err_name'))
            return None
        return (name, int(spin_transp.value()))

    def _on_disp_changed(self, idx):
        self.display = ['griff', 'klingend'][idx]
        self._refresh_table()

    def _on_a4_changed(self, idx):
        """Concert pitch changed. Cents are a function of frequency + A4,
        so the cents values stored in self.stats are invalidated — but the
        underlying frequencies are immutable and still live in the log.
        Re-derive the table by walking the log's measurements at the new
        A4 instead of throwing away everything the user just recorded."""
        new_a4 = float(self._a4_combo.itemData(idx))
        # Wave-1 bug #5: A4-change race. The audio callback can fire
        # _on_note between the moment we build `remapped` and the moment
        # we assign it to self.stats, dropping a measurement into the
        # old dict that's about to be discarded — or worse, into a
        # half-built dict. pause_emissions() tells the engine to drop
        # incoming detections while we rebuild; we hold self._lock
        # across the swap so any future emission seeing the new stats
        # never partially-overlaps with the rebuild.
        self._engine.pause_emissions()
        try:
            self._engine.set_a4(new_a4)
            remapped: dict[int, NoteStats] = {}
            for m in self._log.measurements():
                # v0.5.7.3: skip non-positive / non-finite freqs. An
                # imported CSV row with freq_hz=0 (or NaN/inf) would
                # otherwise call freq_to_midi -> math.log2(0) -> ValueError
                # and crash the A4 combo handler. Such rows carry no
                # tuning information; drop them silently.
                f = float(m.freq_hz)
                if not math.isfinite(f) or f <= 0.0:
                    continue
                midi_round, cents = cents_dev(f, new_a4)
                if midi_round in SAX_MIDI:
                    ns = remapped.setdefault(midi_round, NoteStats())
                    ns.add(cents)
            with self._lock:
                self.stats = remapped
        finally:
            self._engine.resume_emissions()
        # Re-seed expected blank rows on top of the remapped data so the
        # instrument's range still shows even after a clean re-derive.
        self._seed_expected_notes()

        # Open a new run at the new A4 so subsequent measurements are
        # tagged correctly; coalescing in `start_run` keeps empty runs out
        # of the log when the user is just scrubbing the combo.
        if AUDIO_OK and self._recording:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=new_a4,
                                label=self._nick_edit.text().strip())
        self._refresh_table()

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _on_record_toggle(self):
        self._recording = not self._recording
        if self._recording:
            # Resumed recording ⇒ open a fresh run for the log.
            if AUDIO_OK:
                self._log.start_run(instrument=self.instrument,
                                    a4_hz=self._engine.a4,
                                    label=self._nick_edit.text().strip())
            self._btn_record.setText(self._t('btn_stop'))
            self._btn_record.setStyleSheet(self._btn_record.styleSheet().replace('#2ecc71', '#b7770d').replace('#27ae60', '#b7770d'))
            self._update_record_btn_style()
            self._status_lbl.setText(self._t('no_signal'))
        else:
            self._log.end_run()
            self._update_record_btn_style()
            self._btn_record.setText(self._t('btn_start'))

    def _update_record_btn_style(self):
        if self._recording:
            color = '#b7770d'   # orange = läuft
        else:
            color = '#27ae60'   # grün = pausiert, klicken zum Starten
        self._btn_record.setStyleSheet(f"""
            QPushButton{{background:{color};color:white;border:none;
                         border-radius:6px;font-size:13px;padding:0 12px;}}
            QPushButton:hover{{background:{color}cc;}}
            QPushButton:pressed{{background:{color}99;}}
        """)
        self._btn_record.setText(
            self._t('btn_stop') if self._recording else self._t('btn_start'))

    # ── Tabellen-Kontextmenü ──────────────────────────────────────────────────
    def _on_table_context_menu(self, pos):
        row = self._table.rowAt(pos.y())
        if row < 0:
            return

        transp = TRANSP_MAP.get(self.instrument, 0)
        midi_kl: int | None = None

        if self._layout_mode == 'matrix':
            # Matrix mode: map (row, col) to the sounding MIDI for the
            # clicked cell. Row = chroma index (0..11), col = octave from
            # the current octave range.
            col = self._table.columnAt(pos.x())
            if col < 0:
                return
            lo_oct, _hi_oct = self._matrix_octave_range()
            oct_ = lo_oct + col
            midi_visible = (oct_ + 1) * 12 + row
            if self.display == 'griff':
                midi_kl = midi_visible + transp
            else:
                midi_kl = midi_visible
            # Only offer the action if this cell actually holds played data.
            with self._lock:
                st = self.stats.get(midi_kl)
            if st is None or st.n == 0:
                return
        else:
            # Single-column mode: row directly indexes into sorted stats.
            with self._lock:
                keys = sorted(self.stats.keys())
            if row >= len(keys):
                return
            midi_kl = keys[row]

        midi_gr  = midi_kl - transp
        note_str = f"{midi_note_name(midi_gr)} / {midi_note_name(midi_kl)}"

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:#1e1e2e; color:#ddd; border:1px solid #444;
                    font-size:13px; padding:4px; }
            QMenu::item { padding:6px 20px; border-radius:4px; }
            QMenu::item:selected { background:#c03030; color:white; }
        """)
        action = menu.addAction(f"\U0001f5d1  {self._t('ctx_discard')}")
        action.setData(midi_kl)

        chosen = menu.exec(self._table.viewport().mapToGlobal(pos))
        if chosen and chosen.data() == midi_kl:
            if QMessageBox.question(
                self, self._t('reset_title'),
                self._t('ctx_discard_confirm', note=note_str),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                with self._lock:
                    self.stats.pop(midi_kl, None)
                self._refresh_table()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _on_reset(self):
        if QMessageBox.question(
            self, self._t('reset_title'), self._t('reset_msg'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            with self._lock:
                self.stats.clear()
            # Reset implies "I might be switching instruments" — clear the
            # cached maker/model so the next export re-prompts.
            self._instr_info_asked = False
            self._refresh_table()

    # ── Kammerton ermitteln ───────────────────────────────────────────────────
    def _on_autotune(self):
        with self._lock:
            items = [(st.mean, st.n) for st in self.stats.values() if st.n >= 5]

        if len(items) < 3:
            QMessageBox.warning(self, self._t('autotune_title'), self._t('autotune_nodata'))
            return

        means   = np.array([m for m, _ in items])
        weights = np.array([math.sqrt(n) for _, n in items])
        order   = np.argsort(means)
        cumw    = np.cumsum(weights[order])
        offset_ct     = float(means[order][np.searchsorted(cumw, cumw[-1] / 2.0)])
        mean_weighted = float(np.sum(means * weights) / np.sum(weights))

        a4_current    = self._engine.a4
        a4_optimal    = a4_current * (2.0 ** (offset_ct / 1200.0))
        a4_rounded    = round(a4_optimal)
        a4_clamped    = max(430, min(450, a4_rounded))

        sign  = '+' if offset_ct    >= 0 else ''
        sign2 = '+' if mean_weighted >= 0 else ''
        msg   = self._t('autotune_result',
                         notes=len(items), sign=sign, offset=offset_ct,
                         sign2=sign2, mean=mean_weighted,
                         current=a4_current, optimal=a4_optimal, rounded=a4_rounded)
        if a4_clamped != a4_rounded:
            msg += self._t('autotune_clamp', rounded=a4_rounded, clamped=a4_clamped)
        msg += self._t('autotune_confirm', clamped=a4_clamped)

        if QMessageBox.question(
            self, self._t('autotune_title'), msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self._a4_combo.blockSignals(True)
            self._a4_combo.setCurrentIndex(a4_clamped - 430)
            self._a4_combo.blockSignals(False)
            self._engine.a4 = float(a4_clamped)
            with self._lock:
                self.stats.clear()
            self._refresh_table()

    # ── Instrument-Modell-Dialog ──────────────────────────────────────────────
    def _ask_instrument_model(self) -> tuple[str, str] | None:
        """Prompt for maker + model. After the first answer in a session,
        subsequent calls return the cached values without re-prompting.
        Reset clears the cache so a new instrument can be tagged."""
        if getattr(self, '_instr_info_asked', False):
            return (getattr(self, '_last_maker', ''),
                    getattr(self, '_last_model', ''))
        dlg = QDialog(self)
        dlg.setProperty('_lang_title_key', 'model_dialog_title')
        dlg.setWindowTitle(self._t('model_dialog_title'))
        dlg.setMinimumWidth(440)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #bbb; font-size: 13px; }
            QLineEdit {
                background: #252540; border: 1px solid #444; border-radius: 5px;
                color: #eee; padding: 6px 10px; font-size: 13px;
            }
            QLineEdit:focus { border: 1px solid #6699cc; }
            QDialogButtonBox QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 6px 18px; font-size: 13px;
            }
            QDialogButtonBox QPushButton:hover   { background: #3a6da8; }
            QDialogButtonBox QPushButton:pressed  { background: #1e3a5f; }
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        info = QLabel(self._t('model_dialog_info'))
        info.setStyleSheet('color: #888; font-size: 12px;')
        info.setWordWrap(True)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        edit_maker = QLineEdit()
        edit_maker.setPlaceholderText(self._t('model_placeholder_maker'))
        if hasattr(self, '_last_maker'):
            edit_maker.setText(self._last_maker)

        edit_model = QLineEdit()
        edit_model.setPlaceholderText(self._t('model_placeholder_model'))
        if hasattr(self, '_last_model'):
            edit_model.setText(self._last_model)

        maker_lbl = QLabel(self._t('model_label_maker'))
        model_lbl = QLabel(self._t('model_label_model'))
        form.addRow(maker_lbl, edit_maker)
        form.addRow(model_lbl, edit_model)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        def relabel():
            info.setText(self._t('model_dialog_info'))
            maker_lbl.setText(self._t('model_label_maker'))
            model_lbl.setText(self._t('model_label_model'))
            edit_maker.setPlaceholderText(self._t('model_placeholder_maker'))
            edit_model.setPlaceholderText(self._t('model_placeholder_model'))

        self._add_dialog_lang_toggle(dlg, layout, on_change=relabel)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        maker = edit_maker.text().strip()
        model = edit_model.text().strip()
        # Für nächsten Export merken
        self._last_maker = maker
        self._last_model = model
        self._instr_info_asked = True
        return maker, model

    # ── Export TXT ────────────────────────────────────────────────────────────
    def _export_txt(self):
        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        maker, model = model_info

        path, _ = QFileDialog.getSaveFileName(
            self, self._t('txt_save_title'),
            f"intonation_{self.instrument}_{_today()}.txt",
            self._t('txt_filter'))
        if not path:
            return
        transp = TRANSP_MAP.get(self.instrument, 0)
        instr_key = f'instr_long_{self.instrument}'
        lines = [
            self._t('txt_header'),
            '=' * 54,
            self._t('txt_instr', name=self._t(instr_key)),
        ]
        if maker:
            lines.append(self._t('txt_maker', maker=maker))
        if model:
            lines.append(self._t('txt_model', model=model))
        lines += [
            (self._t('txt_transp', note=CHROMA[transp % 12])
             if transp else self._t('txt_no_transp')),
            self._t('txt_a4', hz=self._engine.a4),
            self._t('txt_date', dt=datetime.datetime.now().strftime('%d.%m.%Y %H:%M')),
            '',
            self._t('txt_col_header',
                    fingered=self._t('pdf_col_finger'),
                    sounding=self._t('pdf_col_sound'),
                    mean=self._t('txt_col_mean'),
                    std=self._t('txt_col_std'),
                    n='N'),
            '-' * 62,
        ]
        with self._lock:
            items = sorted(self.stats.items())
        sr_now = self._engine_sample_rate()
        a4 = self._engine.a4
        for midi_kl, st in items:
            note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
            mean_str = format_cents(st.mean, note_freq, sr_now)
            std_str = format_cents(st.std, note_freq, sr_now).lstrip('+-')
            lines.append(
                f"{midi_note_name(midi_kl - transp):<12} {midi_note_name(midi_kl):<12}"
                f" {mean_str:>7}   {std_str:>6}  {st.n:>5}  {self._make_bar(st.mean, 24)}")
        lines += ['', self._t('txt_total', total=sum(s.n for _,s in items), notes=len(items))]
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            QMessageBox.information(self, self._t('export_title'), self._t('txt_saved', path=path))
        except Exception as e:
            QMessageBox.critical(self, self._t('err_title'), str(e))

    # ── Export PDF ────────────────────────────────────────────────────────────
    def _export_pdf(self):
        try:
            from reportlab.lib.pagesizes import A4 as RL_A4
            from reportlab.lib import colors
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        except ImportError:
            QMessageBox.critical(self, self._t('err_title'), self._t('reportlab_err'))
            return

        model_info = self._ask_instrument_model()
        if model_info is None:
            return   # Abgebrochen
        maker, model = model_info

        path, _ = QFileDialog.getSaveFileName(
            self, self._t('pdf_save_title'),
            f"intonation_{self.instrument}_{_today()}.pdf",
            self._t('pdf_filter'))
        if not path:
            return

        transp = TRANSP_MAP.get(self.instrument, 0)
        doc    = SimpleDocTemplate(path, pagesize=RL_A4,
                                   leftMargin=20*mm, rightMargin=20*mm,
                                   topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story  = []

        ts_title = ParagraphStyle('T', parent=styles['Title'],
                                  fontSize=18, textColor=colors.HexColor('#1a237e'), spaceAfter=4)
        ts_sub   = ParagraphStyle('S', parent=styles['Normal'],
                                  fontSize=11, textColor=colors.HexColor('#555'), spaceAfter=2)

        story.append(Paragraph(self._t('pdf_title'), ts_title))
        story.append(Paragraph(self._t(f'instr_long_{self.instrument}'), ts_sub))
        if maker:
            story.append(Paragraph(self._t('pdf_maker', maker=maker), ts_sub))
        if model:
            story.append(Paragraph(self._t('pdf_model', model=model), ts_sub))
        story.append(Paragraph(
            (self._t('pdf_transp', note=CHROMA[transp % 12], n=transp)
             if transp else self._t('pdf_no_transp')), ts_sub))
        story.append(Paragraph(self._t('pdf_a4', hz=self._engine.a4), ts_sub))
        story.append(Paragraph(
            self._t('pdf_created', dt=datetime.datetime.now().strftime('%d.%m.%Y %H:%M')), ts_sub))
        story.append(Spacer(1, 10*mm))

        with self._lock:
            items = sorted(self.stats.items())

        if not items:
            story.append(Paragraph(self._t('pdf_no_data'), styles['Normal']))
        else:
            total_n = sum(s.n for _, s in items)
            story.append(Paragraph(
                self._t('pdf_summary', notes=len(items), total=total_n),
                ParagraphStyle('I', parent=styles['Normal'],
                               fontSize=10, textColor=colors.HexColor('#666'), spaceAfter=6)))

            data = [[self._t('pdf_col_finger'), self._t('pdf_col_sound'),
                     self._t('pdf_col_mean'), self._t('pdf_col_std'),
                     self._t('pdf_col_n'), self._t('pdf_col_tend')]]
            sr_now = self._engine_sample_rate()
            a4 = self._engine.a4
            for midi_kl, st in items:
                note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
                mean_str = format_cents(st.mean, note_freq, sr_now)
                std_str = format_cents(st.std, note_freq, sr_now).lstrip('+-')
                data.append([
                    midi_note_name(midi_kl - transp), midi_note_name(midi_kl),
                    mean_str,
                    f"\u00b1{std_str}" if st.n > 1 else '\u2013',
                    str(st.n),
                    self._make_bar_ascii(st.mean),
                ])

            tbl = Table(data, colWidths=[30*mm, 30*mm, 28*mm, 22*mm, 16*mm, 44*mm], repeatRows=1)
            ts = TableStyle([
                ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#1a237e')),
                ('TEXTCOLOR',    (0,0), (-1,0),  colors.white),
                ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
                ('FONTSIZE',     (0,0), (-1,-1), 10),
                ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
                ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),
                 [colors.HexColor('#f5f5ff'), colors.HexColor('#eeeeff')]),
                ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor('#aaaacc')),
                ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
                ('TOPPADDING',   (0,0), (-1,-1), 4),
                ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ])
            for r, (_, st) in enumerate(items, 1):
                c = (colors.HexColor('#1b5e20') if abs(st.mean) <= 5 else
                     colors.HexColor('#e65100') if abs(st.mean) <= 12 else
                     colors.HexColor('#b71c1c'))
                ts.add('TEXTCOLOR', (2,r), (2,r), c)
                ts.add('FONTNAME',  (2,r), (2,r), 'Helvetica-Bold')
            tbl.setStyle(ts)
            story.append(tbl)

        try:
            doc.build(story)
            QMessageBox.information(self, self._t('export_title'), self._t('pdf_saved', path=path))
        except Exception as e:
            QMessageBox.critical(self, self._t('err_title'), str(e))

    # ── Export CSV ────────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._log.measurements():
            QMessageBox.information(self, self._t('export_title'),
                                    self._t('csv_no_data'))
            return

        # Optionally tag the current run with maker/model so they show up in
        # the CSV alongside the measurements. In-memory only — no rewrite of
        # the on-disk JSONL record.
        model_info = self._ask_instrument_model()
        if model_info is None:
            return
        maker, model = model_info
        if maker or model:
            self._log.set_current_run_metadata(maker=maker, model=model)

        sel = self._ask_csv_slice()
        if sel is None:
            return

        if sel['mode'] == 'instrument_avg' and not sel['instrument']:
            QMessageBox.warning(self, self._t('err_title'),
                                self._t('csv_need_instr'))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, self._t('csv_save_title'),
            f"intonation_{sel['mode']}_{_today()}.csv",
            self._t('csv_filter'))
        if not path:
            return

        try:
            n = self._log.export_csv(path,
                                     mode=sel['mode'],
                                     run_id=sel['run_id'],
                                     instrument=sel['instrument'],
                                     nickname=sel['nickname'])
            QMessageBox.information(self, self._t('export_title'),
                                    self._t('csv_saved', rows=n, path=path))
        except Exception as e:
            QMessageBox.critical(self, self._t('err_title'), str(e))

    def _ask_csv_slice(self) -> dict | None:
        """Modal dialog: slice mode + (optional) run/instrument filter.

        Returns {'mode', 'run_id', 'instrument'} or None if cancelled.
        """
        dlg = QDialog(self)
        dlg.setProperty('_lang_title_key', 'csv_dialog_title')
        dlg.setWindowTitle(self._t('csv_dialog_title'))
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #ddd; }
            QLabel  { color: #bbb; font-size: 13px; }
            QComboBox {
                background: #252540; border: 1px solid #444; border-radius: 5px;
                color: #eee; padding: 6px 10px; font-size: 13px; min-width: 260px;
            }
            QComboBox QAbstractItemView { background: #252540; color: #eee; }
            QDialogButtonBox QPushButton {
                background: #2c5282; color: white; border: none;
                border-radius: 5px; padding: 6px 18px; font-size: 13px;
            }
            QDialogButtonBox QPushButton:hover  { background: #3a6da8; }
            QDialogButtonBox QPushButton:pressed { background: #1e3a5f; }
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 14, 20, 16)

        # Closing the dialog and re-opening it is the path to picking up a
        # language switch — the captured labels here aren't re-bound. The
        # toggle at least lets the user notice they're in the wrong language.
        info = QLabel(self._t('csv_dialog_info'))
        info.setStyleSheet('color: #888; font-size: 12px;')
        info.setWordWrap(True)

        def relabel_slice_dialog():
            info.setText(self._t('csv_dialog_info'))
            # Mode combo: keep current selection, swap labels.
            cur_mode = mode_combo.currentData()
            mode_combo.blockSignals(True)
            mode_combo.clear()
            for key in MeasurementLog.SLICE_MODES:
                mode_combo.addItem(self._t(f'csv_mode_{key}'), key)
                if key == cur_mode:
                    mode_combo.setCurrentIndex(mode_combo.count() - 1)
            mode_combo.blockSignals(False)
            mode_lbl.setText(self._t('csv_mode_label'))
            run_lbl.setText(self._t('csv_run_label'))
            instr_lbl.setText(self._t('csv_instr_label'))
            nick_lbl.setText(self._t('csv_nick_label'))
            # Re-translate the "All …" sentinels at index 0 if present.
            if run_combo.itemData(0) is None:
                run_combo.setItemText(0, self._t('csv_all_runs'))
            if instr_combo.count() and instr_combo.itemData(0) is None:
                instr_combo.setItemText(0, self._t('csv_all_instruments'))
            if nick_combo.itemData(0) is None:
                nick_combo.setItemText(0, self._t('csv_all_nicks'))
            summary.setText(self._t('csv_summary',
                                     n=len(self._log.measurements()),
                                     runs=len(self._log.runs())))

        self._add_dialog_lang_toggle(dlg, layout, on_change=relabel_slice_dialog)

        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        mode_combo = QComboBox()
        for key in MeasurementLog.SLICE_MODES:
            mode_combo.addItem(self._t(f'csv_mode_{key}'), key)
        mode_lbl = QLabel(self._t('csv_mode_label'))
        form.addRow(mode_lbl, mode_combo)

        run_combo = QComboBox()
        run_combo.addItem(self._t('csv_all_runs'), None)
        for run in self._log.runs():
            label = (f"{run.started_at} · {self._instr_label(run.instrument)}"
                     f" · A={run.a4_hz:.0f} Hz")
            tail = ' '.join(x for x in (run.maker, run.model) if x)
            if tail:
                label += f" · {tail}"
            run_combo.addItem(label, run.run_id)
        run_lbl = QLabel(self._t('csv_run_label'))
        form.addRow(run_lbl, run_combo)

        instr_combo = QComboBox()
        instr_combo.addItem(self._t('csv_all_instruments'), None)
        for key in self._log.instruments():
            instr_combo.addItem(self._instr_label(key), key)
        instr_lbl = QLabel(self._t('csv_instr_label'))
        form.addRow(instr_lbl, instr_combo)

        # Nickname filter — collect from runs that have one. If nothing in the
        # log has a nickname yet, the combo will only show "All nicknames".
        nick_combo = QComboBox()
        nick_combo.addItem(self._t('csv_all_nicks'), None)
        seen_nicks = sorted({r.label for r in self._log.runs() if r.label})
        for nk in seen_nicks:
            nick_combo.addItem(nk, nk)
        nick_lbl = QLabel(self._t('csv_nick_label'))
        form.addRow(nick_lbl, nick_combo)

        layout.addLayout(form)

        n = len(self._log.measurements())
        r = len(self._log.runs())
        summary = QLabel(self._t('csv_summary', n=n, runs=r))
        summary.setStyleSheet('color: #888; font-size: 11px;')
        layout.addWidget(summary)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        def refresh_filters():
            mode = mode_combo.currentData()
            run_enabled = mode in ('raw', 'per_run_note')
            instr_required = mode == 'instrument_avg'
            instr_enabled = mode in ('raw', 'per_run_note',
                                     'per_instrument_note', 'instrument_avg',
                                     'per_nickname_note')
            nick_enabled = mode in ('raw', 'per_run_note',
                                    'per_instrument_note', 'per_nickname_note')

            run_combo.setEnabled(run_enabled)
            run_lbl.setEnabled(run_enabled)
            instr_combo.setEnabled(instr_enabled)
            instr_lbl.setEnabled(instr_enabled)
            nick_combo.setEnabled(nick_enabled)
            nick_lbl.setEnabled(nick_enabled)

            # In instrument_avg mode the user must pick one instrument; the
            # "All instruments" sentinel at index 0 is removed and reinstated
            # when leaving that mode.
            first_is_all = instr_combo.itemData(0) is None
            if instr_required and first_is_all:
                instr_combo.removeItem(0)
            elif not instr_required and not first_is_all:
                instr_combo.insertItem(0, self._t('csv_all_instruments'),
                                        None)
                instr_combo.setCurrentIndex(0)

        mode_combo.currentIndexChanged.connect(refresh_filters)
        refresh_filters()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return {
            'mode': mode_combo.currentData(),
            'run_id': run_combo.currentData() if run_combo.isEnabled() else None,
            'instrument': (instr_combo.currentData()
                           if instr_combo.isEnabled() else None),
            'nickname': (nick_combo.currentData()
                          if nick_combo.isEnabled() else None),
        }

    def _instr_label(self, key: str) -> str:
        long_key = f'instr_long_{key}'
        if long_key in STRINGS[self.lang]:
            return self._t(long_key)
        return instrument_display_name(key, self.lang)

    # ── Import CSV ────────────────────────────────────────────────────────────
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self._t('csv_import_title'), '',
            self._t('csv_filter'))
        if not path:
            return
        try:
            runs, meas = self._log.import_raw_csv(path)
        except ValueError:
            QMessageBox.warning(self, self._t('err_title'),
                                self._t('csv_import_badhdr'))
            return
        except OSError as e:
            QMessageBox.critical(self, self._t('err_title'), str(e))
            return

        if runs == 0 and meas == 0:
            QMessageBox.information(self, self._t('csv_import_title'),
                                    self._t('csv_import_empty'))
            return
        QMessageBox.information(
            self, self._t('csv_import_title'),
            self._t('csv_import_saved', runs=runs, meas=meas))

    # ── Export Chart (PNG) ────────────────────────────────────────────────────
    def _export_chart(self):
        with self._lock:
            items = sorted(self.stats.items())

        if not items:
            QMessageBox.information(self, self._t('export_title'),
                                    self._t('chart_no_data'))
            return

        # Optional maker/model — same flow as TXT/PDF/CSV export.
        model_info = self._ask_instrument_model()
        if model_info is None:
            return
        maker, model = model_info

        path, _ = QFileDialog.getSaveFileName(
            self, self._t('chart_save_title'),
            f"intonation_chart_{self.instrument}_{_today()}.png",
            self._t('chart_filter'))
        if not path:
            return
        # Ensure the file ends in .png so QPixmap picks the right encoder.
        if not path.lower().endswith('.png'):
            path += '.png'

        transp = TRANSP_MAP.get(self.instrument, 0)
        disp_griff = (self.display == 'griff')
        sr_now = self._engine_sample_rate()
        a4 = self._engine.a4
        notes = []
        for midi_kl, st in items:
            midi_gr = midi_kl - transp
            display_name = (midi_note_name(midi_gr) if disp_griff
                            else midi_note_name(midi_kl))
            note_freq = a4 * (2.0 ** ((midi_kl - 69) / 12.0))
            notes.append((display_name, st.mean, st.std, st.n, note_freq))

        instr_long = self._t(f'instr_long_{self.instrument}')
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        if maker or model:
            subtitle = self._t('chart_subtitle_id',
                                instr=instr_long,
                                maker=maker, model=model,
                                a4=self._engine.a4, dt=dt)
        else:
            subtitle = self._t('chart_subtitle',
                                instr=instr_long,
                                a4=self._engine.a4, dt=dt)
        total = sum(s.n for _, s in items)
        footer = self._t('chart_footer', notes=len(items), total=total)

        try:
            render_intonation_chart(
                notes=notes,
                title=self._t('chart_title'),
                subtitle=subtitle,
                footer=footer,
                output_path=path,
                sample_rate=sr_now,
            )
            QMessageBox.information(self, self._t('export_title'),
                                    self._t('chart_saved', path=path))
        except Exception as e:
            QMessageBox.critical(self, self._t('err_title'), str(e))

    def _make_bar_ascii(self, cents, w=16):
        half = w // 2
        fill = int(min(1.0, abs(cents) / 40.0) * half)
        if cents > 1:  return ' '*half + '|' + '#'*fill + '.'*(half-fill)
        if cents < -1: return '.'*(half-fill) + '#'*fill + '|' + ' '*half
        return ' '*half + '|' + ' '*half

    def _restore_session_state(self) -> None:
        """Apply persisted window geometry, splitter sizes, nickname,
        language/instrument/display/A4 to the freshly-built widgets.

        Called from __init__ right after _build_ui. Lang / instrument /
        display were already applied earlier (they affect _build_ui's
        first paint); here we sync the combos to those values plus the
        widget-level state. First launch (empty cfg) is a no-op."""
        cfg = self._cfg

        # Window geometry — only restore if the saved rect still fits on
        # at least one connected screen. A user who unplugged a monitor
        # should not get a window stranded off-screen.
        geom_b64 = getattr(cfg, 'window_geometry', "")
        if geom_b64:
            try:
                ba = QByteArray.fromBase64(geom_b64.encode('ascii'))
                if not ba.isEmpty() and self.restoreGeometry(ba):
                    screens = QGuiApplication.screens()
                    if screens:
                        visible = False
                        win_geom = self.frameGeometry()
                        for scr in screens:
                            if scr.availableGeometry().intersects(win_geom):
                                visible = True
                                break
                        if not visible:
                            # Off-screen — let Qt's default placement
                            # take over by clearing the geometry.
                            self.resize(1100, 700)
                            primary = QGuiApplication.primaryScreen()
                            if primary is not None:
                                center = primary.availableGeometry().center()
                                self.move(center.x() - self.width() // 2,
                                          center.y() - self.height() // 2)
            except Exception:
                pass
        state_b64 = getattr(cfg, 'window_state', "")
        if state_b64:
            try:
                ba = QByteArray.fromBase64(state_b64.encode('ascii'))
                if not ba.isEmpty():
                    self.restoreState(ba)
            except Exception:
                pass

        # Splitter widths.
        sizes = list(getattr(cfg, 'splitter_sizes', []) or [])
        if (hasattr(self, '_splitter') and len(sizes) == 2
                and all(s >= 0 for s in sizes) and sum(sizes) > 0):
            self._splitter.setSizes(sizes)

        # Nickname text.
        nick = getattr(cfg, 'last_nickname', "")
        if nick and hasattr(self, '_nick_edit'):
            self._nick_edit.setText(nick)

        # Language combo — match self.lang set earlier in __init__.
        if hasattr(self, '_lang_combo'):
            for i in range(self._lang_combo.count()):
                if self._lang_combo.itemData(i) == self.lang:
                    self._lang_combo.blockSignals(True)
                    self._lang_combo.setCurrentIndex(i)
                    self._lang_combo.blockSignals(False)
                    break
            # _build_ui used the lang in effect at construction; if we
            # overrode it post-load, retranslate now to repaint labels.
            self._retranslate()

        # Display combo — match self.display.
        if hasattr(self, '_disp_combo'):
            target_idx = 0 if self.display == 'griff' else 1
            if self._disp_combo.currentIndex() != target_idx:
                self._disp_combo.blockSignals(True)
                self._disp_combo.setCurrentIndex(target_idx)
                self._disp_combo.blockSignals(False)

        # A4 — apply to combo and engine.
        a4 = int(getattr(cfg, 'last_a4_hz', 440))
        if 430 <= a4 <= 450:
            if hasattr(self, '_a4_combo'):
                self._a4_combo.blockSignals(True)
                self._a4_combo.setCurrentIndex(a4 - 430)
                self._a4_combo.blockSignals(False)
            try:
                self._engine.a4 = float(a4)
            except Exception:
                pass

    def closeEvent(self, ev):
        # v0.5.5: snapshot the full session state to the config file so the
        # next launch lands the user back where they were. Wrapped in
        # try/except because save-on-exit must never block the engine
        # from stopping cleanly — a corrupt config is recoverable, a
        # dangling audio stream is not.
        try:
            self._save_session_state()
        except Exception:
            pass
        self._engine.stop()
        ev.accept()

    def _save_session_state(self) -> None:
        """Assemble a full AppConfig snapshot from current widget state
        and persist. Called from closeEvent. The per-setting saves
        scattered through the GUI remain as a belt; this is the
        suspenders."""
        cfg = self._cfg
        # Window geometry + state — QByteArray base64 round-trip.
        geom = self.saveGeometry()
        state = self.saveState()
        try:
            cfg.window_geometry = bytes(geom.toBase64()).decode('ascii')
        except Exception:
            cfg.window_geometry = ""
        try:
            cfg.window_state = bytes(state.toBase64()).decode('ascii')
        except Exception:
            cfg.window_state = ""
        # Splitter sizes.
        if hasattr(self, '_splitter'):
            cfg.splitter_sizes = list(self._splitter.sizes())
        # Per-widget UI state.
        cfg.last_instrument_key = self.instrument
        if hasattr(self, '_nick_edit'):
            cfg.last_nickname = self._nick_edit.text().strip()
        cfg.last_display_mode = self.display
        try:
            cfg.last_a4_hz = int(self._engine.a4)
        except Exception:
            pass
        cfg.last_lang = self.lang
        sax_config.save_config(cfg)


def _today():
    return datetime.date.today().strftime('%Y%m%d')


# =============================================================================
def _load_app_icon() -> QIcon | None:
    """Locate the bundled application icon. Returns None if it can't be
    found — never crashes startup over a missing asset."""
    candidates = [
        Path(__file__).parent / 'assets' / 'icon.png',
        Path(__file__).parent / 'assets' / 'icon.ico',
    ]
    # PyInstaller drops resources into sys._MEIPASS at runtime.
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.insert(0, Path(meipass) / 'assets' / 'icon.png')
        candidates.insert(0, Path(meipass) / 'assets' / 'icon.ico')
    for p in candidates:
        if p.exists():
            icon = QIcon(str(p))
            if not icon.isNull():
                return icon
    return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    icon = _load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = MainWindow()
    if icon is not None:
        win.setWindowIcon(icon)
    win.show()
    sys.exit(app.exec())
