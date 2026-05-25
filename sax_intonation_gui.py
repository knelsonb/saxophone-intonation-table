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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF, QPointF, QLocale
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QIcon

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
APP_VERSION = '0.2.0'


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
        'nickname_tip':     'Spitzname (z.B. "Tenor #1")',
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
        'status_fmt':    'Gegriffen: {fingered}   Klingend: {sounding}   {freq:.1f} Hz   {sign}{cents:.1f} ct   (A={a4:.0f} Hz)',
        # Tabellen-Label
        'table_title':   'Intonationstabelle',
        'table_summary': 'Intonationstabelle  \u2013  {notes} Töne  |  {total} Messungen',
        'table_empty_hint': 'Intonationstabelle  \u2013  spiel einen Ton, dann erscheinen hier Mittelwert und Standardabweichung pro Ton.',
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
    },
    'en': {
        'window_title': 'Intonation Analyzer',
        'grp_instrument': 'Instrument',
        'grp_family':       'Family',
        'grp_subinstrument':'Model',
        'grp_nickname':     'Nickname',
        'nickname_tip':     'Nickname (e.g. "Tenor #1")',
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
        'status_fmt':    'Fingered: {fingered}   Sounding: {sounding}   {freq:.1f} Hz   {sign}{cents:.1f} ct   (A={a4:.0f} Hz)',
        'table_title':   'Intonation Table',
        'table_summary': 'Intonation Table  \u2013  {notes} notes  |  {total} measurements',
        'table_empty_hint': 'Intonation Table  \u2013  play a note to begin; per-note mean and standard deviation appear here as you play.',
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
    },
}


# =============================================================================
# Konstanten & Musik-Logik
# =============================================================================
SAMPLE_RATE   = 44100
HOP_SIZE      = 2048   # größerer Hop für tiefe Frequenzen (Bass-Sax ~29 Hz)
BLOCK_SIZE    = 16384  # ~372 ms – mindestens 2× tau_max bei 29 Hz (tau≈1521)
MIN_FREQ      = 27.0   # C1 – tiefstes Bass-Sax-Fundament mit Sicherheitspuffer
MAX_FREQ      = 1400.0
YIN_THRESHOLD = 0.12   # etwas strenger für sauberere Erkennung tiefer Töne
A4_DEFAULT    = 440.0

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
# YIN Pitch-Detektion
# =============================================================================
def yin_pitch(sig, sr=SAMPLE_RATE, fmin=MIN_FREQ, fmax=MAX_FREQ, thr=YIN_THRESHOLD):
    N = len(sig)
    tmin = max(1, int(sr / fmax))
    tmax = min(N // 2, int(sr / fmin))
    if tmax <= tmin:
        return 0.0, 1.0
    diff = np.array([np.dot(d := sig[:N-t] - sig[t:N], d) for t in range(tmax+1)])
    cmnd = np.ones(tmax + 1)
    run = 0.0
    for t in range(1, tmax + 1):
        run += diff[t]
        cmnd[t] = diff[t] * t / run if run > 0 else 1.0
    tau, mv = -1, 1.0
    for t in range(tmin, tmax):
        if cmnd[t] < thr:
            while t + 1 < tmax and cmnd[t+1] < cmnd[t]:
                t += 1
            tau, mv = t, cmnd[t]
            break
    if tau == -1:
        tau = tmin + int(np.argmin(cmnd[tmin:tmax]))
        mv  = cmnd[tau]
    if 1 < tau < tmax - 1:
        s0, s1, s2 = cmnd[tau-1], cmnd[tau], cmnd[tau+1]
        d = 2*s1 - s0 - s2
        if d:
            tau += 0.5 * (s0 - s2) / d
    return (sr / tau if tau > 0 else 0.0), mv


# =============================================================================
# Messdaten
# =============================================================================
class NoteStats:
    def __init__(self):
        self.vals: list[float] = []
    def add(self, c): self.vals.append(c)
    @property
    def mean(self): return float(np.mean(self.vals)) if self.vals else 0.0
    @property
    def std(self):  return float(np.std(self.vals))  if len(self.vals) > 1 else 0.0
    @property
    def n(self):    return len(self.vals)


# =============================================================================
# Audio-Engine
# =============================================================================
class AudioSignals(QObject):
    note_detected = pyqtSignal(int, float, float)

class AudioEngine:
    def __init__(self):
        self.signals    = AudioSignals()
        self._buf       = np.zeros(BLOCK_SIZE, dtype=np.float32)
        self._stream    = None
        self.a4         = A4_DEFAULT
        self.instr_key  = 'eb_alto'   # aktuell ausgewähltes Instrument

    def start(self, device=None):
        if not AUDIO_OK:
            return
        def cb(indata, frames, ti, st):
            mono = indata[:, 0]
            self._buf = np.roll(self._buf, -frames)
            self._buf[-frames:] = mono
            rms = math.sqrt(float(np.mean(self._buf**2)))
            if rms < 5e-5:   # etwas empfindlicher für tiefe Töne
                return
            sig = self._buf / (rms + 1e-9)
            freq, ap = yin_pitch(sig)
            if ap > YIN_THRESHOLD or not (MIN_FREQ < freq < MAX_FREQ):
                return
            mr, ct = cents_dev(freq, self.a4)
            if mr in SAX_MIDI:
                self.signals.note_detected.emit(int(mr), freq, ct)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=HOP_SIZE,
            channels=1, dtype='float32', callback=cb, device=device)
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


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
        t = QTimer(self)
        t.timeout.connect(self._fade)
        t.start(80)
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
            sign = '+' if self.cents >= 0 else ''
            p.setFont(QFont('Monospace', 32, QFont.Weight.Bold))
            if   abs(self.cents) <= 5:  cc = QColor(60,  220, 100, alpha)
            elif abs(self.cents) <= 15: cc = QColor(255, 200, 40,  alpha)
            else:                       cc = QColor(240, 70,  70,  alpha)
            p.setPen(cc)
            p.drawText(QRectF(0, sy + sh + 26, W, 55),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{sign}{self.cents:.1f} ct")
        p.end()


# =============================================================================
# Delegate: grafischer Intonationsbalken in der Tabelle
# =============================================================================
class CentBarDelegate(QStyledItemDelegate):
    """Zeichnet einen zentrierten, farbcodierten Balken für Cent-Abweichungen.
    Der Wert wird als float-String im UserRole gespeichert."""

    MAX_CENT = 50.0   # ±50 ct = volle Balkenhälfte

    def paint(self, painter, option, index):
        try:
            cents = float(index.data(Qt.ItemDataRole.UserRole))
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

        # Cent-Wert als Text rechts
        sign = '+' if cents >= 0 else ''
        txt = f"{sign}{cents:.1f} ct"
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
# Haupt-Fenster
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Pick the user's system language as the default. German-speaking
        # locales get the original DE strings; anyone else gets English.
        self.lang       = 'de' if QLocale.system().name().startswith('de') else 'en'
        self.instrument = 'eb_alto'   # default; saxophone family, alto
        self.display    = 'griff'
        self.stats: dict[int, NoteStats] = {}
        self._lock = threading.Lock()
        self._recording = True   # Aufnahme läuft beim Start
        self._active_midi: int | None = None
        self._active_midi_at: datetime.datetime | None = None

        # Load user config + previously-registered custom instruments before
        # building the UI so the catalog reflects them at first paint.
        self._cfg = sax_config.load_config()
        for c in sax_config.load_customs():
            register_custom(c.key, c.transp, c.name_de, c.name_en)
        _rebuild_transp_map()

        self._engine = AudioEngine()
        # Persistence comes from config (welcome dialog), with the env var
        # SAX_INTONATION_LOG_PATH as a power-user override that always wins.
        env_path = os.environ.get('SAX_INTONATION_LOG_PATH')
        log_path = env_path if env_path else self._cfg.effective_log_path()
        self._log = MeasurementLog(path=log_path or None)
        if AUDIO_OK:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=self._engine.a4)
            self._engine.signals.note_detected.connect(self._on_note)
            self._engine.start()

        self._build_ui()
        self._seed_expected_notes()
        self._update_record_btn_style()

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
        self._nick_edit.setMaximumWidth(160)
        self._nick_edit.editingFinished.connect(self._on_nickname_changed)
        self._nick_edit.setStyleSheet("""
            QLineEdit{background:#1e1e2e;border:1px solid #444;
                       border-radius:5px;color:#ddd;padding:4px 8px;font-size:12px;}
            QLineEdit:focus{border:1px solid #6699cc;}
        """)
        il.addWidget(self._nick_edit)

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

        # Inputs row: instrument config + import (open belongs near inputs).
        toolbar.addWidget(self._grp_instr)
        toolbar.addWidget(self._grp_disp)
        toolbar.addWidget(self._grp_a4)
        toolbar.addWidget(self._grp_lang)
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

        # Links: Tuner
        left = QWidget()
        ll3 = QVBoxLayout(left)
        ll3.setContentsMargins(0, 0, 6, 0)
        self._tuner = TunerWidget()
        self._tuner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ll3.addWidget(self._tuner)
        self._status_lbl = QLabel(self._t('no_signal'))
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet('color:#888;font-size:13px;padding:4px;')
        ll3.addWidget(self._status_lbl)

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
        self._bar_delegate = CentBarDelegate(self._table)
        self._table.setItemDelegateForColumn(5, self._bar_delegate)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 620])
        root.addWidget(splitter, 1)

        # Fensterstil
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#12121a;color:#ddd;}
            QGroupBox{border:1px solid #333;border-radius:6px;font-size:12px;
                      color:#aaa;margin-top:6px;padding-top:4px;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;top:-2px;}
            QComboBox{background:#1e1e2e;border:1px solid #444;border-radius:5px;
                      color:#ddd;padding:4px 8px;font-size:13px;min-height:28px;}
            QComboBox::drop-down{border:none;width:20px;}
            QComboBox QAbstractItemView{background:#1e1e2e;color:#ddd;}
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
        self._nick_edit.setPlaceholderText(self._t('nickname_tip'))

        # Display-Combo
        idx_d = self._disp_combo.currentIndex()
        self._disp_combo.blockSignals(True)
        self._disp_combo.clear()
        self._disp_combo.addItems([self._t('disp_griff'), self._t('disp_klingend')])
        self._disp_combo.setCurrentIndex(idx_d)
        self._disp_combo.blockSignals(False)

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
        self._refresh_table()

    # ── Audio-Callback ────────────────────────────────────────────────────────
    def _on_note(self, midi_kl: int, freq: float, cents: float):
        if not self._recording:
            return
        with self._lock:
            if midi_kl not in self.stats:
                self.stats[midi_kl] = NoteStats()
            self.stats[midi_kl].add(cents)
        # Per-measurement log. Instrument/A4 are read off the active run
        # inside the log, not from `self`, so a callback firing during a UI
        # change still attributes to the run that was active when it fired.
        midi_gr = midi_kl - TRANSP_MAP[self.instrument]
        self._log.add_measurement(midi_sounding=midi_kl,
                                   midi_fingered=midi_gr,
                                   cents=cents, freq_hz=freq)

        transp     = TRANSP_MAP[self.instrument]
        midi_gr    = midi_kl - transp
        kl_name    = midi_note_name(midi_kl)
        gr_name    = midi_note_name(midi_gr)
        disp_name  = gr_name if self.display == 'griff' else kl_name
        sign       = '+' if cents >= 0 else ''
        self._tuner.set_note(disp_name, freq, cents)
        self._status_lbl.setText(self._t(
            'status_fmt', fingered=gr_name, sounding=kl_name,
            freq=freq, sign=sign, cents=cents, a4=self._engine.a4))
        # Highlight the row currently being played so the user can see
        # which entry in a long table just ticked. _refresh_table reads
        # this on its next tick (every 300ms via _refresh_timer).
        self._active_midi = midi_kl
        self._active_midi_at = datetime.datetime.now()

    # ── Tabelle ───────────────────────────────────────────────────────────────
    def _refresh_table(self):
        # Can be called during initial UI construction (instrument combo
        # populate fires _on_instr_changed before _build_ui finishes the
        # right-hand panel). Bail out until the table widget exists.
        if not hasattr(self, '_table'):
            return
        transp     = TRANSP_MAP[self.instrument]
        disp_griff = (self.display == 'griff')

        if disp_griff:
            hdrs = [self._t('col_fingered'), self._t('col_sounding')]
        else:
            hdrs = [self._t('col_sounding'), self._t('col_fingered')]
        self._table.setHorizontalHeaderLabels(
            hdrs + [self._t('col_mean'), self._t('col_std'),
                    self._t('col_n'), self._t('col_tendency')])

        with self._lock:
            items = sorted(self.stats.items())

        self._table.setRowCount(len(items))
        played_n = 0
        # Decide whether the highlight is still fresh — fade it after 1.5s
        # of silence so the table reverts to neutral when the player stops.
        now = datetime.datetime.now()
        active_midi = self._active_midi
        if (self._active_midi_at is None
                or (now - self._active_midi_at).total_seconds() > 1.5):
            active_midi = None
        for row, (midi_kl, st) in enumerate(items):
            midi_gr = midi_kl - transp
            kl_name = midi_note_name(midi_kl)
            gr_name = midi_note_name(midi_gr)
            n1, n2  = (gr_name, kl_name) if disp_griff else (kl_name, gr_name)
            mean    = st.mean
            has_data = st.n > 0
            if has_data:
                played_n += 1
            sign    = '+' if mean >= 0 else ''

            col = (QColor('#3a9e5f') if abs(mean) <= 5 else
                   QColor('#c8a020') if abs(mean) <= 12 else QColor('#c03030'))
            dim_col = QColor('#555')   # un-played pre-seeded notes

            is_active = (active_midi == midi_kl)
            for c, val in enumerate([
                n1, n2,
                f"{sign}{mean:.1f}" if has_data else '\u2013',
                f"\u00b1{st.std:.1f}" if st.n > 1 else '\u2013',
                str(st.n) if has_data else '\u2013',
                '',   # Balken wird vom Delegate gezeichnet
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 2:
                    item.setForeground(col if has_data else dim_col)
                elif not has_data:
                    item.setForeground(dim_col)
                if c == 5 and has_data:
                    item.setData(Qt.ItemDataRole.UserRole, mean)
                if is_active:
                    item.setBackground(QColor('#2c5a8a'))
                self._table.setItem(row, c, item)

        total = sum(s.n for _, s in items)
        if not items:
            label = self._t('table_empty_hint')
        elif played_n == 0:
            # Only seeded blanks visible \u2014 keep the play-a-note hint up.
            label = self._t('table_empty_hint')
        else:
            label = self._t('table_summary', notes=played_n, total=total)
        self._table_lbl.setText(label)

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
        self._engine.a4 = float(self._a4_combo.itemData(idx))
        with self._lock:
            self.stats.clear()
        # A4 changes invalidate cent values ⇒ start a new run. Scrubbing the
        # combo to find the right value does NOT pile up empty runs because
        # `start_run` coalesces the predecessor if it never recorded.
        if AUDIO_OK and self._recording:
            self._log.start_run(instrument=self.instrument,
                                a4_hz=self._engine.a4,
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

        # klingende MIDI-Note aus den gespeicherten Stats ermitteln
        with self._lock:
            keys = sorted(self.stats.keys())
        if row >= len(keys):
            return
        midi_kl = keys[row]

        transp   = TRANSP_MAP[self.instrument]
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
        transp = TRANSP_MAP[self.instrument]
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
        for midi_kl, st in items:
            sign = '+' if st.mean >= 0 else ''
            lines.append(
                f"{midi_note_name(midi_kl - transp):<12} {midi_note_name(midi_kl):<12}"
                f" {sign}{st.mean:>6.1f}   {st.std:>6.1f}  {st.n:>5}  {self._make_bar(st.mean, 24)}")
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

        transp = TRANSP_MAP[self.instrument]
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
            for midi_kl, st in items:
                sign = '+' if st.mean >= 0 else ''
                data.append([
                    midi_note_name(midi_kl - transp), midi_note_name(midi_kl),
                    f"{sign}{st.mean:.1f}",
                    f"\u00b1{st.std:.1f}" if st.n > 1 else '\u2013',
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

        transp = TRANSP_MAP[self.instrument]
        disp_griff = (self.display == 'griff')
        notes = []
        for midi_kl, st in items:
            midi_gr = midi_kl - transp
            display_name = (midi_note_name(midi_gr) if disp_griff
                            else midi_note_name(midi_kl))
            notes.append((display_name, st.mean, st.std, st.n))

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

    def closeEvent(self, ev):
        self._engine.stop()
        ev.accept()


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
