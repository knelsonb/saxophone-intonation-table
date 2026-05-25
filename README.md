# 🎷 Saxophon-Intonationsanalysator · Saxophone Intonation Analyzer

---

## 🇬🇧 English

### Description

This program records notes via microphone, detects the pitch in real time, and measures intonation deviation from equal temperament in cents. The longer you play, the more accurate the averages become. Supports Eb, Bb, and C instruments with automatic transposition logic.

**Features:**
- Large graphical tuner (readable from a distance)
- Live intonation table with mean and standard deviation per note
- Saxophone type selection (Eb / Bb / C) and note display (fingered / sounding)
- Adjustable concert pitch A (430–450 Hz)
- Automatic detection of optimal concert pitch from measurement data
- Export as TXT, PDF and CSV (including manufacturer and model)
- CSV export with multiple slice modes: raw measurements, per run, per instrument, one instrument averaged across all runs, or overall per-note mean
- Import a previously-exported raw CSV to resume, compare, or view someone else's session
- Chart export (PNG): one-click shareable bar chart of mean cents per note with ±1σ whiskers
- Optional cross-session persistence: set the environment variable `SAX_INTONATION_LOG_PATH` to a file path and every measurement is appended there as JSONL
- Interface switchable between German and English

---

### Requirements

| Component | Version |
|---|---|
| Python | 3.10 or newer |
| Operating System | Ubuntu 22.04+ / Windows 10+ |
| Microphone | Any input device |

---

### Installation on Linux (Ubuntu / Debian)

#### 1. Install system packages

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev portaudio19-dev
```

#### 2. Create a virtual environment

```bash
python3 -m venv ~/sax-venv
```

#### 3. Activate the virtual environment

```bash
source ~/sax-venv/bin/activate
```

> ⚠️ This command must be repeated **every time you open a new terminal**.  
> The prompt will show `(sax-venv)` at the beginning when active.

#### 4. Install Python packages

```bash
pip install PyQt6 numpy sounddevice reportlab
```

#### 5. Run the program

```bash
python3 sax_intonation_gui.py
```

#### Shortcut (activate + run in one command)

```bash
source ~/sax-venv/bin/activate && python3 sax_intonation_gui.py
```

---

### Installation on Windows

#### 1. Install Python

1. Download Python from [python.org/downloads](https://www.python.org/downloads/) (version 3.10 or newer)
2. Run the installer
3. ✅ Check **"Add Python to PATH"** — this is important!
4. Complete the installation

#### 2. Open Command Prompt

`Win + R` → type `cmd` → press Enter  
or: Start Menu → search "Command Prompt"

#### 3. Create a virtual environment

```cmd
python -m venv %USERPROFILE%\sax-venv
```

#### 4. Activate the virtual environment

```cmd
%USERPROFILE%\sax-venv\Scripts\activate
```

> ⚠️ This command must be repeated **every time you open a new Command Prompt**.  
> The prompt will show `(sax-venv)` at the beginning when active.

#### 5. Install Python packages

```cmd
pip install PyQt6 numpy sounddevice reportlab
```

#### 6. Run the program

```cmd
python sax_intonation_gui.py
```

> 💡 **Tip:** Copy `sax_intonation_gui.py` into the folder shown when the Command Prompt opens (usually `C:\Users\<Name>`), or navigate to the correct folder using `cd`:
> ```cmd
> cd C:\Path\To\Folder
> python sax_intonation_gui.py
> ```

---

### Troubleshooting (Linux & Windows)

| Problem | Solution |
|---|---|
| `externally-managed-environment` error | Use a virtual environment (see above) |
| `sounddevice` fails to install | On Linux: run `sudo apt install portaudio19-dev` first |
| Wrong microphone is used | List audio devices (see below) |
| No notes detected | Check microphone volume; play steadily at medium volume |
| GUI does not start (Linux) | Run `sudo apt install python3-dev` and reinstall PyQt6 |

#### List audio devices

```bash
# Linux
source ~/sax-venv/bin/activate
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

```cmd
:: Windows
%USERPROFILE%\sax-venv\Scripts\activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

### Usage notes

- **Changing concert pitch:** Changing the concert pitch A automatically resets all measurements, as cent deviations need to be recalculated.
- **Detect optimal pitch:** Play at least 3 notes with ≥ 5 measurements each, then press "Detect Concert Pitch". The program calculates the optimal concert pitch using a weighted median.
- **Export:** When exporting, you will be prompted for the instrument manufacturer and model (both optional). The values are remembered for the current session.
- **CSV export:** The "Export CSV" button opens a dialog with five slice modes:
  - *Raw* — one row per measurement
  - *Per run and note* — aggregated by (run, note): mean, std, min, max, N
  - *Per instrument and note* — aggregated across all runs of an instrument
  - *One instrument, per-note average* — requires picking one instrument
  - *Overall mean per note* — aggregated across everything
  Run and instrument filters are enabled depending on the mode. A "run" begins on app start, on instrument or concert-pitch change, and when recording is resumed after a pause.
- **Cross-session persistence (optional):** If the environment variable `SAX_INTONATION_LOG_PATH` points to a file, every measurement is also appended as JSON Lines. On the next launch the history is reloaded so the CSV slice modes can summarise across sessions.
- **Accuracy:** From approximately N = 20 measurements per note, averages become very reliable. Playing steadily at medium volume yields the best results.

---

### Package overview

| Package | Purpose |
|---|---|
| `PyQt6` | Graphical user interface |
| `numpy` | Signal processing and statistics |
| `sounddevice` | Microphone access via PortAudio |
| `reportlab` | PDF export |

---

*Developed for saxophone intonation analysis. Compatible with alto, baritone, tenor, soprano, bass and C instruments.*

---

## 🇩🇪 Deutsch

### Beschreibung

Dieses Programm nimmt Töne über das Mikrofon auf, erkennt die Tonhöhe in Echtzeit und misst die Intonationsabweichung vom temperierten Ideal in Cent. Je länger gespielt wird, desto genauer werden die Durchschnittswerte. Unterstützt Eb-, Bb- und C-Instrumente mit automatischer Transpositionslogik.

**Funktionen:**
- Großer grafischer Tuner (auch aus größerer Entfernung ablesbar)
- Live-Intonationstabelle mit Durchschnitt und Standardabweichung pro Ton
- Wahl des Saxophontyps (Eb / Bb / C) und der Tondarstellung (gegriffen / klingend)
- Einstellbarer Kammerton A (430–450 Hz)
- Automatische Ermittlung des optimalen Kammertons aus den Messdaten
- Export als TXT, PDF und CSV (mit Hersteller- und Modellangabe)
- CSV-Export mit verschiedenen Aufteilungen: Rohdaten, pro Lauf, pro Instrument, ein Instrument über alle Läufe gemittelt, oder Gesamtmittel je Ton
- CSV-Import: eine zuvor exportierte Rohdaten-CSV wieder einlesen, um eine Sitzung fortzusetzen, zu vergleichen oder die Daten von jemand anderem anzusehen
- Diagramm-Export (PNG): ein-Klick-Balkendiagramm der mittleren Cent-Abweichung pro Ton mit ±1σ-Whiskers, fertig zum Teilen
- Optionale sitzungsübergreifende Persistenz: Umgebungsvariable `SAX_INTONATION_LOG_PATH` auf einen Pfad setzen, dann werden alle Messungen zusätzlich als JSONL angehängt
- Oberfläche auf Deutsch und Englisch umschaltbar

---

### Voraussetzungen

| Komponente | Version |
|---|---|
| Python | 3.10 oder neuer |
| Betriebssystem | Ubuntu 22.04+ / Windows 10+ |
| Mikrofon | Beliebiges Eingangsgerät |

---

### Installation unter Linux (Ubuntu / Debian)

#### 1. Systempakete installieren

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev portaudio19-dev
```

#### 2. Virtuelle Umgebung anlegen

```bash
python3 -m venv ~/sax-venv
```

#### 3. Virtuelle Umgebung aktivieren

```bash
source ~/sax-venv/bin/activate
```

> ⚠️ Dieser Befehl muss bei **jeder neuen Terminal-Sitzung** wiederholt werden.  
> Das Prompt zeigt dann `(sax-venv)` am Anfang.

#### 4. Python-Pakete installieren

```bash
pip install PyQt6 numpy sounddevice reportlab
```

#### 5. Programm starten

```bash
python3 sax_intonation_gui.py
```

#### Kurzform (Aktivieren + Starten in einem Befehl)

```bash
source ~/sax-venv/bin/activate && python3 sax_intonation_gui.py
```

---

### Installation unter Windows

#### 1. Python installieren

1. Python von [python.org/downloads](https://www.python.org/downloads/) herunterladen (Version 3.10 oder neuer)
2. Installer ausführen
3. ✅ **„Add Python to PATH"** aktivieren — wichtig!
4. Installation abschließen

#### 2. Eingabeaufforderung öffnen

`Win + R` → `cmd` → Enter  
oder: Startmenü → „Eingabeaufforderung"

#### 3. Virtuelle Umgebung anlegen

```cmd
python -m venv %USERPROFILE%\sax-venv
```

#### 4. Virtuelle Umgebung aktivieren

```cmd
%USERPROFILE%\sax-venv\Scripts\activate
```

> ⚠️ Dieser Befehl muss bei **jeder neuen Eingabeaufforderung** wiederholt werden.  
> Das Prompt zeigt dann `(sax-venv)` am Anfang.

#### 5. Python-Pakete installieren

```cmd
pip install PyQt6 numpy sounddevice reportlab
```

#### 6. Programm starten

```cmd
python sax_intonation_gui.py
```

> 💡 **Tipp:** Die Datei `sax_intonation_gui.py` in den Ordner kopieren, der beim Start der Eingabeaufforderung als aktuelles Verzeichnis angezeigt wird (meist `C:\Users\<Name>`), oder per `cd` dorthin navigieren:
> ```cmd
> cd C:\Pfad\zum\Ordner
> python sax_intonation_gui.py
> ```

---

### Fehlerbehebung (Linux & Windows)

| Problem | Lösung |
|---|---|
| `externally-managed-environment` | Virtuelle Umgebung verwenden (siehe oben) |
| `sounddevice` lässt sich nicht installieren | Unter Linux: `sudo apt install portaudio19-dev` nachholen |
| Falsches Mikrofon wird verwendet | Gerätliste anzeigen (siehe unten) |
| Kein Ton erkannt | Lautstärke prüfen, mittelstark und gleichmäßig spielen |
| GUI startet nicht (Linux) | `sudo apt install python3-dev` und PyQt6 neu installieren |

#### Audiogeräte auflisten (Terminalversion)

```bash
# Linux
source ~/sax-venv/bin/activate
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

```cmd
:: Windows
%USERPROFILE%\sax-venv\Scripts\activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

### Bedienungshinweise

- **Kammerton ändern:** Beim Ändern des Kammertons werden alle Messungen automatisch zurückgesetzt, da die Centabweichungen neu berechnet werden.
- **Kammerton ermitteln:** Mindestens 3 Töne mit je ≥ 5 Messungen spielen, dann den Button „Kammerton ermitteln" drücken. Das Programm berechnet den optimalen Kammerton per gewichtetem Median.
- **Export:** Beim Export wird nach Hersteller und Modell des Instruments gefragt (optional). Die Eingaben werden für die Sitzung gespeichert.
- **CSV-Export:** Der Button „Export CSV" öffnet einen Dialog mit fünf Aufteilungsmodi:
  - *Rohdaten* — eine Zeile pro Messung
  - *Pro Lauf und Ton* — pro (Lauf, Ton) aggregiert (Mittel, Standardabw., Min, Max, N)
  - *Pro Instrument und Ton* — über alle Läufe eines Instruments aggregiert
  - *Ein Instrument, je Ton gemittelt* — verlangt die Wahl eines Instruments
  - *Gesamtmittel je Ton* — über alles aggregiert
  Filter für Lauf und Instrument werden je nach Modus aktiviert. Ein „Lauf" beginnt beim Programmstart, bei Instrument- oder Kammertonwechsel sowie nach dem Wiederaufnehmen einer pausierten Aufnahme.
- **Sitzungsübergreifende Persistenz (optional):** Wenn die Umgebungsvariable `SAX_INTONATION_LOG_PATH` auf eine Datei zeigt, werden alle Messungen zusätzlich als JSONL angehängt. Beim nächsten Start sind sie wieder verfügbar und können über die CSV-Aufteilungen ausgewertet werden.
- **Genauigkeit:** Ab ca. N = 20 Messungen pro Ton sind die Durchschnittswerte sehr zuverlässig. Am besten gleichmäßig und mittelstark spielen.

---
