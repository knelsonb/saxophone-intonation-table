# 🎷 Saxophon-Intonationsanalysator · Saxophone Intonation Analyzer

---

## 🇬🇧 English

### Description

This program listens through your microphone, detects the pitch of each note you play, and measures its deviation from equal temperament in cents. It keeps a mean and standard deviation for every note, so a practice session builds into an intonation chart of your instrument. It knows 73 instruments in 12 families with their transpositions, and you can add your own.

**Features:**
- Large tuner display, readable from a distance; cents are shown as whole numbers
- Live intonation table with mean, standard deviation and reading count (N) per note, as a list or as an octave grid
- 73 instruments in 12 families: saxophones, clarinets, flutes, trumpets, horns, trombones, low brass, double reeds, recorders, strings, plucked instruments, and voice / concert pitch
- Custom instruments with your own name and transposition; a nickname per horn; a range editor per instrument
- Note display as fingered or sounding pitch
- Concert pitch A from 430 to 450 Hz, and detection of the best-fitting concert pitch from your measurements
- Three response modes (Fast, Normal, Slow) — see [`docs/response-modes.md`](docs/response-modes.md)
- Export as TXT, PDF, PNG chart and CSV (with optional manufacturer and model)
- CSV export with six slice modes and run, instrument and nickname filters
- CSV import of a raw export, to combine or compare sessions
- Optional measurement history saved between sessions
- Interface in German or English, switchable at any time
- Four tabs: TUNER · METRO · DECK · SETUP
- Metronome: 30–300 BPM, accented downbeat, 2/4, 3/4, 4/4 and 6/8, tap tempo, sample-accurate click timing
- Drone reference tone: 5 quick presets plus all 128 General MIDI voices from the bundled SoundFont, and a vibrato-free pure-sine voice for precise tuning
- Pitch pipes: sustained sine reference tones from C4 to B4
- Tape deck: record a take through the microphone, play it back, export it as WAV
- Themes: Dark, Night (red-shifted to protect dark-adapted eyes in a dark hall or pit) and Light, switchable live
- Input device and sample-rate picker (AUDIO IN), output device selection, and mic gain; the tuner keeps reading the microphone while a reference tone sounds

---

### Download

Each [release](https://github.com/knelsonb/saxophone-intonation-table/releases) provides ready-to-run builds. Python is only needed if you run from source.

| File | System |
|---|---|
| `intonation-analyzer-windows-x86_64.exe` | Windows 10 / 11, 64-bit |
| `intonation-analyzer-linux-x86_64.tar.gz` | Linux x86-64 with glibc 2.35 or newer (built on Ubuntu 22.04) |
| `intonation-analyzer-macos-arm64.tar.gz` | macOS on Apple silicon (M1 and later) |
| `SHA256SUMS.txt` | SHA-256 checksums of the files above |

**Check the download** against the matching line in `SHA256SUMS.txt`:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing          # Linux
shasum -a 256 intonation-analyzer-macos-arm64.tar.gz  # macOS: compare by eye
```

```cmd
certutil -hashfile intonation-analyzer-windows-x86_64.exe SHA256
```

**Windows:** run the `.exe`. The file is unsigned, so Windows may show "Windows protected your PC"; click **More info**, then **Run anyway**.

**Linux:**

```bash
mkdir intonation-analyzer-linux
tar -xzf intonation-analyzer-linux-x86_64.tar.gz -C intonation-analyzer-linux
./intonation-analyzer-linux/intonation-analyzer
```

**macOS:** the binary is unsigned, so Gatekeeper blocks the first start.

```bash
mkdir intonation-analyzer-macos
tar -xzf intonation-analyzer-macos-arm64.tar.gz -C intonation-analyzer-macos
./intonation-analyzer-macos/intonation-analyzer
```

When macOS refuses to open it, right-click (Control-click) the file in Finder and choose **Open**, then confirm. On macOS 15 and later that shortcut is gone: try to open the file once, then go to **System Settings › Privacy & Security** and click **Open Anyway**. macOS then asks once for microphone access; allow it.

---

### Requirements (running from source)

| Component | Version |
|---|---|
| Python | 3.10 to 3.12 (release builds use 3.12). On 3.13 and newer the drone's SoundFont voices need a C compiler to install `tinysoundfont`, because it has no prebuilt package for those versions yet. |
| Operating system | Windows 10+, macOS, Linux (for example Ubuntu 22.04+) |
| Microphone | Any input device |

---

### Installation on Linux (Ubuntu / Debian)

#### 1. Install system packages

```bash
sudo apt update
sudo apt install git python3 python3-venv libportaudio2 libxcb-cursor0
```

`libportaudio2` is the audio library `sounddevice` uses; `libxcb-cursor0` is required by the Qt window system plugin. PyQt6, numpy and the other Python packages come as prebuilt wheels, so nothing needs compiling.

#### 2. Get the code

```bash
git clone https://github.com/knelsonb/saxophone-intonation-table.git
cd saxophone-intonation-table
```

#### 3. Create a virtual environment

```bash
python3 -m venv ~/sax-venv
```

#### 4. Activate the virtual environment

```bash
source ~/sax-venv/bin/activate
```

> ⚠️ This command must be repeated **every time you open a new terminal**.  
> The prompt will show `(sax-venv)` at the beginning when active.

#### 5. Install Python packages

```bash
pip install -r requirements.txt
pip install --no-deps "tinysoundfont~=0.3.7"
```

The second command installs the SoundFont synthesizer for the drone voices. `--no-deps` skips its `pyaudio` dependency, which this program does not use. Without `tinysoundfont` the program still runs, and the drone uses the pure-sine voice.

#### 6. Run the program

```bash
python sax_intonation_gui.py
```

#### Shortcut (activate + run in one command)

This assumes you cloned into your home folder:

```bash
cd ~/saxophone-intonation-table && source ~/sax-venv/bin/activate && python sax_intonation_gui.py
```

---

### Installation on Windows

#### 1. Install Python

1. Download Python 3.12 from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. ✅ Check **"Add Python to PATH"** — this is important!
4. Complete the installation

#### 2. Get the code

Either install [Git for Windows](https://git-scm.com/download/win) and run, in a Command Prompt:

```cmd
git clone https://github.com/knelsonb/saxophone-intonation-table.git
```

or, on the GitHub page, choose **Code › Download ZIP** and extract the ZIP. The program consists of `sax_intonation_gui.py` plus about twenty `sax_*.py` modules and the `assets` folder; keep them together in one folder.

#### 3. Open Command Prompt in the program folder

`Win + R` → type `cmd` → press Enter, then change into the folder:

```cmd
cd C:\Path\To\saxophone-intonation-table
```

#### 4. Create a virtual environment

```cmd
python -m venv %USERPROFILE%\sax-venv
```

#### 5. Activate the virtual environment

```cmd
%USERPROFILE%\sax-venv\Scripts\activate
```

> ⚠️ This command must be repeated **every time you open a new Command Prompt**.  
> The prompt will show `(sax-venv)` at the beginning when active.

#### 6. Install Python packages

```cmd
pip install -r requirements.txt
pip install --no-deps "tinysoundfont~=0.3.7"
```

#### 7. Run the program

```cmd
python sax_intonation_gui.py
```

---

### Installation on macOS

1. Install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/) and the command line tools (they include `git`): `xcode-select --install`
2. In Terminal:

```bash
git clone https://github.com/knelsonb/saxophone-intonation-table.git
cd saxophone-intonation-table
python3 -m venv ~/sax-venv
source ~/sax-venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "tinysoundfont~=0.3.7"
python sax_intonation_gui.py
```

On the first start macOS asks whether Terminal may use the microphone; allow it. On Intel Macs `tinysoundfont` has no prebuilt package and is compiled with the command line tools during the install.

---

### Troubleshooting

| Problem | Solution |
|---|---|
| `externally-managed-environment` error | Use a virtual environment (see above) |
| `OSError: PortAudio library not found` (Linux) | `sudo apt install libportaudio2` |
| Window does not open on Linux; the terminal mentions the `xcb` platform plugin | `sudo apt install libxcb-cursor0`. If the message names another missing library, install it too (for example `libxkbcommon-x11-0`). |
| Wrong microphone is used | Click **AUDIO IN** in the toolbar and choose the device. The same dialog sets the sample rate; **Auto** tries 48, 44.1, 96, 88.2 and 192 kHz in that order and uses the first rate the device accepts, falling back to the device's own default rate. |
| No notes detected | Check that **AUDIO IN** shows your microphone. Raise **Mic gain** in **SETUP**; it lowers the level at which playing counts as a note and leaves the cents readings unchanged. Play steadily at medium volume. |
| Notes are detected but the table stays empty | The table hides notes with fewer readings than **Min measurements per note** (default 5). Hold the note a moment longer, or lower the value. |
| The tuner ignores the note the drone or a pitch pipe is sounding | Intended; see "Drone, pitch pipes and the tuner" below. |
| The drone shows a message that `tinysoundfont` is not installed | Run `pip install --no-deps "tinysoundfont~=0.3.7"` in the virtual environment and restart. Until then the drone uses the pure-sine voice. |
| macOS: no input at all | Allow microphone access in **System Settings › Privacy & Security › Microphone** for Terminal (or for the program). |

#### List audio devices from the command line

```bash
# Linux / macOS
source ~/sax-venv/bin/activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

```cmd
:: Windows
%USERPROFILE%\sax-venv\Scripts\activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

### Usage notes

- **Layout:** the toolbar above the tabs holds the instrument, note display, concert pitch, response mode, language, **AUDIO IN**, CSV import, **Detect Concert Pitch**, recording, **Reset** and the export buttons. **TUNER** shows the tuner, the intonation table, the drone and the pitch pipes. **METRO** is the metronome, **DECK** the tape deck. **SETUP** holds the audio output, drone voice, theme, nickname and range editor, concert pitch, response mode, mic gain and a test tone.
- **Instruments:** pick a family, then the instrument. **+ Custom …** adds your own instrument with a display name and a transposition in semitones from fingered to sounding pitch, octave included (Bb soprano −2, Eb alto −9, Bb tenor −14). The nickname field tags your measurements (for example "Tenor #1") so the CSV export can compare two horns of the same type. **Edit instrument range …** in SETUP (or the gear button) sets which notes the table shows in advance; notes outside the range are still recorded unless **Filter to instrument range** is on.
- **Changing concert pitch:** the table keeps its notes. Their cents are recalculated from the frequencies already recorded, against the new A. A reading that ends up more than 50 cents from its note moves to the neighbouring note. Only the notes currently in the table take part; earlier sessions, imports and notes you reset or deleted stay out. New measurements after the change belong to a new run.
- **Detect Concert Pitch:** play at least 3 notes with at least 5 readings each, then press **Detect Concert Pitch**. The program takes each note's mean deviation, forms their median weighted by √N, and proposes the matching concert pitch, rounded to a whole Hz within 430–450 Hz. When you accept, it sets the new A and clears the table, so you start fresh at the new pitch.
- **Reset** clears the table, and the next export asks for manufacturer and model again. Right-clicking a note offers **Delete measurements for this note**. Both act on the table only; the measurement log behind the CSV export keeps every reading.
- **Colours:** green within ±5 cents, yellow within ±12 cents, red beyond.
- **Export TXT / PDF / PNG chart:** these list the notes you played in the current table; empty rows from the instrument range are left out. The first export asks for manufacturer and model (both optional). The answer is reused for every later export until you press **Reset**; the next prompt is pre-filled with it.
- **CSV export:** **Export CSV** writes from the measurement log — everything recorded since the program started, plus saved history and imported CSVs. It opens a dialog with six slice modes:
  - *Raw* — one row per measurement
  - *Per run and note* — aggregated by (run, note): mean, std, min, max, N
  - *Per instrument and note* — aggregated across all runs of an instrument
  - *Per nickname and note* — aggregated across runs sharing a nickname, for example to compare two tenors
  - *One instrument, per-note average* — requires picking one instrument
  - *Overall mean per note* — aggregated across everything

  The **Run** filter applies to *Raw* and *Per run and note*. The **Instrument** filter applies to every mode except *Overall mean per note*. The **Nickname** filter applies to *Raw*, *Per run and note*, *Per instrument and note* and *Per nickname and note*, and keeps only runs recorded under that nickname. A "run" begins on program start, on a change of instrument or concert pitch, and when recording resumes after a pause.
- **CSV import:** **Import CSV** accepts files written by the *Raw* mode (up to 50 MB or 500,000 rows). After a confirmation it adds the runs and measurements to the measurement log, skipping runs that are already there. Imported data appears in the CSV export; the live table stays as it is.
- **Measurement history:** the welcome dialog on first launch offers **Save measurement history to disk**. With it on, every measurement is appended to `~/.intonation_analyzer/log.jsonl` (on Windows `C:\Users\<Name>\.intonation_analyzer\log.jsonl`) and loaded again on the next start, so the CSV export can cover several sessions. The table always starts empty. **Decide later** brings the dialog back on the next start. To switch history on after confirming the dialog without it, close the program and set `"persistence_enabled": true` in `config.json` (see below). The environment variable `SAX_INTONATION_LOG_PATH` overrides both: when it names a file, measurements are saved there whatever the setting says.
- **Settings folder:** `~/.intonation_analyzer/` holds `config.json` (preferences and window layout), `custom_instruments.json` (your custom instruments), `instrument_ranges.json` (ranges you edited) and, with history on, `log.jsonl`. Deleting the folder resets the program to its first-launch state and deletes the saved history.
- **Drone, pitch pipes and the tuner:** while the drone or a pitch pipe sounds, the tuner ignores readings at the reference note itself: through the microphone, your note at that pitch and the speaker's own sound are indistinguishable. Other notes are measured normally. To measure the note the reference is playing, switch the reference off, or set the drone to a different note (the semitone buttons) and tune by interval against it. When the microphone keeps hearing the drone's note, the drone turns itself down, and back up once that stops.
- **Accuracy:** while you hold a steady note, the tuner adds one reading about every 46 ms, roughly 20 per second. The first reading of a note arrives after about 0.2 s (Fast), 0.3 s (Normal) or 0.65 s (Slow). About one second of steady playing per note gives a stable mean; playing each note several times with fresh attacks during a session shows how you actually play it. Steady playing at medium volume gives the best results.
- **Testing:** every release first runs the full automated test suite, including a virtual bench that measures the pitch and timing of every audio input and output without audio hardware (see [`docs/verification-bench.md`](docs/verification-bench.md)). Binaries are built only when all tests pass.

---

### Package overview

| Package | Purpose |
|---|---|
| `PyQt6` | Graphical user interface |
| `numpy` | Signal processing and statistics |
| `sounddevice` | Microphone and speaker access via the PortAudio library |
| `reportlab` | PDF export |
| `tinysoundfont` | SoundFont synthesizer for the drone voices (installed separately, see above) |
| GeneralUser GS SoundFont | General MIDI instrument sounds for the drone, bundled in `assets/` |

---

### Credits and licence

This program is a fork of [EarlJay1/saxophone-intonation-table](https://github.com/EarlJay1/saxophone-intonation-table). The changes made in this fork are licensed under the GNU General Public License, version 3 or later (see [`LICENSE`](LICENSE)). The upstream repository publishes no licence; code that originates there remains its author's work (see [`NOTICE.md`](NOTICE.md)). Third-party components, including the GeneralUser GS SoundFont by S. Christian Collins, keep their own licences (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

---

## 🇩🇪 Deutsch

### Beschreibung

Dieses Programm hört über das Mikrofon zu, erkennt die Tonhöhe jedes gespielten Tons und misst seine Abweichung von der gleichstufigen Stimmung in Cent. Für jeden Ton führt es Mittelwert und Standardabweichung, sodass eine Übesitzung zu einer Intonationstabelle des eigenen Instruments wird. Es kennt 73 Instrumente in 12 Familien mit ihren Transpositionen; eigene Instrumente lassen sich ergänzen.

**Funktionen:**
- Großer Tuner, auch aus größerer Entfernung ablesbar; Cent werden als ganze Zahlen angezeigt
- Live-Intonationstabelle mit Mittelwert, Standardabweichung und Anzahl der Messungen (N) pro Ton, als Liste oder als Oktav-Raster
- 73 Instrumente in 12 Familien: Saxophone, Klarinetten, Flöten, Trompeten, Hörner, Posaunen, tiefes Blech, Doppelrohrblatt, Blockflöten, Streicher, Zupfinstrumente sowie Stimme / Konzertstimmung
- Eigene Instrumente mit eigenem Namen und eigener Transposition; ein Spitzname pro Instrument; ein Tonumfang-Editor pro Instrument
- Tondarstellung gegriffen oder klingend
- Kammerton A von 430 bis 450 Hz sowie Ermittlung des am besten passenden Kammertons aus den Messdaten
- Drei Reaktionsmodi (Schnell, Normal, Langsam) — siehe [`docs/response-modes.md`](docs/response-modes.md)
- Export als TXT, PDF, PNG-Diagramm und CSV (mit optionaler Hersteller- und Modellangabe)
- CSV-Export mit sechs Aufteilungsmodi und Filtern für Lauf, Instrument und Spitzname
- CSV-Import einer Rohdaten-CSV, um Sitzungen zusammenzuführen oder zu vergleichen
- Optionaler Messdatenverlauf über Sitzungen hinweg
- Oberfläche auf Deutsch oder Englisch, jederzeit umschaltbar
- Vier Reiter: STIMMGERÄT · METRONOM · AUFNAHME · EINSTELLUNGEN
- Metronom: 30–300 BPM, betonte erste Zählzeit, 2/4, 3/4, 4/4 und 6/8, Tap-Tempo, samplegenaue Klick-Planung
- Bordun-Referenzton: 5 Schnellvorgaben plus alle 128 General-MIDI-Klänge aus der mitgelieferten SoundFont sowie eine vibratofreie reine Sinus-Stimme als Präzisionsreferenz
- Stimmpfeifen: gehaltene Sinus-Referenztöne von C4 bis H4
- Bandmaschine: eine Aufnahme über das Mikrofon machen, wiedergeben und als WAV exportieren
- Themes: Dunkel, Nacht (rotverschoben zum Schutz der dunkeladaptierten Augen in dunklem Saal oder Graben) und Hell, live umschaltbar
- Auswahl von Eingabegerät und Abtastrate (AUDIO-EINGANG), Ausgabegerät und Mikrofon-Verstärkung; der Tuner liest weiter das Mikrofon, während ein Referenzton klingt

---

### Download

Jede [Veröffentlichung](https://github.com/knelsonb/saxophone-intonation-table/releases) enthält fertige Programmdateien. Python wird nur für den Start aus dem Quellcode gebraucht.

| Datei | System |
|---|---|
| `intonation-analyzer-windows-x86_64.exe` | Windows 10 / 11, 64 Bit |
| `intonation-analyzer-linux-x86_64.tar.gz` | Linux x86-64 mit glibc 2.35 oder neuer (gebaut auf Ubuntu 22.04) |
| `intonation-analyzer-macos-arm64.tar.gz` | macOS auf Apple Silicon (M1 und neuer) |
| `SHA256SUMS.txt` | SHA-256-Prüfsummen der obigen Dateien |

**Download prüfen** gegen die passende Zeile in `SHA256SUMS.txt`:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing          # Linux
shasum -a 256 intonation-analyzer-macos-arm64.tar.gz  # macOS: Werte vergleichen
```

```cmd
certutil -hashfile intonation-analyzer-windows-x86_64.exe SHA256
```

**Windows:** die `.exe` starten. Die Datei ist nicht signiert, daher zeigt Windows eventuell „Der Computer wurde durch Windows geschützt"; dann auf **Weitere Informationen** und **Trotzdem ausführen** klicken.

**Linux:**

```bash
mkdir intonation-analyzer-linux
tar -xzf intonation-analyzer-linux-x86_64.tar.gz -C intonation-analyzer-linux
./intonation-analyzer-linux/intonation-analyzer
```

**macOS:** die Programmdatei ist nicht signiert, daher blockiert Gatekeeper den ersten Start.

```bash
mkdir intonation-analyzer-macos
tar -xzf intonation-analyzer-macos-arm64.tar.gz -C intonation-analyzer-macos
./intonation-analyzer-macos/intonation-analyzer
```

Verweigert macOS das Öffnen, die Datei im Finder mit Rechtsklick (Control-Klick) **Öffnen** und bestätigen. Ab macOS 15 gibt es diesen Weg nicht mehr: die Datei einmal öffnen, dann unter **Systemeinstellungen › Datenschutz & Sicherheit** auf **Dennoch öffnen** klicken. Danach fragt macOS einmal nach dem Mikrofonzugriff; erlauben.

---

### Voraussetzungen (Start aus dem Quellcode)

| Komponente | Version |
|---|---|
| Python | 3.10 bis 3.12 (die Release-Builds nutzen 3.12). Ab 3.13 brauchen die SoundFont-Klänge des Borduns einen C-Compiler, um `tinysoundfont` zu installieren, weil es dafür noch keine vorgebauten Pakete gibt. |
| Betriebssystem | Windows 10+, macOS, Linux (z. B. Ubuntu 22.04+) |
| Mikrofon | Beliebiges Eingangsgerät |

---

### Installation unter Linux (Ubuntu / Debian)

#### 1. Systempakete installieren

```bash
sudo apt update
sudo apt install git python3 python3-venv libportaudio2 libxcb-cursor0
```

`libportaudio2` ist die Audio-Bibliothek, die `sounddevice` nutzt; `libxcb-cursor0` braucht das Qt-Fenstersystem-Plugin. PyQt6, numpy und die übrigen Python-Pakete kommen als fertige Wheels, es muss nichts kompiliert werden.

#### 2. Code holen

```bash
git clone https://github.com/knelsonb/saxophone-intonation-table.git
cd saxophone-intonation-table
```

#### 3. Virtuelle Umgebung anlegen

```bash
python3 -m venv ~/sax-venv
```

#### 4. Virtuelle Umgebung aktivieren

```bash
source ~/sax-venv/bin/activate
```

> ⚠️ Dieser Befehl muss bei **jeder neuen Terminal-Sitzung** wiederholt werden.  
> Das Prompt zeigt dann `(sax-venv)` am Anfang.

#### 5. Python-Pakete installieren

```bash
pip install -r requirements.txt
pip install --no-deps "tinysoundfont~=0.3.7"
```

Der zweite Befehl installiert den SoundFont-Synthesizer für die Bordun-Klänge. `--no-deps` lässt seine Abhängigkeit `pyaudio` weg, die dieses Programm nicht nutzt. Ohne `tinysoundfont` läuft das Programm trotzdem; der Bordun nutzt dann die reine Sinus-Stimme.

#### 6. Programm starten

```bash
python sax_intonation_gui.py
```

#### Kurzform (Aktivieren + Starten in einem Befehl)

Setzt voraus, dass der Code im Home-Ordner liegt:

```bash
cd ~/saxophone-intonation-table && source ~/sax-venv/bin/activate && python sax_intonation_gui.py
```

---

### Installation unter Windows

#### 1. Python installieren

1. Python 3.12 von [python.org/downloads](https://www.python.org/downloads/) herunterladen
2. Installer ausführen
3. ✅ **„Add Python to PATH"** aktivieren — wichtig!
4. Installation abschließen

#### 2. Code holen

Entweder [Git for Windows](https://git-scm.com/download/win) installieren und in der Eingabeaufforderung ausführen:

```cmd
git clone https://github.com/knelsonb/saxophone-intonation-table.git
```

oder auf der GitHub-Seite **Code › Download ZIP** wählen und das ZIP entpacken. Das Programm besteht aus `sax_intonation_gui.py`, rund zwanzig `sax_*.py`-Modulen und dem Ordner `assets`; alles zusammen in einem Ordner lassen.

#### 3. Eingabeaufforderung im Programmordner öffnen

`Win + R` → `cmd` → Enter, dann in den Ordner wechseln:

```cmd
cd C:\Pfad\zum\saxophone-intonation-table
```

#### 4. Virtuelle Umgebung anlegen

```cmd
python -m venv %USERPROFILE%\sax-venv
```

#### 5. Virtuelle Umgebung aktivieren

```cmd
%USERPROFILE%\sax-venv\Scripts\activate
```

> ⚠️ Dieser Befehl muss bei **jeder neuen Eingabeaufforderung** wiederholt werden.  
> Das Prompt zeigt dann `(sax-venv)` am Anfang.

#### 6. Python-Pakete installieren

```cmd
pip install -r requirements.txt
pip install --no-deps "tinysoundfont~=0.3.7"
```

#### 7. Programm starten

```cmd
python sax_intonation_gui.py
```

---

### Installation unter macOS

1. Python 3.12 von [python.org/downloads](https://www.python.org/downloads/) installieren, außerdem die Befehlszeilenwerkzeuge (sie enthalten `git`): `xcode-select --install`
2. Im Terminal:

```bash
git clone https://github.com/knelsonb/saxophone-intonation-table.git
cd saxophone-intonation-table
python3 -m venv ~/sax-venv
source ~/sax-venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "tinysoundfont~=0.3.7"
python sax_intonation_gui.py
```

Beim ersten Start fragt macOS, ob das Terminal das Mikrofon nutzen darf; erlauben. Auf Intel-Macs gibt es für `tinysoundfont` kein vorgebautes Paket; es wird bei der Installation mit den Befehlszeilenwerkzeugen kompiliert.

---

### Fehlerbehebung

| Problem | Lösung |
|---|---|
| `externally-managed-environment` | Virtuelle Umgebung verwenden (siehe oben) |
| `OSError: PortAudio library not found` (Linux) | `sudo apt install libportaudio2` |
| Fenster öffnet sich unter Linux nicht; im Terminal ist vom Plattform-Plugin `xcb` die Rede | `sudo apt install libxcb-cursor0`. Nennt die Meldung eine weitere fehlende Bibliothek, diese ebenfalls installieren (z. B. `libxkbcommon-x11-0`). |
| Falsches Mikrofon wird verwendet | In der Werkzeugleiste auf **AUDIO-EINGANG** klicken und das Gerät wählen. Im selben Dialog lässt sich die Abtastrate einstellen; **Auto** probiert 48, 44,1, 96, 88,2 und 192 kHz in dieser Reihenfolge und nimmt die erste Rate, die das Gerät annimmt, ersatzweise die Standardrate des Geräts. |
| Kein Ton erkannt | Prüfen, ob **AUDIO-EINGANG** das richtige Mikrofon zeigt. Unter **EINSTELLUNGEN** die **Mikrofon-Verstärkung** erhöhen; sie senkt die Schwelle, ab der Spielen als Ton zählt, und lässt die Cent-Werte unverändert. Mittelstark und gleichmäßig spielen. |
| Töne werden erkannt, aber die Tabelle bleibt leer | Die Tabelle blendet Töne mit weniger Messungen als **Mindest-Messungen pro Ton** (Standard 5) aus. Den Ton etwas länger halten oder den Wert senken. |
| Der Tuner ignoriert den Ton, den Bordun oder Stimmpfeife gerade spielen | So gewollt; siehe „Bordun, Stimmpfeifen und Tuner" unten. |
| Der Bordun meldet, dass `tinysoundfont` nicht installiert ist | In der virtuellen Umgebung `pip install --no-deps "tinysoundfont~=0.3.7"` ausführen und neu starten. Bis dahin nutzt der Bordun die reine Sinus-Stimme. |
| macOS: gar kein Eingangssignal | Unter **Systemeinstellungen › Datenschutz & Sicherheit › Mikrofon** dem Terminal (bzw. dem Programm) den Zugriff erlauben. |

#### Audiogeräte über die Befehlszeile auflisten

```bash
# Linux / macOS
source ~/sax-venv/bin/activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

```cmd
:: Windows
%USERPROFILE%\sax-venv\Scripts\activate
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

### Bedienungshinweise

- **Aufbau:** Die Werkzeugleiste über den Reitern enthält Instrument, Tondarstellung, Kammerton, Reaktionsmodus, Sprache, **AUDIO-EINGANG**, CSV-Import, **Kammerton ermitteln**, Aufnahme, **Reset** und die Export-Buttons. **STIMMGERÄT** zeigt Tuner, Intonationstabelle, Bordun und Stimmpfeifen. **METRONOM** ist das Metronom, **AUFNAHME** die Bandmaschine. **EINSTELLUNGEN** enthält Audioausgang, Bordun-Klang, Farbschema, Spitzname und Tonumfang-Editor, Kammerton, Reaktionsmodus, Mikrofon-Verstärkung und einen Testton.
- **Instrumente:** erst die Familie wählen, dann das Instrument. **+ Eigenes …** fügt ein eigenes Instrument mit Anzeigename und Transposition in Halbtönen vom gegriffenen zum klingenden Ton hinzu, einschließlich Oktave (Bb-Sopran −2, Eb-Alt −9, Bb-Tenor −14). Das Spitzname-Feld kennzeichnet die Messungen (z. B. „Tenor #1"), damit der CSV-Export zwei gleichartige Instrumente vergleichen kann. **Tonumfang bearbeiten …** in den EINSTELLUNGEN (oder das Zahnrad) legt fest, welche Töne die Tabelle vorab zeigt; Töne außerhalb des Umfangs werden trotzdem erfasst, solange **Auf Instrumentenumfang beschränken** aus ist.
- **Kammerton ändern:** Die Tabelle behält ihre Töne. Ihre Cent-Werte werden aus den bereits gemessenen Frequenzen gegen das neue A neu berechnet. Eine Messung, die danach mehr als 50 Cent von ihrem Ton entfernt liegt, wandert zum Nachbarton. Es zählen nur die Töne, die gerade in der Tabelle stehen; frühere Sitzungen, Importe und zurückgesetzte oder gelöschte Töne bleiben draußen. Messungen nach dem Wechsel gehören zu einem neuen Lauf.
- **Kammerton ermitteln:** Mindestens 3 Töne mit je mindestens 5 Messungen spielen, dann **Kammerton ermitteln** drücken. Das Programm nimmt die mittlere Abweichung jedes Tons, bildet daraus den mit √N gewichteten Median und schlägt den passenden Kammerton vor, auf ganze Hz gerundet und auf 430–450 Hz begrenzt. Nach der Bestätigung setzt es das neue A und leert die Tabelle, damit die Messung beim neuen Kammerton von vorn beginnt.
- **Reset** leert die Tabelle; der nächste Export fragt erneut nach Hersteller und Modell. Ein Rechtsklick auf einen Ton bietet **Messungen für diesen Ton löschen**. Beides wirkt nur auf die Tabelle; das Messprotokoll hinter dem CSV-Export behält jede Messung.
- **Farben:** grün bis ±5 Cent, gelb bis ±12 Cent, rot darüber.
- **Export TXT / PDF / PNG-Diagramm:** Diese Exporte enthalten die gespielten Töne der aktuellen Tabelle; leere Zeilen aus dem Instrumentenumfang fallen weg. Beim ersten Export fragt das Programm nach Hersteller und Modell (beides optional). Die Angabe gilt für alle weiteren Exporte, bis **Reset** gedrückt wird; die nächste Abfrage ist damit vorausgefüllt.
- **CSV-Export:** **Export CSV** schreibt aus dem Messprotokoll — alles seit dem Programmstart Gemessene, dazu gespeicherter Verlauf und importierte CSVs. Der Button öffnet einen Dialog mit sechs Aufteilungsmodi:
  - *Rohdaten* — eine Zeile pro Messung
  - *Pro Lauf und Ton* — pro (Lauf, Ton) aggregiert (Mittel, Standardabw., Min, Max, N)
  - *Pro Instrument und Ton* — über alle Läufe eines Instruments aggregiert
  - *Pro Spitzname und Ton* — über alle Läufe mit demselben Spitznamen aggregiert, z. B. um zwei Tenöre zu vergleichen
  - *Ein Instrument, je Ton gemittelt* — verlangt die Wahl eines Instruments
  - *Gesamtmittel je Ton* — über alles aggregiert

  Der Filter **Lauf** gilt für *Rohdaten* und *Pro Lauf und Ton*. Der Filter **Instrument** gilt für alle Modi außer *Gesamtmittel je Ton*. Der Filter **Spitzname** gilt für *Rohdaten*, *Pro Lauf und Ton*, *Pro Instrument und Ton* und *Pro Spitzname und Ton* und behält nur Läufe, die unter diesem Spitznamen gemessen wurden. Ein „Lauf" beginnt beim Programmstart, bei Instrument- oder Kammertonwechsel sowie beim Fortsetzen einer pausierten Aufnahme.
- **CSV-Import:** **CSV importieren** nimmt Dateien aus dem Modus *Rohdaten* an (bis 50 MB oder 500.000 Zeilen). Nach einer Rückfrage fügt es Läufe und Messungen dem Messprotokoll hinzu und überspringt Läufe, die schon vorhanden sind. Importierte Daten erscheinen im CSV-Export; die Live-Tabelle bleibt, wie sie ist.
- **Messdatenverlauf:** Der Willkommensdialog beim ersten Start bietet **Messdatenverlauf dauerhaft speichern** an. Ist das aktiviert, wird jede Messung an `~/.intonation_analyzer/log.jsonl` angehängt (unter Windows `C:\Users\<Name>\.intonation_analyzer\log.jsonl`) und beim nächsten Start wieder geladen, sodass der CSV-Export mehrere Sitzungen umfassen kann. Die Tabelle beginnt immer leer. **Später entscheiden** zeigt den Dialog beim nächsten Start erneut. Um den Verlauf einzuschalten, nachdem der Dialog ohne Häkchen bestätigt wurde, das Programm schließen und in `config.json` (siehe unten) `"persistence_enabled": true` setzen. Die Umgebungsvariable `SAX_INTONATION_LOG_PATH` hat Vorrang: Nennt sie eine Datei, werden die Messungen unabhängig von der Einstellung dort gespeichert.
- **Einstellungsordner:** `~/.intonation_analyzer/` enthält `config.json` (Einstellungen und Fensterlayout), `custom_instruments.json` (eigene Instrumente), `instrument_ranges.json` (bearbeitete Tonumfänge) und bei aktivem Verlauf `log.jsonl`. Wer den Ordner löscht, setzt das Programm auf den Zustand des ersten Starts zurück und löscht den gespeicherten Verlauf.
- **Bordun, Stimmpfeifen und Tuner:** Solange Bordun oder eine Stimmpfeife klingt, ignoriert der Tuner Messungen genau auf dem Referenzton: Über das Mikrofon sind der eigene Ton auf dieser Höhe und der Klang aus dem Lautsprecher nicht zu unterscheiden. Andere Töne werden normal gemessen. Um den Referenzton selbst zu messen, die Referenz ausschalten oder den Bordun auf einen anderen Ton legen (Halbton-Buttons) und im Intervall dagegen stimmen. Hört das Mikrofon den Bordun-Ton anhaltend, regelt sich der Bordun selbst leiser und danach wieder lauter.
- **Genauigkeit:** Bei einem gehaltenen Ton kommt etwa alle 46 ms eine Messung hinzu, also rund 20 pro Sekunde. Die erste Messung eines Tons erscheint nach etwa 0,2 s (Schnell), 0,3 s (Normal) bzw. 0,65 s (Langsam). Etwa eine Sekunde gleichmäßiges Spiel pro Ton ergibt einen stabilen Mittelwert; wer jeden Ton im Lauf einer Sitzung mehrmals neu anstößt, sieht, wie er ihn tatsächlich spielt. Am besten gleichmäßig und mittelstark spielen.
- **Tests:** Vor jeder Veröffentlichung läuft die vollständige automatische Testsuite, darunter eine virtuelle Messbank, die Tonhöhe und Timing aller Audio-Ein- und -Ausgänge ohne Audio-Hardware misst (siehe [`docs/verification-bench.md`](docs/verification-bench.md)). Programmdateien werden nur gebaut, wenn alle Tests bestehen.

---

### Paketübersicht

| Paket | Zweck |
|---|---|
| `PyQt6` | Grafische Oberfläche |
| `numpy` | Signalverarbeitung und Statistik |
| `sounddevice` | Mikrofon- und Lautsprecherzugriff über die Bibliothek PortAudio |
| `reportlab` | PDF-Export |
| `tinysoundfont` | SoundFont-Synthesizer für die Bordun-Klänge (separat installiert, siehe oben) |
| GeneralUser-GS-SoundFont | General-MIDI-Instrumentenklänge für den Bordun, mitgeliefert in `assets/` |

---

### Herkunft und Lizenz

Dieses Programm ist ein Fork von [EarlJay1/saxophone-intonation-table](https://github.com/EarlJay1/saxophone-intonation-table). Die Änderungen dieses Forks stehen unter der GNU General Public License, Version 3 oder neuer (siehe [`LICENSE`](LICENSE)). Das Ursprungs-Repository veröffentlicht keine Lizenz; Code, der von dort stammt, bleibt Werk seines Autors (siehe [`NOTICE.md`](NOTICE.md)). Komponenten Dritter, darunter die GeneralUser-GS-SoundFont von S. Christian Collins, behalten ihre eigenen Lizenzen (siehe [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).
