# Third-party notices

The release binaries bundle the following components. Each keeps its own
licence; versions are those resolved at build time from `requirements.txt`.

| Component | Use | Licence |
|---|---|---|
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) and Qt 6 | User interface | GPL-3.0-only (PyQt6); LGPL v3 (Qt libraries) |
| [NumPy](https://numpy.org/) | Signal processing | BSD 3-Clause |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) | Audio input and output | MIT |
| [PortAudio](https://www.portaudio.com/) | Audio backend bundled with sounddevice | MIT-style PortAudio licence |
| [ReportLab](https://www.reportlab.com/) | PDF export | BSD 3-Clause |
| [tinysoundfont](https://pypi.org/project/tinysoundfont/) | SoundFont synthesis for the drone voices | MIT |
| [GeneralUser GS 2.0.3](https://www.schristiancollins.com/generaluser.php) by S. Christian Collins | General MIDI SoundFont (`assets/GeneralUser-GS.sf2`) | GeneralUser GS License v2.0, reproduced in `assets/GeneralUser-GS-LICENSE.txt` |
| Qt FlowLayout example | Basis of `intonation/ui/flow_layout.py` | BSD 3-Clause (The Qt Company) |

## GeneralUser GS

The SoundFont's author permits its use in software projects free of charge and
states that the provenance of a few samples cannot be fully guaranteed. The full
licence text is in `assets/GeneralUser-GS-LICENSE.txt` and inside the `.sf2`
file's information chunk.

## PyQt6

PyQt6 is available under the GPL v3 or a commercial licence from Riverbank
Computing. These binaries use it under the GPL v3, which is why the fork's own
code is GPL-3.0-or-later.
