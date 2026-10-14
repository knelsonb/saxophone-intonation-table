"""Last-resort handling for exceptions that nothing else caught.

With the default sys.excepthook, a Python exception that escapes a Qt slot
either aborts the process (PyQt's qFatal path in older releases) or is only
printed to stderr, which a windowed build does not show. Either way the user
gets no explanation. install() replaces the hook: the traceback is appended
to crash.log in the data directory (intonation.persistence.paths) and, when a QApplication is
running, a translated message box tells the user where the log is. The app
keeps running afterwards.

Everything here must work before QApplication exists and must never raise:
a failure inside the hook would hide the original error.
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Callable, Optional

from intonation.i18n.strings import STRINGS
from intonation.persistence.paths import CRASH_LOG_FILENAME, AppPaths

CRASH_LOG_NAME = CRASH_LOG_FILENAME
# Large enough for dozens of tracebacks; older entries are dropped first.
CRASH_LOG_MAX_BYTES = 256 * 1024

# Returns 'de' or 'en'. Replaced by the window once it exists, so the dialog
# follows the language the user picked.
def _default_lang() -> str:
    return "en"


_lang_source: Callable[[], str] = _default_lang
# A slot that keeps failing would otherwise stack one modal box per failure.
_dialog_open = False


# Set by the window from its AppPaths. Until then (an error during start-up)
# the path is resolved from the environment when the error happens.
_log_path: Optional[Path] = None


def set_log_path(path: Optional[Path]) -> None:
    global _log_path
    _log_path = path


def crash_log_path() -> Path:
    if _log_path is not None:
        return _log_path
    return AppPaths.from_environment().crash_log


def append_crash_log(text: str, path: Optional[Path] = None,
                     max_bytes: int = CRASH_LOG_MAX_BYTES) -> Path:
    """Append one timestamped entry and keep the file within ``max_bytes``
    by dropping the oldest bytes, cut at a line boundary. Written via a temp
    file and os.replace so a crash during the write cannot truncate the log.
    Raises OSError if the log cannot be written."""
    path = path if path is not None else crash_log_path()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    entry = f"=== {stamp} ===\n{text.rstrip()}\n\n".encode("utf-8", "replace")
    try:
        old = path.read_bytes()
    except FileNotFoundError:
        old = b""
    data = old + entry
    if len(data) > max_bytes:
        data = data[-max_bytes:]
        cut = data.find(b"\n")
        if 0 <= cut < len(data) - 1:
            data = data[cut + 1:]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".crash.", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def set_lang_source(source: Callable[[], str]) -> None:
    global _lang_source
    _lang_source = source


def _lang() -> str:
    try:
        lang = _lang_source()
    except Exception:
        return "en"
    return lang if lang in STRINGS else "en"


def _t(key: str, **kwargs) -> str:
    text = STRINGS[_lang()].get(key) or STRINGS["en"][key]
    return text.format(**kwargs) if kwargs else text


def _echo(text: str) -> None:
    # sys.__stderr__ is None in a windowed (no console) frozen build.
    stream = sys.__stderr__
    if stream is None:
        return
    try:
        stream.write(text)
        stream.flush()
    except Exception:
        pass


def show_crash_dialog(log_path: Optional[Path], details: str) -> None:
    """Tell the user something went wrong. Only on the GUI thread of a
    running QApplication; a widget created anywhere else would crash."""
    global _dialog_open
    if _dialog_open:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return
    if QApplication.instance() is None:
        return
    if log_path is not None:
        message = _t("crash_msg", path=str(log_path))
    else:
        message = _t("crash_msg_no_log")
    _dialog_open = True
    try:
        box = QMessageBox(QMessageBox.Icon.Critical, _t("crash_title"), message)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setDetailedText(details)
        box.exec()
    except Exception:
        pass
    finally:
        _dialog_open = False


def make_excepthook(show_dialog: Callable[[Optional[Path], str], None]
                    = show_crash_dialog):
    """Build a sys.excepthook that logs the traceback and calls
    ``show_dialog(log_path_or_None, traceback_text)``."""
    def hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            log_path: Optional[Path] = append_crash_log(details)
        except Exception:
            log_path = None
        _echo(details)
        try:
            show_dialog(log_path, details)
        except Exception:
            pass
    return hook


def _thread_hook(args) -> None:
    """threading.excepthook: log only. A dialog cannot be shown from a
    worker thread, and the default hook would only print to a console the
    frozen build does not have."""
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    details = f"Thread {name}:\n" + "".join(traceback.format_exception(
        args.exc_type, args.exc_value, args.exc_traceback))
    try:
        append_crash_log(details)
    except Exception:
        pass
    _echo(details)


def install() -> None:
    """Install the crash handlers for the main thread and worker threads.
    Safe to call before QApplication exists."""
    sys.excepthook = make_excepthook()
    threading.excepthook = _thread_hook
