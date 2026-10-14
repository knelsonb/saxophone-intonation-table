"""Start the app: ``python -m intonation`` (``python main.py`` delegates here).

The crash handler goes in first and the window module loads after it, so a
failing import or an error before the window exists is still logged and
shown (see intonation.crash).
"""
import sys

from intonation import APP_NAME, APP_VERSION, crash


def main() -> int:
    crash.install()
    from PyQt6.QtCore import QLocale
    from PyQt6.QtWidgets import QApplication

    from intonation.ui import main_window

    # Until the window exists the crash dialog follows the system locale, as
    # the window's own first-launch default does.
    crash.set_lang_source(
        lambda: 'de' if QLocale.system().name().startswith('de') else 'en')
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    icon = main_window.load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = main_window.MainWindow()
    crash.set_lang_source(lambda: win.lang)
    if icon is not None:
        win.setWindowIcon(icon)
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
