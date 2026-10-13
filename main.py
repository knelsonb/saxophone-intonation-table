"""Intonation Analyzer launcher: ``python main.py``.

The crash handler goes in first and the app modules load after it, so a
failing import or an error before the window exists is still logged and
shown (see sax_crash).
"""
import sys

import sax_crash


def main() -> int:
    sax_crash.install()
    from PyQt6.QtCore import QLocale
    from PyQt6.QtWidgets import QApplication

    import sax_intonation_gui as gui

    # Until the window exists the crash dialog follows the system locale, as
    # the window's own first-launch default does.
    sax_crash.set_lang_source(
        lambda: 'de' if QLocale.system().name().startswith('de') else 'en')
    app = QApplication(sys.argv)
    app.setApplicationName(gui.APP_NAME)
    app.setApplicationDisplayName(gui.APP_NAME)
    app.setApplicationVersion(gui.APP_VERSION)
    icon = gui.load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    win = gui.MainWindow()
    sax_crash.set_lang_source(lambda: win.lang)
    if icon is not None:
        win.setWindowIcon(icon)
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
