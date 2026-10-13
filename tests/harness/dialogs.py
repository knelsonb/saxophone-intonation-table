"""Modal-dialog interception.

A headless test must never block on a modal, so every way the app can open one
is replaced while an app is alive:

* ``QMessageBox.question / information / warning / critical`` (static calls),
* ``QFileDialog.getSaveFileName / getOpenFileName``,
* ``QDialog.exec()`` for the app's own form dialogs (welcome, maker/model,
  CSV slice, custom instrument, range editor, ...), and
* ``QMenu.exec()`` for context menus.

Tests pre-program the answers::

    app.dialogs.expect('question', answer='Yes')
    app.dialogs.save_path(tmp_path / 'out.txt')
    app.dialogs.expect('maker_model', maker='Selmer', model='Mark VI')

Anything the app opens that nobody asked for is *recorded and answered with the
safe "no / cancel"*, never raised from inside Qt (an exception escaping a slot
would abort the interpreter). The recorded problem is raised from the test by
``assert_clean()``, which every harness action calls on its way out.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

MESSAGE_KINDS = ("question", "information", "warning", "critical")


@dataclass
class Message:
    """A QMessageBox the app showed."""
    kind: str
    title: str
    text: str
    answer: str


@dataclass
class _Expectation:
    answer: Optional[str]
    match: Optional[str]
    optional: bool
    fields: dict
    scope: Optional[int]


class ScopeResult:
    """Messages shown while a ``Dialogs.scope()`` block was open."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    @property
    def last(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None


class Dialogs:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.file_dialogs: list[tuple] = []        # (kind, caption, directory, filter)
        self.problems: list[str] = []              # unexpected modals / unmet expectations
        self.forms_shown: list[str] = []           # kind of every form dialog opened, in order
        self._msg: dict[str, deque] = {k: deque() for k in MESSAGE_KINDS}
        self._files: dict[str, deque] = {"save": deque(), "open": deque()}
        self._forms: dict[str, deque] = {}
        self._menus: deque = deque()
        self._scopes: list[tuple[int, ScopeResult]] = []
        self._next_scope = 1
        # Registered by the driver, which is the only code that knows how to
        # recognise and fill the app's form dialogs.
        self.classify: Optional[Callable[[Any], Optional[str]]] = None
        self.fillers: dict[str, Callable[[Any, str, dict], bool]] = {}
        self.auto: set[str] = set()

    # ------------------------------------------------------------- scripting
    def expect(self, kind: str, answer: Optional[str] = None, *,
               match: Optional[str] = None, optional: bool = False, **fields) -> None:
        """Queue an answer for the next dialog of ``kind``.

        For message boxes ``kind`` is one of ``question``, ``information``,
        ``warning``, ``critical`` and ``answer`` names the button ('Yes', 'No',
        'Ok', 'Cancel'). For the app's form dialogs ``kind`` is the harness name
        of the form and ``answer`` is 'accept' (default) or 'reject'; extra
        keyword arguments fill the form's fields. ``match`` is a case-insensitive
        substring the dialog's title or text must contain.
        """
        scope = self._scopes[-1][0] if self._scopes else None
        entry = _Expectation(answer, match, optional, dict(fields), scope)
        if kind in MESSAGE_KINDS:
            self._msg[kind].append(entry)
        else:
            self._forms.setdefault(kind, deque()).append(entry)

    def save_path(self, path) -> None:
        """Answer the next 'save file' dialog with ``path`` (None cancels)."""
        self._files["save"].append(None if path is None else str(path))

    def open_path(self, path) -> None:
        """Answer the next 'open file' dialog with ``path`` (None cancels)."""
        self._files["open"].append(None if path is None else str(path))

    def expect_menu(self, item: str) -> None:
        """Choose the context-menu entry whose text contains ``item``."""
        self._menus.append(item)

    def auto_acknowledge(self, *kinds: str) -> None:
        """Acknowledge message boxes of these kinds without a scripted answer
        (they are still recorded). Off by default: a surprise is a failure."""
        self.auto.update(kinds)

    @contextmanager
    def scope(self):
        """Group the expectations an API call adds. Unconsumed non-optional ones
        are reported as problems when the block ends; optional ones vanish."""
        sid = self._next_scope
        self._next_scope += 1
        result = ScopeResult()
        self._scopes.append((sid, result))
        try:
            yield result
        finally:
            self._scopes.pop()
            for queues in (self._msg, self._forms):
                for kind, q in queues.items():
                    keep = deque()
                    for e in q:
                        if e.scope == sid:
                            if not e.optional:
                                self.problems.append(
                                    f"expected {kind} dialog was never shown"
                                    + (f" (match={e.match!r})" if e.match else ""))
                        else:
                            keep.append(e)
                    queues[kind] = keep

    # ------------------------------------------------------------- inspection
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def pending(self) -> list[str]:
        """Scripted answers nobody has used yet."""
        out = []
        for kind, q in list(self._msg.items()) + list(self._forms.items()):
            out += [f"{kind}" + (f" (match={e.match!r})" if e.match else "")
                    for e in q if not e.optional]
        out += [f"{k}-file path" for k, q in self._files.items() for _ in q]
        out += [f"menu item {m!r}" for m in self._menus]
        return out

    def assert_clean(self) -> None:
        """Fail if the app opened a modal nobody scripted (or a scripted one
        was skipped). Clears the problem list so the failure is reported once."""
        if self.problems:
            problems, self.problems = self.problems, []
            raise AssertionError("modal dialog problems:\n  - " + "\n  - ".join(problems))

    def assert_none_pending(self) -> None:
        left = self.pending()
        if left:
            raise AssertionError("scripted dialog answers never used: " + ", ".join(left))

    # ------------------------------------------------------------- internals
    def _scope_record(self, msg: Message) -> None:
        self.messages.append(msg)
        for _sid, res in self._scopes:
            res.messages.append(msg)

    def _take(self, queue: deque, title: str, text: str) -> Optional[_Expectation]:
        hay = f"{title}\n{text}".lower()
        for i, e in enumerate(queue):
            if e.match is None or e.match.lower() in hay:
                del queue[i]
                return e
        return None

    def _message_box(self, kind: str, title: str, text: str, default_answer: str) -> str:
        entry = self._take(self._msg[kind], title, text)
        if entry is not None:
            answer = entry.answer or default_answer
        elif kind in self.auto:
            answer = default_answer
        else:
            self.problems.append(f"unexpected {kind} message box {title!r}: {text}")
            answer = "No" if kind == "question" else "Ok"
        self._scope_record(Message(kind, title, text, answer))
        return answer

    # -------------------------------------------------------------- patching
    def install(self, patches) -> None:
        """Replace the Qt modal entry points. ``patches`` is a PatchStack."""
        from PyQt6.QtWidgets import QDialog, QFileDialog, QMenu, QMessageBox

        dialogs = self
        button = QMessageBox.StandardButton

        def make_box(kind: str, default_answer: str):
            def box(parent=None, title="", text="", buttons=None, defaultButton=None):
                # question() defaults to Yes|No; the app passes Ok|Cancel for the
                # CSV import prompt. A scripted answer must name a button present
                # in `buttons` when given.
                answer = dialogs._message_box(kind, str(title), str(text), default_answer)
                return getattr(button, answer)
            return staticmethod(box)

        patches.setattr(QMessageBox, "question", make_box("question", "No"))
        patches.setattr(QMessageBox, "information", make_box("information", "Ok"))
        patches.setattr(QMessageBox, "warning", make_box("warning", "Ok"))
        patches.setattr(QMessageBox, "critical", make_box("critical", "Ok"))

        def make_file(kind: str):
            def pick(parent=None, caption="", directory="", filter="", *a, **k):
                dialogs.file_dialogs.append((kind, str(caption), str(directory), str(filter)))
                q = dialogs._files[kind]
                if q:
                    path = q.popleft()
                    return ("" if path is None else path, "")
                dialogs.problems.append(
                    f"unexpected {kind} file dialog {str(caption)!r} (start: {directory!r})")
                return ("", "")
            return staticmethod(pick)

        patches.setattr(QFileDialog, "getSaveFileName", make_file("save"))
        patches.setattr(QFileDialog, "getOpenFileName", make_file("open"))

        def fake_exec(dlg, *a, **k):
            kind = dialogs.classify(dlg) if dialogs.classify else None
            if kind is not None:
                dialogs.forms_shown.append(kind)
            title = dlg.windowTitle() if hasattr(dlg, "windowTitle") else ""
            if kind is None:
                dialogs.problems.append(f"unexpected modal dialog {title!r} "
                                        f"({type(dlg).__name__})")
                return int(QDialog.DialogCode.Rejected)
            q = dialogs._forms.get(kind)
            entry = dialogs._take(q, title, "") if q else None
            if entry is None:
                dialogs.problems.append(f"unexpected {kind} dialog {title!r}")
                return int(QDialog.DialogCode.Rejected)
            answer = entry.answer or "accept"
            accepted = dialogs.fillers[kind](dlg, answer, entry.fields)
            return int(QDialog.DialogCode.Accepted if accepted
                       else QDialog.DialogCode.Rejected)

        patches.setattr(QDialog, "exec", fake_exec)

        def fake_menu_exec(menu, *a, **k):
            if not dialogs._menus:
                dialogs.problems.append("unexpected context menu: "
                                        + ", ".join(x.text() for x in menu.actions()))
                return None
            wanted = dialogs._menus.popleft().lower()
            for act in menu.actions():
                if wanted in act.text().lower():
                    return act
            dialogs.problems.append(f"context menu has no item {wanted!r}")
            return None

        patches.setattr(QMenu, "exec", fake_menu_exec)
