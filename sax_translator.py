# =============================================================================
# sax_translator.py — the UI language and every text drawn in it
#
# One Translator per window. It answers ``t(key, **kwargs)`` in the current
# language and keeps a registry of the texts on screen: each widget says once
# how its text is made (``bind``, ``bind_text``, ``bind_items`` ...) and the
# Translator re-renders all of them when the language changes. A message that
# is showing (a banner, an inline status) is bound the same way, by key and
# arguments rather than by its rendered text, so it is re-worded too.
#
# Why a registry instead of a hand-written retranslate walk: the walk had to
# name every widget a second time and kept missing the ones added later
# (whole tabs, combo entries, open dialogs). Binding at the place a text is
# first set makes "built in one language, never updated" impossible to write.
#
# Rules for bindings:
#   * Binding the same target again replaces the earlier binding, so a label
#     that shows one message at a time just binds the new one.
#   * A binding whose widget has been deleted is dropped on the next
#     re-render; nothing has to unbind on close.
#   * Bindings re-render first, then ``language_changed`` fires, so slots that
#     rebuild content (instrument names, the table) see current widget texts.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional

from PyQt6 import sip
from PyQt6.QtCore import QObject, pyqtSignal

from sax_i18n import STRINGS

LANGUAGES = ('de', 'en')
_FALLBACK = 'de'


def translate(lang: str, key: str, **kwargs: Any) -> str:
    """``key`` in ``lang``; German when the language lacks it, ``[key]`` when
    no language has it. Formats only when arguments are given, so a string
    without placeholders may contain literal braces."""
    table = STRINGS.get(lang, STRINGS[_FALLBACK])
    text = table.get(key, STRINGS[_FALLBACK].get(key, f'[{key}]'))
    return text.format(**kwargs) if kwargs else text


@dataclass(frozen=True)
class Message:
    """A text by key and arguments, rendered in whatever language is current
    when it is called. Two messages with the same key and arguments are
    equal, so code can ask whether a banner still shows a given message."""
    translator: 'Translator' = field(compare=False, repr=False)
    key: str
    kwargs: tuple = ()

    def __call__(self) -> str:
        return self.translator.t(self.key, **dict(self.kwargs))


class _Binding:
    __slots__ = ('owner', 'render')

    def __init__(self, owner: Optional[QObject], render: Callable[[], None]):
        self.owner = owner
        self.render = render


def _owner_of(setter: Callable) -> Optional[QObject]:
    owner = getattr(setter, '__self__', None)
    return owner if isinstance(owner, QObject) else None


class Translator(QObject):
    """The current UI language, ``t()`` in it, and the registry of bound
    texts that follow a language switch."""

    language_changed = pyqtSignal(str)

    def __init__(self, lang: str = 'en', parent: Optional[QObject] = None):
        super().__init__(parent)
        self._lang = lang if lang in STRINGS else 'en'
        self._bindings: dict[Hashable, _Binding] = {}

    # ── Language ─────────────────────────────────────────────────────────────
    @property
    def lang(self) -> str:
        return self._lang

    def t(self, key: str, **kwargs: Any) -> str:
        return translate(self._lang, key, **kwargs)

    def msg(self, key: str, **kwargs: Any) -> Message:
        """A message to show now and re-word on every later switch."""
        return Message(self, key, tuple(sorted(kwargs.items())))

    def set_language(self, lang: str) -> None:
        """Switch the language and re-render everything bound. An unknown
        language or the current one changes nothing."""
        if lang not in STRINGS or lang == self._lang:
            return
        self._lang = lang
        self.retranslate()
        self.language_changed.emit(lang)

    def retranslate(self) -> None:
        for target, binding in list(self._bindings.items()):
            if not self._render(binding):
                self._bindings.pop(target, None)

    @staticmethod
    def _render(binding: _Binding) -> bool:
        """Run one binding; False when its widget is gone."""
        owner = binding.owner
        if owner is not None and sip.isdeleted(owner):
            return False
        try:
            binding.render()
        except RuntimeError:
            # The underlying C++ object of a widget the binding writes to was
            # deleted without the wrapper knowing yet.
            return False
        return True

    # ── Bindings ─────────────────────────────────────────────────────────────
    def bind_text(self, setter: Callable[[str], Any], text: Callable[[], str], *,
                  target: Optional[Hashable] = None,
                  owner: Optional[QObject] = None) -> None:
        """Call ``setter(text())`` now and after every language switch.
        ``target`` identifies the binding (default: the setter itself, so
        binding ``label.setText`` again replaces the earlier text); ``owner``
        is the widget whose deletion ends it (default: the setter's object)."""
        binding = _Binding(owner if owner is not None else _owner_of(setter),
                           lambda: setter(text()))
        self._bindings[target if target is not None else setter] = binding
        self._render(binding)

    def bind(self, setter: Callable[[str], Any], key: str, **kwargs: Any) -> None:
        """``setter`` shows the string ``key`` (formatted with ``kwargs``)."""
        self.bind_text(setter, lambda: self.t(key, **kwargs))

    def unbind(self, target: Hashable) -> None:
        """Stop re-rendering ``target`` (a setter, or a combo / tab target),
        e.g. before it shows text that is not translated, like a device name."""
        self._bindings.pop(target, None)

    def bind_item(self, combo, index: int, key: str, **kwargs: Any) -> None:
        """Entry ``index`` of ``combo`` shows ``key``; the other entries are
        the caller's (device names, say). Ignored while the entry is absent."""
        def render() -> None:
            if index < combo.count():
                combo.setItemText(index, self.t(key, **kwargs))
        self.bind_text(lambda _text: render(), lambda: '',
                       target=(combo, 'item', index), owner=combo)

    def bind_items(self, combo, items: Iterable[tuple[str, Any]]) -> None:
        """``combo`` holds exactly ``items``, ``(key, data)`` pairs, labelled
        in the current language. The selection (by data) survives a switch and
        no signal is emitted while the labels change."""
        items = list(items)

        def render() -> None:
            blocked = combo.blockSignals(True)
            try:
                same = (combo.count() == len(items) and all(
                    combo.itemData(i) == data for i, (_k, data) in enumerate(items)))
                if same:
                    for i, (key, _data) in enumerate(items):
                        combo.setItemText(i, self.t(key))
                else:
                    current = combo.currentData()
                    combo.clear()
                    for key, data in items:
                        combo.addItem(self.t(key), data)
                    idx = combo.findData(current) if current is not None else -1
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            finally:
                combo.blockSignals(blocked)
        self.bind_text(lambda _text: render(), lambda: '',
                       target=(combo, 'items'), owner=combo)

    def bind_tab(self, tabs, index: int, key: str,
                 tip_key: Optional[str] = None) -> None:
        """Tab ``index`` of ``tabs`` is labelled ``key`` (tooltip ``tip_key``)."""
        self.bind_text(lambda text: tabs.setTabText(index, text),
                       lambda: self.t(key),
                       target=(tabs, 'tab', index), owner=tabs)
        if tip_key is not None:
            self.bind_text(lambda text: tabs.setTabToolTip(index, text),
                           lambda: self.t(tip_key),
                           target=(tabs, 'tab_tip', index), owner=tabs)
