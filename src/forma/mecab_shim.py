"""Shim: make mecab-python3 + mecab-ko-dic look like python-mecab-kor.

Kss 6.x tries ``from mecab import MeCab`` first (python-mecab-kor).
That package cannot build on NixOS because its install script writes
to ``/usr/local/``.  However, ``mecab-python3`` and ``mecab-ko-dic``
ARE installed and functional.

This module creates a thin adapter that wraps ``mecab-python3``'s
``MeCab.Tagger`` with the ``python-mecab-kor`` API (``MeCab().pos()``,
``MeCab().morphs()``) and registers it as the ``mecab`` module in
``sys.modules`` so that kss's ``_get_linux_mecab()`` succeeds.

Import this module before kss is first imported.  The recommended way
is via ``forma/__init__.py``.
"""

from __future__ import annotations

import sys
import types


def install() -> bool:
    """Install the mecab shim into ``sys.modules``.

    Guards:
    - Skips if real ``python-mecab-kor`` is already usable.
    - Skips if ``mecab-python3`` or ``mecab-ko-dic`` is missing.

    Returns:
        True if the shim was installed, False otherwise.
    """
    # Guard 1: real python-mecab-kor already works → skip
    if "mecab" in sys.modules:
        return False
    try:
        from mecab import MeCab as _Real  # noqa: F811

        _Real().morphs("_")
        return False  # real package works
    except Exception:
        pass

    # Guard 2: mecab-python3 + mecab-ko-dic must be available
    try:
        import MeCab as _MeCab  # noqa: N811
        import mecab_ko_dic
    except ImportError:
        return False

    # Verify Tagger actually works before committing
    try:
        _tagger = _MeCab.Tagger(mecab_ko_dic.MECAB_ARGS)
        _tagger.parseToNode("_")
    except Exception:
        return False

    _mecab_args = mecab_ko_dic.MECAB_ARGS

    class MeCab:
        """python-mecab-kor compatible wrapper around mecab-python3."""

        def __init__(self) -> None:
            self._tagger = _MeCab.Tagger(_mecab_args)

        def pos(self, text: str) -> list[tuple[str, str]]:
            """Morphological analysis returning (surface, POS) tuples."""
            node = self._tagger.parseToNode(text)
            result: list[tuple[str, str]] = []
            while node:
                if node.surface:
                    tag = node.feature.split(",")[0]
                    result.append((node.surface, tag))
                node = node.next
            return result

        def morphs(self, text: str) -> list[str]:
            """Return surface forms only."""
            return [surface for surface, _ in self.pos(text)]

    # Inject into sys.modules
    mod = types.ModuleType("mecab")
    mod.__doc__ = "mecab shim: mecab-python3 wrapped as python-mecab-kor"
    mod.MeCab = MeCab  # type: ignore[attr-defined]
    sys.modules["mecab"] = mod

    # Belt-and-suspenders: if kss was already imported, refresh the
    # MecabAnalyzer class variable so it picks up the new backend.
    try:
        from kss._modules.morphemes.utils import _get_mecab
        from kss._modules.morphemes.analyzers import MecabAnalyzer

        if MecabAnalyzer._backend != "mecab":
            MecabAnalyzer._analyzer, MecabAnalyzer._backend = _get_mecab()
    except Exception:
        pass

    return True


_installed = install()
