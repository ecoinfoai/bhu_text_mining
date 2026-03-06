"""Tests for the mecab shim (mecab-python3 → python-mecab-kor adapter)."""

from __future__ import annotations

import sys

import pytest


class TestShimInstallation:
    """Verify the shim registers correctly in sys.modules."""

    def test_mecab_in_sys_modules(self):
        import forma.mecab_shim  # noqa: F401

        assert "mecab" in sys.modules

    def test_shim_installed_flag(self):
        import forma.mecab_shim

        # On this NixOS system, python-mecab-kor is absent so shim
        # should be installed.  On systems where python-mecab-kor
        # works natively, _installed may be False (that is also fine).
        assert isinstance(forma.mecab_shim._installed, bool)

    def test_from_mecab_import_mecab(self):
        import forma.mecab_shim  # noqa: F401

        from mecab import MeCab

        m = MeCab()
        assert hasattr(m, "pos")
        assert hasattr(m, "morphs")


class TestMeCabAPI:
    """Verify the shimmed MeCab matches python-mecab-kor's API."""

    @pytest.fixture()
    def mecab(self):
        import forma.mecab_shim  # noqa: F401
        from mecab import MeCab

        return MeCab()

    def test_pos_returns_list_of_tuples(self, mecab):
        result = mecab.pos("테스트")
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_pos_surfaces_are_strings(self, mecab):
        result = mecab.pos("아버지가 방에 들어오시다.")
        surfaces = [s for s, _ in result]
        assert all(isinstance(s, str) for s in surfaces)
        assert "아버지" in surfaces

    def test_pos_tags_are_korean_tagset(self, mecab):
        result = mecab.pos("아버지가 방에 들어오시다.")
        tags = [t for _, t in result]
        # NNG = common noun, JKS = subject marker — standard mecab-ko-dic tags
        assert "NNG" in tags

    def test_morphs_returns_list_of_strings(self, mecab):
        result = mecab.morphs("테스트 문장입니다.")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_morphs_consistent_with_pos(self, mecab):
        text = "세포막은 인지질로 구성된다."
        morphs = mecab.morphs(text)
        surfaces = [s for s, _ in mecab.pos(text)]
        assert morphs == surfaces

    def test_smoke_test_underscore(self, mecab):
        """Kss calls MeCab().morphs('_') as a smoke test."""
        result = mecab.morphs("_")
        assert isinstance(result, list)


class TestKssIntegration:
    """Verify kss detects and uses mecab backend via the shim."""

    def test_mecab_analyzer_backend(self):
        import forma.mecab_shim  # noqa: F401
        from kss._modules.morphemes.analyzers import MecabAnalyzer

        assert MecabAnalyzer._backend == "mecab"

    def test_mecab_analyzer_not_none(self):
        import forma.mecab_shim  # noqa: F401
        from kss._modules.morphemes.analyzers import MecabAnalyzer

        assert MecabAnalyzer._analyzer is not None

    def test_split_sentences_works(self):
        import forma  # noqa: F401 — triggers shim
        import kss

        result = kss.split_sentences(
            "세포막은 중요하다. 삼투도 중요하다."
        )
        assert len(result) == 2
        assert "세포막" in result[0]
        assert "삼투" in result[1]

    def test_no_runtime_warning(self):
        """With mecab backend, pecab's overflow warning should not appear."""
        import warnings

        import forma  # noqa: F401

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            import kss

            # This would raise if pecab overflow occurs
            kss.split_sentences("형태소 분석 테스트입니다.")
