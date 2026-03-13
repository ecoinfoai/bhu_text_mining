"""Tests for shared chart utilities (chart_utils.py).

Covers:
- save_fig() returns BytesIO with valid PNG
- plt.close() called after save_fig()
- try/finally safety (exception during savefig still closes figure)
"""

from __future__ import annotations

import io
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from forma.chart_utils import save_fig  # noqa: E402


class TestSaveFig:
    """Tests for save_fig() function."""

    def test_returns_bytesio_with_png_header(self) -> None:
        """save_fig() returns BytesIO with content starting with PNG signature."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        buf = save_fig(fig)
        assert isinstance(buf, io.BytesIO)
        data = buf.read()
        assert data[:4] == b"\x89PNG"

    def test_buffer_is_seeked_to_start(self) -> None:
        """Returned BytesIO is positioned at offset 0."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        buf = save_fig(fig)
        assert buf.tell() == 0

    def test_figure_is_closed_after_save(self) -> None:
        """plt.close(fig) is called after saving."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        with patch.object(plt, "close", wraps=plt.close) as mock_close:
            save_fig(fig)
            mock_close.assert_called_once_with(fig)

    def test_figure_is_closed_even_on_exception(self) -> None:
        """plt.close(fig) is called even when savefig raises."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        with patch.object(fig, "savefig", side_effect=RuntimeError("boom")):
            with patch.object(plt, "close") as mock_close:
                try:
                    save_fig(fig)
                except RuntimeError:
                    pass
                mock_close.assert_called_once_with(fig)

    def test_custom_dpi(self) -> None:
        """Higher DPI produces a larger PNG buffer."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        buf_low = save_fig(fig)

        fig2, ax2 = plt.subplots(figsize=(2, 2))
        ax2.plot([0, 1], [0, 1])
        buf_high = save_fig(fig2, dpi=300)

        assert len(buf_high.getvalue()) > len(buf_low.getvalue())
