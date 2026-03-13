"""Shared chart utilities for matplotlib figure handling.

Provides ``save_fig()`` — a canonical function to serialize a matplotlib
Figure into an in-memory PNG BytesIO, with guaranteed ``plt.close()``
in a try/finally block.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_fig(fig: Figure, *, dpi: int = 150) -> io.BytesIO:
    """Save *fig* to a BytesIO PNG buffer and close the figure.

    Args:
        fig: Matplotlib Figure to render.
        dpi: Resolution in dots per inch.

    Returns:
        BytesIO buffer positioned at the start, containing PNG data.
    """
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    buf.seek(0)
    return buf
