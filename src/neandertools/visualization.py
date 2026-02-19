"""Visualization helpers for image collections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def cutouts_grid(
    images: Sequence[Any],
    ncols: int = 5,
    titles: Sequence[str] | None = None,
    figsize_per_cell: tuple[float, float] = (3.2, 3.2),
    qmin: float = 0.0,
    qmax: float = 0.99,
    add_colorbar: bool = False,
    cmap: str = "gray_r",
    show: bool = True,
):
    """Display images in a grid with linear quantile normalization.

    Parameters
    ----------
    images : sequence
        Sequence of image-like objects. Supported forms are LSST-like objects
        exposing ``obj.image.array`` and array-like objects exposing ``obj.array``.
    ncols : int, optional
        Number of columns in the grid.
    titles : sequence of str, optional
        Optional per-image titles.
    figsize_per_cell : tuple of float, optional
        Width and height per subplot cell.
    qmin : float, optional
        Lower quantile used for ``vmin`` (NaN-aware).
    qmax : float, optional
        Upper quantile used for ``vmax`` (NaN-aware).
    add_colorbar : bool, optional
        If ``True``, draw one colorbar per subplot.
    cmap : str, optional
        Matplotlib colormap name.
    show : bool, optional
        If ``True``, call ``plt.show()`` before returning.

    Returns
    -------
    tuple
        ``(fig, axes)`` from matplotlib.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images provided.")
    if not (0.0 <= qmin <= 1.0 and 0.0 <= qmax <= 1.0):
        raise ValueError("qmin and qmax must be in [0, 1]")
    if qmax < qmin:
        raise ValueError("qmax must be >= qmin")

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )

    for i, obj in enumerate(images):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        if hasattr(obj, "image"):
            arr = np.asarray(obj.image.array)
        else:
            arr = np.asarray(obj.array)

        vmin = np.nanquantile(arr, qmin)
        vmax = np.nanquantile(arr, qmax)
        if vmax <= vmin:
            vmax = vmin + 1e-12

        im = ax.imshow(
            arr,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        if titles is not None:
            ax.set_title(titles[i], fontsize=10)

        if add_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes
