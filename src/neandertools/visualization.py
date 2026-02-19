"""Visualization helpers for image collections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval


def cutouts_grid(
    images: Sequence[Any],
    ncols: int = 5,
    titles: Sequence[str] | None = None,
    figsize_per_cell: tuple[float, float] = (3.2, 3.2),
    zscale_contrast: float = 0.25,
    asinh_a: float = 0.1,
    add_colorbar: bool = False,
    cmap: str = "gray",
    show: bool = True,
):
    """Display images in a grid with zscale+asinh normalization.

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
    zscale_contrast : float, optional
        Contrast value passed to ``ZScaleInterval``.
    asinh_a : float, optional
        ``a`` parameter for ``AsinhStretch``.
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

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )

    interval = ZScaleInterval(contrast=zscale_contrast)
    stretch = AsinhStretch(a=asinh_a)

    for i, obj in enumerate(images):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        if hasattr(obj, "image"):
            arr = np.asarray(obj.image.array)
        else:
            arr = np.asarray(obj.array)

        vmin, vmax = interval.get_limits(arr)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=True)

        im = ax.imshow(
            arr,
            origin="lower",
            norm=norm,
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
