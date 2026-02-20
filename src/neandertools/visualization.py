"""Visualization helpers for image collections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize


def cutouts_grid(
    images: Sequence[Any],
    ncols: int = 5,
    titles: Sequence[str] | None = None,
    figsize_per_cell: tuple[float, float] = (3.2, 3.2),
    qmin: float = 0.0,
    qmax: float = 0.99,
    match_background: bool = True,
    match_noise: bool = False,
    sigma_clip: float = 3.0,
    sigma_clip_iters: int = 5,
    warp_common_grid: bool = False,
    warp_shape: tuple[int, int] | None = None,
    warp_pixel_scale_arcsec: float | None = None,
    show_ne_indicator: bool = False,
    ne_indicator_scale: float = 0.10,
    add_colorbar: bool = False,
    cmap: str = "gray",
    show: bool = True,
    auto_vlims: bool = False,
    contrast: float = 0.1
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
    match_background : bool, optional
        If ``True``, subtract a robust sigma-clipped background estimate from
        each cutout before plotting.
    match_noise : bool, optional
        If ``True``, divide each cutout by its robust background RMS estimate
        after background subtraction.
    sigma_clip : float, optional
        Sigma threshold used for iterative clipping when estimating per-cutout
        background/noise.
    sigma_clip_iters : int, optional
        Maximum number of sigma-clipping iterations.
    warp_common_grid : bool, optional
        If ``True``, warp all cutouts onto a common sky grid with +x in the
        increasing Right Ascension direction.
    warp_shape : tuple of int, optional
        Output ``(height, width)`` for warped cutouts. Defaults to the first
        image shape when ``warp_common_grid=True``.
    warp_pixel_scale_arcsec : float, optional
        Pixel scale of the common grid in arcsec/pixel. If omitted, a robust
        scale is estimated from input WCS objects.
    show_ne_indicator : bool, optional
        Draw a small North/East indicator in the top-right of each panel.
    ne_indicator_scale : float, optional
        Indicator arrow length as a fraction of axis span.
    add_colorbar : bool, optional
        If ``True``, draw one colorbar per subplot.
    cmap : str, optional
        Matplotlib colormap name.
    show : bool, optional
        If ``True``, call ``plt.show()`` before returning.
    auto_vlims : bool, optional
        If ``True``, automatically adjust display limits. Overrides the qmin and qmax parameters if True.
    contrast : float, optional
        Contrast parameter for automatic display limits when ``auto_vlims=True``. This is passed to ``astropy.visualization.ZScaleInterval`` and controls the aggressiveness of the scaling; higher values result in a smaller range.

    Returns
    -------
    tuple
        ``(fig, axes)`` from matplotlib.
    """
    n = len(images)
    arrays, vmins, vmaxs, extents, ne_vectors = _prepare_cutouts_for_display(
        images=images,
        qmin=qmin,
        qmax=qmax,
        match_background=match_background,
        match_noise=match_noise,
        sigma_clip=sigma_clip,
        sigma_clip_iters=sigma_clip_iters,
        warp_common_grid=warp_common_grid,
        warp_shape=warp_shape,
        warp_pixel_scale_arcsec=warp_pixel_scale_arcsec,
        auto_vlims=auto_vlims,
        contrast=contrast,
    )
    if ne_indicator_scale <= 0:
        raise ValueError("ne_indicator_scale must be > 0")

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )
    cmap_obj = _cmap_with_black_nan(cmap)

    for i, arr in enumerate(arrays):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        im = ax.imshow(
            arr,
            origin="lower",
            vmin=vmins[i],
            vmax=vmaxs[i],
            cmap=cmap_obj,
            interpolation="nearest",
            extent=extents[i],
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Delta x (arcsec)")
        if c == 0:
            ax.set_ylabel("Delta y (arcsec)")

        if titles is not None:
            ax.set_title(titles[i], fontsize=10)

        if add_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if warp_common_grid:
            ax.set_xlabel("Delta R.A. (arcsec)")
            if c == 0:
                ax.set_ylabel("Delta Dec. (arcsec)")
        if show_ne_indicator:
            _draw_ne_indicator(ax, ne_vectors[i], scale_frac=ne_indicator_scale)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def cutouts_gif(
    images: Sequence[Any],
    output_path: str | Path = "cutouts.gif",
    titles: Sequence[str] | None = None,
    figsize: tuple[float, float] = (5.0, 5.0),
    qmin: float = 0.0,
    qmax: float = 0.99,
    match_background: bool = True,
    match_noise: bool = False,
    sigma_clip: float = 3.0,
    sigma_clip_iters: int = 5,
    warp_common_grid: bool = False,
    warp_shape: tuple[int, int] | None = None,
    warp_pixel_scale_arcsec: float | None = None,
    show_ne_indicator: bool = False,
    ne_indicator_scale: float = 0.10,
    cmap: str = "gray",
    frame_duration_ms: int = 300,
    dpi: int = 100,
    title_fontsize: float = 12.0,
    show: bool = False,
    auto_vlims: bool = False,
    contrast: float = 0.1,
) -> Path:
    """Save cutouts as an animated GIF.

    Parameters are analogous to ``cutouts_grid``; each image is rendered as one
    frame of the output GIF. When ``titles`` is omitted, metadata-derived
    two-line titles are used.

    Returns
    -------
    pathlib.Path
        Path to the created GIF file.
    """
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be > 0")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    if title_fontsize <= 0:
        raise ValueError("title_fontsize must be > 0")

    arrays, vmins, vmaxs, extents, ne_vectors = _prepare_cutouts_for_display(
        images=images,
        qmin=qmin,
        qmax=qmax,
        match_background=match_background,
        match_noise=match_noise,
        sigma_clip=sigma_clip,
        sigma_clip_iters=sigma_clip_iters,
        warp_common_grid=warp_common_grid,
        warp_shape=warp_shape,
        warp_pixel_scale_arcsec=warp_pixel_scale_arcsec,
        auto_vlims=auto_vlims,
        contrast=contrast,
    )
    if ne_indicator_scale <= 0:
        raise ValueError("ne_indicator_scale must be > 0")

    from matplotlib.animation import PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=True)
    cmap_obj = _cmap_with_black_nan(cmap)
    im = ax.imshow(
        arrays[0],
        origin="lower",
        vmin=vmins[0],
        vmax=vmaxs[0],
        cmap=cmap_obj,
        interpolation="nearest",
        extent=extents[0],
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Delta x (arcsec)")
    ax.set_ylabel("Delta y (arcsec)")
    if warp_common_grid:
        ax.set_xlabel("Delta R.A. (arcsec)")
        ax.set_ylabel("Delta Dec. (arcsec)")
    ne_artists: list[Any] = []
    if show_ne_indicator:
        ne_artists = _draw_ne_indicator(ax, ne_vectors[0], scale_frac=ne_indicator_scale)

    if titles is None:
        auto_titles = [_build_cutout_metadata_title(obj) for obj in images]
    else:
        auto_titles = [str(t) for t in titles]

    frame_title = ax.set_title("", fontsize=title_fontsize)
    fps = 1000.0 / float(frame_duration_ms)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=dpi):
        for i, arr in enumerate(arrays):
            im.set_data(arr)
            im.set_clim(vmins[i], vmaxs[i])
            im.set_extent(extents[i])
            ax.set_xlim(extents[i][0], extents[i][1])
            ax.set_ylim(extents[i][2], extents[i][3])
            if show_ne_indicator:
                for artist in ne_artists:
                    artist.remove()
                ne_artists = _draw_ne_indicator(ax, ne_vectors[i], scale_frac=ne_indicator_scale)
            title_text = auto_titles[i]
            if title_text:
                frame_title.set_text(title_text)
            elif len(arrays) > 1:
                frame_title.set_text(f"Frame {i + 1}/{len(arrays)}")
            writer.grab_frame()

    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _prepare_cutouts_for_display(
    *,
    images: Sequence[Any],
    qmin: float,
    qmax: float,
    match_background: bool,
    match_noise: bool,
    sigma_clip: float,
    sigma_clip_iters: int,
    warp_common_grid: bool,
    warp_shape: tuple[int, int] | None,
    warp_pixel_scale_arcsec: float | None,
    auto_vlims: bool,
    contrast: float,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[float],
    list[tuple[float, float, float, float]],
    list[tuple[np.ndarray, np.ndarray] | None],
]:
    n = len(images)
    if n == 0:
        raise ValueError("No images provided.")
    if not (0.0 <= qmin <= 1.0 and 0.0 <= qmax <= 1.0):
        raise ValueError("qmin and qmax must be in [0, 1]")
    if qmax < qmin and not auto_vlims:
        raise ValueError("qmax must be >= qmin")
    if sigma_clip <= 0:
        raise ValueError("sigma_clip must be > 0")
    if sigma_clip_iters < 1:
        raise ValueError("sigma_clip_iters must be >= 1")
    if warp_shape is not None and (warp_shape[0] <= 0 or warp_shape[1] <= 0):
        raise ValueError("warp_shape dimensions must be > 0")
    if warp_pixel_scale_arcsec is not None and warp_pixel_scale_arcsec <= 0:
        raise ValueError("warp_pixel_scale_arcsec must be > 0")

    arrays = []
    image_info = []
    for obj in images:
        arr = _extract_image_array(obj)
        arrays.append(arr)
        wcs = obj.getWcs() if hasattr(obj, "getWcs") else None
        x0 = 0.0
        y0 = 0.0
        if hasattr(obj, "getBBox"):
            try:
                bbox = obj.getBBox()
                x0 = float(bbox.getMinX())
                y0 = float(bbox.getMinY())
            except Exception:
                pass
        image_info.append({"wcs": wcs, "x0": x0, "y0": y0})

    extents: list[tuple[float, float, float, float]] = []
    ne_vectors: list[tuple[np.ndarray, np.ndarray] | None] = []
    if warp_common_grid:
        if any(info["wcs"] is None for info in image_info):
            raise ValueError("warp_common_grid=True requires WCS for all input cutouts.")
        arrays, extent = _warp_to_common_radec_grid(
            arrays=arrays,
            image_info=image_info,
            warp_shape=warp_shape,
            warp_pixel_scale_arcsec=warp_pixel_scale_arcsec,
        )
        extents = [extent] * n
        ne_vectors = [(np.array([1.0, 0.0]), np.array([0.0, 1.0]))] * n
    else:
        for arr, info in zip(arrays, image_info):
            extent_i, ne_i = _estimate_nonwarp_extent_and_ne(arr, info)
            extents.append(extent_i)
            ne_vectors.append(ne_i)

    proc_arrays = []
    for arr in arrays:
        if match_background or match_noise:
            bg, rms = _sigma_clipped_bg_rms(arr, sigma=sigma_clip, maxiters=sigma_clip_iters)
            arr_proc = arr.astype(np.float32, copy=True)
            if match_background:
                arr_proc = arr_proc - bg
            if match_noise:
                arr_proc = arr_proc / max(rms, 1e-12)
            proc_arrays.append(arr_proc)
        else:
            proc_arrays.append(arr)

    shared_scale = match_background or match_noise
    vmins = []
    vmaxs = []
    if shared_scale:
        finite_parts = [arr[np.isfinite(arr)] for arr in proc_arrays if np.any(np.isfinite(arr))]
        if not finite_parts:
            raise ValueError("No finite pixels available to determine display scale.")
        all_values = np.concatenate(finite_parts)
        
        if auto_vlims:
            norm = ZScaleInterval(contrast=contrast, krej=3).get_limits(all_values)
            vmins = [norm[0]] * n
            vmaxs = [norm[1]] * n
        else:
            shared_vmin = float(np.quantile(all_values, qmin))
            shared_vmax = float(np.quantile(all_values, qmax))
            if shared_vmax <= shared_vmin:
                shared_vmax = shared_vmin + 1e-12
            vmins = [shared_vmin] * n
            vmaxs = [shared_vmax] * n
        
    else:
        for arr in proc_arrays:
            vmin = float(np.nanquantile(arr, qmin))
            vmax = float(np.nanquantile(arr, qmax))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            vmins.append(vmin)
            vmaxs.append(vmax)

    print(vmins, vmaxs)

    return proc_arrays, vmins, vmaxs, extents, ne_vectors


def _sigma_clipped_bg_rms(arr: np.ndarray, sigma: float, maxiters: int) -> tuple[float, float]:
    values = np.asarray(arr, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    clipped = values
    for _ in range(maxiters):
        med = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - med)))
        rms = 1.4826 * mad
        if not np.isfinite(rms) or rms <= 0:
            break
        keep = np.abs(clipped - med) <= sigma * rms
        if keep.all() or keep.sum() == 0:
            break
        clipped = clipped[keep]

    bg = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped - bg)))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.std(clipped))
    if not np.isfinite(rms) or rms <= 0:
        rms = 1.0
    return bg, rms


def _extract_image_array(obj: Any) -> np.ndarray:
    if hasattr(obj, "image") and hasattr(obj.image, "array"):
        return np.asarray(obj.image.array)
    if hasattr(obj, "array"):
        return np.asarray(obj.array)
    if hasattr(obj, "getArray"):
        return np.asarray(obj.getArray())
    if hasattr(obj, "getImage"):
        img = obj.getImage()
        if hasattr(img, "getArray"):
            return np.asarray(img.getArray())
    raise ValueError("Unsupported image object: could not find array data.")


def _build_cutout_metadata_title(obj: Any) -> str:
    visit = _extract_visit_id(obj)
    detector = _extract_detector_id(obj)
    band = _extract_band(obj)
    midpoint = _extract_midpoint_time_iso(obj)
    line1 = f"visit={visit} det={detector} band={band}"
    line2 = f"mid={midpoint}"
    return f"{line1}\n{line2}"


def _extract_visit_id(obj: Any) -> str:
    try:
        info = obj.getInfo()
        vi = info.getVisitInfo() if info is not None else None
        if vi is not None and hasattr(vi, "getId"):
            visit_id = vi.getId()
            if visit_id is not None:
                return str(visit_id)
    except Exception:
        pass
    md = _extract_metadata(obj)
    for key in ("VISIT", "visit", "EXPID", "expId"):
        if key in md:
            return str(md[key])
    return "?"


def _extract_detector_id(obj: Any) -> str:
    try:
        if hasattr(obj, "getDetector"):
            det = obj.getDetector()
            if det is not None and hasattr(det, "getId"):
                return str(det.getId())
    except Exception:
        pass
    md = _extract_metadata(obj)
    for key in ("DETECTOR", "detector", "CCDNUM"):
        if key in md:
            return str(md[key])
    return "?"


def _extract_band(obj: Any) -> str:
    try:
        if hasattr(obj, "getFilter"):
            filt = obj.getFilter()
            if filt is not None:
                if hasattr(filt, "bandLabel") and filt.bandLabel:
                    return str(filt.bandLabel)
                if hasattr(filt, "physicalLabel") and filt.physicalLabel:
                    return str(filt.physicalLabel)
                return str(filt)
    except Exception:
        pass
    md = _extract_metadata(obj)
    for key in ("BAND", "band", "FILTER", "filter"):
        if key in md:
            return str(md[key])
    return "?"


def _extract_midpoint_time_iso(obj: Any) -> str:
    try:
        from astropy.time import TimeDelta
        import astropy.units as u

        info = obj.getInfo()
        vi = info.getVisitInfo() if info is not None else None
        if vi is not None and hasattr(vi, "getDate") and vi.getDate() is not None:
            t0 = vi.getDate().toAstropy()
            exp_time = float(vi.getExposureTime()) if hasattr(vi, "getExposureTime") else 0.0
            tm = t0 + TimeDelta(0.5 * exp_time * u.s)
            return tm.utc.isot
    except Exception:
        pass
    md = _extract_metadata(obj)
    for key in ("DATE-AVG", "DATE-OBS", "MJD-MID", "MJD-OBS"):
        if key in md:
            return str(md[key])
    return "?"


def _extract_metadata(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        if hasattr(obj, "getMetadata"):
            md = obj.getMetadata()
            if md is not None:
                for name in md.names():
                    try:
                        out[name] = md.getScalar(name)
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _cmap_with_black_nan(cmap: str) -> Any:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    return cmap_obj


def _estimate_nonwarp_extent_and_ne(
    arr: np.ndarray, info: dict[str, Any]
) -> tuple[tuple[float, float, float, float], tuple[np.ndarray, np.ndarray] | None]:
    h, w = arr.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    wcs = info["wcs"]
    x0 = float(info["x0"])
    y0 = float(info["y0"])

    if wcs is None:
        # Fallback: assume 1 arcsec/pixel and cardinal orientation.
        scale = 1.0
        x_arcsec = (np.arange(w, dtype=np.float64) - cx) * scale
        y_arcsec = (np.arange(h, dtype=np.float64) - cy) * scale
        extent = (float(x_arcsec[0]), float(x_arcsec[-1]), float(y_arcsec[0]), float(y_arcsec[-1]))
        return extent, (np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    xc = x0 + cx
    yc = y0 + cy
    ra_c, dec_c = wcs.pixelToSkyArray(np.array([xc]), np.array([yc]), degrees=True)
    ra_x, dec_x = wcs.pixelToSkyArray(np.array([xc + 1.0]), np.array([yc]), degrees=True)
    ra_y, dec_y = wcs.pixelToSkyArray(np.array([xc]), np.array([yc + 1.0]), degrees=True)
    cos_dec = max(abs(np.cos(np.deg2rad(float(dec_c[0])))), 1e-6)
    a11 = _wrap_angle_diff_deg(float(ra_x[0]) - float(ra_c[0])) * cos_dec * 3600.0
    a21 = (float(dec_x[0]) - float(dec_c[0])) * 3600.0
    a12 = _wrap_angle_diff_deg(float(ra_y[0]) - float(ra_c[0])) * cos_dec * 3600.0
    a22 = (float(dec_y[0]) - float(dec_c[0])) * 3600.0
    jac = np.array([[a11, a12], [a21, a22]], dtype=np.float64)  # pix -> (E,N) arcsec

    sx = float(np.hypot(a11, a21))
    sy = float(np.hypot(a12, a22))
    scale = float(np.median([v for v in (sx, sy) if np.isfinite(v) and v > 0])) if (sx > 0 or sy > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    x_arcsec = (np.arange(w, dtype=np.float64) - cx) * scale
    y_arcsec = (np.arange(h, dtype=np.float64) - cy) * scale
    extent = (float(x_arcsec[0]), float(x_arcsec[-1]), float(y_arcsec[0]), float(y_arcsec[-1]))

    ne_vec: tuple[np.ndarray, np.ndarray] | None = None
    try:
        inv = np.linalg.inv(jac)  # (E,N) -> pix
        p_e = inv @ np.array([1.0, 0.0])
        p_n = inv @ np.array([0.0, 1.0])
        # Convert pix vectors into displayed arcsec vectors (scaled pixel axes).
        d_e = scale * p_e
        d_n = scale * p_n
        if np.all(np.isfinite(d_e)) and np.all(np.isfinite(d_n)):
            ne_vec = (d_e.astype(np.float64), d_n.astype(np.float64))
    except Exception:
        ne_vec = None
    return extent, ne_vec


def _draw_ne_indicator(
    ax: Any,
    ne_vec: tuple[np.ndarray, np.ndarray] | None,
    *,
    scale_frac: float,
) -> list[Any]:
    artists: list[Any] = []
    if ne_vec is None:
        return artists
    v_e, v_n = ne_vec
    norm_e = float(np.hypot(v_e[0], v_e[1]))
    norm_n = float(np.hypot(v_n[0], v_n[1]))
    if norm_e <= 0 or norm_n <= 0:
        return artists
    v_e = v_e / norm_e
    v_n = v_n / norm_n

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    length = scale_frac * min(abs(dx), abs(dy))
    bx = x1 - 0.12 * dx
    by = y1 - 0.12 * dy

    e_end = (bx + length * v_e[0], by + length * v_e[1])
    n_end = (bx + length * v_n[0], by + length * v_n[1])
    ann_e = ax.annotate(
        "",
        xy=e_end,
        xytext=(bx, by),
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "yellow"},
        zorder=5,
    )
    ann_n = ax.annotate(
        "",
        xy=n_end,
        xytext=(bx, by),
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "cyan"},
        zorder=5,
    )
    txt_e = ax.text(e_end[0], e_end[1], "E", color="yellow", fontsize=7, ha="left", va="bottom", zorder=6)
    txt_n = ax.text(n_end[0], n_end[1], "N", color="cyan", fontsize=7, ha="left", va="bottom", zorder=6)
    artists.extend([ann_e, ann_n, txt_e, txt_n])
    return artists


def _warp_to_common_radec_grid(
    *,
    arrays: list[np.ndarray],
    image_info: list[dict[str, Any]],
    warp_shape: tuple[int, int] | None,
    warp_pixel_scale_arcsec: float | None,
) -> tuple[list[np.ndarray], tuple[float, float, float, float]]:
    try:
        import lsst.afw.geom as afwGeom
        import lsst.afw.image as afwImage
        import lsst.afw.math as afwMath
        import lsst.geom as geom
    except Exception as e:
        raise ValueError("warp_common_grid=True requires LSST stack image/geom/math modules.") from e

    if warp_shape is None:
        out_h, out_w = arrays[0].shape
    else:
        out_h, out_w = warp_shape

    ra_centers: list[float] = []
    dec_centers: list[float] = []
    scale_samples_deg: list[float] = []

    for arr, info in zip(arrays, image_info):
        wcs = info["wcs"]
        x0 = info["x0"]
        y0 = info["y0"]
        h, w = arr.shape
        xc = x0 + (w - 1) / 2.0
        yc = y0 + (h - 1) / 2.0
        ra_c, dec_c = wcs.pixelToSkyArray(np.array([xc]), np.array([yc]), degrees=True)
        ra_centers.append(float(ra_c[0]))
        dec_centers.append(float(dec_c[0]))

        ra_x, dec_x = wcs.pixelToSkyArray(np.array([xc + 1.0]), np.array([yc]), degrees=True)
        ra_y, dec_y = wcs.pixelToSkyArray(np.array([xc]), np.array([yc + 1.0]), degrees=True)
        cos_dec = max(abs(np.cos(np.deg2rad(float(dec_c[0])))), 1e-6)
        dra_x = _wrap_angle_diff_deg(float(ra_x[0]) - float(ra_c[0])) * cos_dec
        ddec_x = float(dec_x[0]) - float(dec_c[0])
        dra_y = _wrap_angle_diff_deg(float(ra_y[0]) - float(ra_c[0])) * cos_dec
        ddec_y = float(dec_y[0]) - float(dec_c[0])
        sx = float(np.hypot(dra_x, ddec_x))
        sy = float(np.hypot(dra_y, ddec_y))
        if np.isfinite(sx) and sx > 0:
            scale_samples_deg.append(sx)
        if np.isfinite(sy) and sy > 0:
            scale_samples_deg.append(sy)

    ra0 = _circular_mean_deg(ra_centers)
    dec0 = float(np.mean(dec_centers))

    if warp_pixel_scale_arcsec is None:
        pixel_scale_deg = float(np.median(scale_samples_deg)) if scale_samples_deg else (1.0 / 3600.0)
    else:
        pixel_scale_deg = float(warp_pixel_scale_arcsec) / 3600.0

    # Define common target TAN WCS with +x toward increasing RA and +y toward
    # increasing Dec.
    cd = np.array(
        [
            [pixel_scale_deg / max(abs(np.cos(np.deg2rad(dec0))), 1e-6), 0.0],
            [0.0, pixel_scale_deg],
        ],
        dtype=np.float64,
    )
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0
    dest_wcs = afwGeom.makeSkyWcs(
        crpix=geom.Point2D(cx, cy),
        crval=geom.SpherePoint(ra0 * geom.degrees, dec0 * geom.degrees),
        cdMatrix=cd,
    )

    warp_ctrl = afwMath.WarpingControl("lanczos3")
    warped_arrays = []
    for arr, info in zip(arrays, image_info):
        src_wcs = info["wcs"]
        x0 = int(round(info["x0"]))
        y0 = int(round(info["y0"]))

        src_img = afwImage.ImageF(np.asarray(arr, dtype=np.float32))
        src_img.setXY0(geom.Point2I(x0, y0))

        dest_img = afwImage.ImageF(out_w, out_h)
        afwMath.warpImage(dest_img, dest_wcs, src_img, src_wcs, warp_ctrl, np.nan)
        warped_arrays.append(np.asarray(dest_img.array, dtype=np.float32))

    x_arcsec = (np.arange(out_w, dtype=np.float64) - cx) * (pixel_scale_deg * 3600.0)
    y_arcsec = (np.arange(out_h, dtype=np.float64) - cy) * (pixel_scale_deg * 3600.0)
    extent = (float(x_arcsec[0]), float(x_arcsec[-1]), float(y_arcsec[0]), float(y_arcsec[-1]))
    return warped_arrays, extent


def _circular_mean_deg(values_deg: Sequence[float]) -> float:
    values_rad = np.deg2rad(np.asarray(values_deg, dtype=np.float64))
    s = np.mean(np.sin(values_rad))
    c = np.mean(np.cos(values_rad))
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0)


def _wrap_angle_diff_deg(delta_deg: float) -> float:
    return (delta_deg + 180.0) % 360.0 - 180.0
