"""Asteroid cutout GIF pipeline.

Orchestrates the full workflow: ephemeris query, image search,
cutout extraction, and GIF assembly.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .trackbuilder import query_ephemeris, calculate_polygons
from .imagefinder import find_overlapping_images, interpolate_position, create_cutout
from .imagebuilder import get_exposure, cutout_to_png, create_gif, normalize_cutouts

logger = logging.getLogger(__name__)


class AsteroidCutoutPipeline:
    """Generate animated GIFs of asteroid cutouts from Rubin/LSST data.

    Parameters
    ----------
    target : str
        Asteroid name or designation (e.g. ``"Ceres"``, ``"2024 AA"``).
    start : str
        Start date, e.g. ``"2024-11-01"``.
    end : str
        End date, e.g. ``"2024-11-15"``.
    dr : str
        Butler data release label (e.g. ``"dp1"``).
    collection : str
        Butler collection name (e.g. ``"LSSTComCam/DP1"``).
    target_type : str
        Horizons ``id_type`` (default ``"smallbody"``).
    location : str
        Observer location code (default ``"X05"`` for Rubin).
    bands : list of str or None
        Filter bands to search.  ``None`` defaults to all ugrizy.
    step : str
        Ephemeris time step (default ``"12h"``).
    cutout_size : int
        Cutout side length in pixels (default ``100``).
    polygon_interval_days : float
        Max duration of each search polygon in days (default ``3.0``).
    polygon_widening_arcsec : float
        Width of the search polygon on each side of the track (default ``2.0``).
    """

    def __init__(
        self,
        target: str,
        start: str,
        end: str,
        dr: str = "dp1",
        collection: str = "LSSTComCam/DP1",
        target_type: str = "smallbody",
        location: str = "X05",
        bands: Optional[list[str]] = None,
        step: str = "12h",
        cutout_size: int = 100,
        polygon_interval_days: float = 3.0,
        polygon_widening_arcsec: float = 2.0,
    ) -> None:
        self.target = target
        self.start = start
        self.end = end
        self.dr = dr
        self.collection = collection
        self.target_type = target_type
        self.location = location
        self.bands = bands or ["u", "g", "r", "i", "z", "y"]
        self.step = step
        self.cutout_size = cutout_size
        self.polygon_interval_days = polygon_interval_days
        self.polygon_widening_arcsec = polygon_widening_arcsec

        # Populated by run()
        self.ephemeris: Optional[dict] = None
        self.polygons: Optional[list[dict]] = None
        self.cutouts: list = []
        self.frame_metadata: list[dict] = []

    def run(
        self,
        output_path: Union[str, Path] = "asteroid.gif",
        frame_duration_ms: int = 500,
        match_background: bool = True,
        match_noise: bool = False,
    ) -> Path:
        """Execute the full pipeline and write an animated GIF.

        Parameters
        ----------
        output_path : str or Path
            Output GIF file path.
        frame_duration_ms : int
            Duration of each GIF frame in milliseconds.
        match_background : bool
            If ``True``, subtract per-cutout background so all frames share
            the same zero level. Strongly recommended for GIFs.
        match_noise : bool
            If ``True``, also divide by per-cutout noise RMS so all frames
            share the same noise scale (SNR-like display).

        Returns
        -------
        Path
            Path to the created GIF file.
        """
        self._query_ephemeris()
        self._build_polygons()
        self._find_images()
        self._extract_cutouts()

        if not self.cutouts:
            logger.warning("No cutouts produced — no matching images found.")
            return Path(output_path)

        gif_path = self._create_gif(
            output_path=output_path,
            frame_duration_ms=frame_duration_ms,
            match_background=match_background,
            match_noise=match_noise,
        )
        return gif_path

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _query_ephemeris(self) -> None:
        """Step 1: Query JPL Horizons for the ephemeris."""
        logger.info(
            "Querying Horizons for %s (%s — %s, step=%s)",
            self.target, self.start, self.end, self.step,
        )
        self.ephemeris = query_ephemeris(
            target=self.target,
            target_type=self.target_type,
            start=self.start,
            end=self.end,
            step=self.step,
            location=self.location,
        )
        n = len(self.ephemeris["ra_deg"])
        logger.info("Ephemeris: %d points", n)

    def _build_polygons(self) -> None:
        """Step 2: Build search polygons from the ephemeris track."""
        eph = self.ephemeris
        self.polygons = calculate_polygons(
            times=eph["times"],
            ra_deg=eph["ra_deg"],
            dec_deg=eph["dec_deg"],
            time_interval_days=self.polygon_interval_days,
            widening_arcsec=self.polygon_widening_arcsec,
        )
        logger.info("Created %d search polygons", len(self.polygons))

    def _find_images(self) -> None:
        """Step 3: Find overlapping visit_image datasets via Butler."""
        logger.info(
            "Searching for images in bands=%s, dr=%s, collection=%s",
            self.bands, self.dr, self.collection,
        )
        self._dataset_refs = find_overlapping_images(
            polygons=self.polygons,
            bands=self.bands,
            dr=self.dr,
            collection=self.collection,
        )
        logger.info("Found %d unique matching images", len(self._dataset_refs))

    def _extract_cutouts(self) -> None:
        """Step 4: Load images, interpolate positions, extract cutouts."""
        from lsst.daf.butler import Butler

        if not self._dataset_refs:
            return

        butler = Butler(self.dr, collections=self.collection)

        eph = self.ephemeris
        mjd_grid = eph["times"].tai.mjd

        paired = []

        for ref in self._dataset_refs:
            visit_id = ref.dataId["visit"]
            detector_id = ref.dataId["detector"]
            band = ref.dataId["band"]

            # Get observation midpoint (lightweight — no full image load)
            try:
                visit_info = butler.get(
                    "visit_image.visitInfo", visit=visit_id, detector=detector_id,
                )
            except Exception as exc:
                logger.warning(
                    "Cannot get visitInfo for visit=%s det=%s: %s",
                    visit_id, detector_id, exc,
                )
                continue

            t_mid = visit_info.date.toAstropy()
            if t_mid.scale != "tai":
                t_mid = t_mid.tai

            # Interpolate asteroid position at observation midpoint
            ra_interp, dec_interp = interpolate_position(
                t_mid.mjd, mjd_grid, eph["ra_deg"], eph["dec_deg"],
            )

            # Load full exposure
            exposure = get_exposure(butler, visit_id, detector_id)
            if exposure is None:
                continue

            # Extract cutout
            cutout = create_cutout(
                exposure, ra_interp, dec_interp, cutout_size_px=self.cutout_size,
            )
            if cutout is None:
                logger.warning(
                    "Target outside image for visit=%s det=%s", visit_id, detector_id,
                )
                continue

            paired.append((cutout, {"time": t_mid, "band": band}))

        # Sort by observation time
        paired.sort(key=lambda x: x[1]["time"].mjd)

        self.cutouts = [p[0] for p in paired]
        self.frame_metadata = [p[1] for p in paired]
        logger.info("Extracted %d cutouts", len(self.cutouts))

    def _create_gif(
        self,
        output_path: Union[str, Path],
        frame_duration_ms: int,
        match_background: bool = True,
        match_noise: bool = False,
    ) -> Path:
        """Step 5: Render cutouts to PNGs and assemble into a GIF."""
        output_path = Path(output_path)

        # Normalize all cutouts to a shared background/noise scale
        raw_arrays = [c.image.array for c in self.cutouts]
        if match_background or match_noise:
            norm_arrays, vmin, vmax = normalize_cutouts(
                raw_arrays,
                match_background=match_background,
                match_noise=match_noise,
            )
        else:
            norm_arrays = [None] * len(self.cutouts)
            vmin, vmax = None, None

        # Use a temp directory for intermediate PNGs
        tmpdir = tempfile.mkdtemp(prefix="neandertools_")

        for i, (cutout, meta) in enumerate(zip(self.cutouts, self.frame_metadata)):
            title = f"{meta['band']}-band  {meta['time'].utc.iso[:16]}"
            png_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            cutout_to_png(
                cutout, png_path, title=title,
                vmin=vmin, vmax=vmax,
                array_override=norm_arrays[i] if norm_arrays[i] is not None else None,
            )

        logger.info("Creating GIF with %d frames -> %s", len(self.cutouts), output_path)
        gif_path = create_gif(
            png_dir=tmpdir,
            output_path=output_path,
            duration=frame_duration_ms,
        )
        return gif_path
