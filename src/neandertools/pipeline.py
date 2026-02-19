"""Asteroid cutout GIF pipeline.

Orchestrates the full workflow: ephemeris query, image search,
cutout extraction, and GIF assembly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.time import Time

from .trackbuilder import query_ephemeris, calculate_polygons
from .imagefinder import find_overlapping_images, interpolate_position
from .butler import ButlerCutoutService
from .visualization import cutouts_gif

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
        Filter bands to search.  ``None`` defaults to ``["g", "r", "i"]``.
    step : str
        Ephemeris time step (default ``"4h"``).
    cutout_size : int
        Cutout side length in pixels (default ``100``).
    polygon_interval_days : float
        Max duration of each search polygon in days (default ``3.0``).
    polygon_widening_arcsec : float
        Width of the search polygon on each side of the track (default ``120.0``).
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
        step: str = "4h",
        cutout_size: int = 100,
        polygon_interval_days: float = 3.0,
        polygon_widening_arcsec: float = 120.0,
    ) -> None:
        self.target = target
        self.start = start
        self.end = end
        self.dr = dr
        self.collection = collection
        self.target_type = target_type
        self.location = location
        self.bands = bands or ["g", "r", "i"]
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
        frame_duration_ms: int = 300,
        warp_common_grid: bool = True,
        show_ne_indicator: bool = False,
        dpi: int = 100,
    ) -> Path:
        """Execute the full pipeline and write an animated GIF.

        Parameters
        ----------
        output_path : str or Path
            Output GIF file path.
        frame_duration_ms : int
            Duration of each GIF frame in milliseconds.
        warp_common_grid : bool
            If ``True``, warp all cutouts onto a common sky grid so the
            asteroid appears to move across a stable background.
        show_ne_indicator : bool
            If ``True``, draw a North/East indicator arrow on each frame.
        dpi : int
            Resolution of the rendered frames.

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
            warp_common_grid=warp_common_grid,
            show_ne_indicator=show_ne_indicator,
            dpi=dpi,
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
        svc = ButlerCutoutService(butler=butler, repo=self.dr, collections=self.collection)

        eph = self.ephemeris
        mjd_grid = eph["times"].tai.mjd

        visits = []
        detectors = []
        ra_values = []
        dec_values = []
        obs_times = []

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
                logger.warning("Cannot get visitInfo for visit=%s det=%s: %s", visit_id, detector_id, exc)
                continue

            t_mid = visit_info.date.toAstropy()
            if t_mid.scale != "tai":
                t_mid = t_mid.tai

            # Interpolate asteroid position at observation midpoint
            ra_interp, dec_interp = interpolate_position(
                t_mid.mjd, mjd_grid, eph["ra_deg"], eph["dec_deg"],
            )

            visits.append(visit_id)
            detectors.append(detector_id)
            ra_values.append(ra_interp)
            dec_values.append(dec_interp)
            obs_times.append({"time": t_mid, "band": band})

        if not visits:
            return

        logger.info("Extracting %d cutouts (size=%dpx)...", len(visits), self.cutout_size)
        raw_cutouts = svc.cutout(
            ra=ra_values,
            dec=dec_values,
            visit=visits,
            detector=detectors,
            h=self.cutout_size,
            w=self.cutout_size,
        )

        # Pair cutouts with metadata and sort by observation time
        paired = list(zip(raw_cutouts, obs_times))
        paired.sort(key=lambda x: x[1]["time"].mjd)

        self.cutouts = [p[0] for p in paired]
        self.frame_metadata = [p[1] for p in paired]
        logger.info("Extracted %d cutouts", len(self.cutouts))

    def _create_gif(
        self,
        output_path: Union[str, Path],
        frame_duration_ms: int,
        warp_common_grid: bool,
        show_ne_indicator: bool,
        dpi: int,
    ) -> Path:
        """Step 5: Assemble cutouts into an animated GIF."""
        titles = [
            f"{meta['band']}-band  {meta['time'].utc.iso[:16]}"
            for meta in self.frame_metadata
        ]

        logger.info("Creating GIF with %d frames -> %s", len(self.cutouts), output_path)
        gif_path = cutouts_gif(
            images=self.cutouts,
            output_path=output_path,
            titles=titles,
            warp_common_grid=warp_common_grid,
            show_ne_indicator=show_ne_indicator,
            frame_duration_ms=frame_duration_ms,
            dpi=dpi,
        )
        logger.info("GIF saved: %s", gif_path)
        return gif_path
