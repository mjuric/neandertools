"""Butler-backed cutout service."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from datetime import datetime, timezone
import multiprocessing as mp
import os
from typing import Any, Optional, Union

from astropy.time import Time
from lsst.daf.butler import Butler
from lsst.geom import Box2I, Point2I, SpherePoint, degrees
from lsst.sphgeom import LonLat, UnitVector3d


class ButlerCutoutService:
    """Generate image cutouts from an LSST Butler repository.

    This service fetches images by ``(visit, detector)`` data IDs and extracts
    cutouts centered either in pixel coordinates (``x``, ``y``) or sky
    coordinates (``ra``, ``dec`` converted through image WCS).

    The extraction path supports optional edge padding so the requested center
    remains at the center pixel of the returned cutout even near detector
    boundaries.
    """

    def __init__(
        self,
        butler: Any,
        *,
        repo: Optional[str] = None,
        collections: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """Create a cutout service bound to a Butler instance.

        Parameters
        ----------
        butler : object
            Butler-like object providing ``get(dataset_type, dataId=...)``.
        repo : str, optional
            Butler repo root, stored for optional multiprocessing mode.
        collections : str or sequence of str, optional
            Butler collections, stored for optional multiprocessing mode.
        """
        self._butler = butler
        self._repo = repo
        self._collections = _normalize_collections(collections) if collections is not None else None
        self._visit_detector_index_cache: dict[str, list[dict[str, Any]]] = {}

    def cutout(
        self,
        ra: Optional[Union[float, Sequence[float]]] = None,
        dec: Optional[Union[float, Sequence[float]]] = None,
        x: Optional[Union[float, Sequence[float]]] = None,
        y: Optional[Union[float, Sequence[float]]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        dataset_type: str = "visit_image",
        *,
        visit: Optional[Union[int, Sequence[int]]] = None,
        detector: Optional[Union[int, Sequence[int]]] = None,
        pad: bool = True,
        ncores: Optional[int] = None,
    ) -> list[Any]:
        """Return one cutout per requested ``(visit, detector)`` pair.

        Parameters
        ----------
        ra, dec : float or 1D sequence of float, optional
            Sky center(s) in degrees. Must be provided together and used with
            ``visit``/``detector``. Converted to pixel coordinates via image WCS.
        x, y : float or 1D sequence of float, optional
            Pixel center(s). Must be provided together and used with
            ``visit``/``detector``.
        h, w : int, optional
            Requested cutout height and width in pixels. If omitted, defaults
            to full image height/width.
        dataset_type : str, optional
            Butler dataset type to read. Defaults to ``"visit_image"``.
        visit, detector : int or 1D sequence of int
            Data IDs selecting source images. Both are required.
        pad : bool, optional
            If ``True`` (default), edge-overlapping requests are padded with
            ``NaN`` so output shape remains ``(h, w)`` and the requested center
            stays centered. If ``False``, requests are clipped to image bounds.
        ncores : int, optional
            Number of process cores used for extraction. If ``None``, choose
            ``min(cpu_count, n_items)``. If ``1``, execute serially in the
            current process. If ``>1``, use multiprocessing.

        Returns
        -------
        list
            List of extracted cutouts, one per broadcasted input row.

        Raises
        ------
        ValueError
            If argument combinations are invalid (for example missing paired
            coordinates or missing ``visit``/``detector``).
        """
        _validate_request(ra=ra, dec=dec, x=x, y=y, h=h, w=w, visit=visit, detector=detector)
        if ncores is not None and ncores < 1:
            raise ValueError("ncores must be >= 1")
        x_mode = _is_provided(x) or _is_provided(y)
        if x_mode:
            x_values = _as_list(x, "x")
            y_values = _as_list(y, "y")
            assert x_values is not None and y_values is not None
            n_centers = _max_len(x_values, y_values)
            x_values = _broadcast_to(x_values, n_centers, "x")
            y_values = _broadcast_to(y_values, n_centers, "y")
            ra_values = [None] * n_centers
            dec_values = [None] * n_centers
        else:
            ra_values = _as_list(ra, "ra")
            dec_values = _as_list(dec, "dec")
            assert ra_values is not None and dec_values is not None
            n_centers = _max_len(ra_values, dec_values)
            ra_values = _broadcast_to(ra_values, n_centers, "ra")
            dec_values = _broadcast_to(dec_values, n_centers, "dec")
            x_values = [None] * n_centers
            y_values = [None] * n_centers

        visit_values = _as_list(visit, "visit")
        detector_values = _as_list(detector, "detector")
        assert visit_values is not None and detector_values is not None
        n_items = max(len(visit_values), len(detector_values), n_centers)
        visit_values = _broadcast_to(visit_values, n_items, "visit")
        detector_values = _broadcast_to(detector_values, n_items, "detector")
        x_values = _broadcast_to(x_values, n_items, "x")
        y_values = _broadcast_to(y_values, n_items, "y")
        ra_values = _broadcast_to(ra_values, n_items, "ra")
        dec_values = _broadcast_to(dec_values, n_items, "dec")

        items = list(zip(visit_values, detector_values, x_values, y_values, ra_values, dec_values))
        workers = _resolve_ncores(ncores, len(items))
        if workers == 1:
            return [
                self._run_cutout_item(dataset_type=dataset_type, item=item, h=h, w=w, pad=pad)
                for item in items
            ]

        if self._repo is None or self._collections is None:
            raise ValueError(
                "ncores>1 requires service with repo/collections metadata; "
                "use cutouts_from_butler(...) to construct it."
            )
        try:
            process_ctx = mp.get_context("fork")
        except ValueError as e:
            raise ValueError("multiprocessing with fork is not supported on this platform.") from e
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=process_ctx,
                initializer=_init_process_cutout_worker,
                initargs=(self._repo, self._collections),
            ) as ex:
                return list(
                    ex.map(
                        _process_cutout_worker,
                        [(dataset_type, item, h, w, pad) for item in items],
                    )
                )
        except (PermissionError, NotImplementedError):
            # Restricted environments (e.g., sandboxed tests) may block process
            # primitives. Fall back to serial execution.
            return [
                self._run_cutout_item(dataset_type=dataset_type, item=item, h=h, w=w, pad=pad)
                for item in items
            ]

    def _run_cutout_item(
        self,
        *,
        dataset_type: str,
        item: tuple[Any, Any, Any, Any, Any, Any],
        h: Optional[int],
        w: Optional[int],
        pad: bool,
        butler: Optional[Any] = None,
    ) -> Any:
        v, d, xx, yy, rr, dd = item
        read_butler = butler if butler is not None else self._butler
        image = read_butler.get(dataset_type, dataId={"visit": int(v), "detector": int(d)})
        return self._extract_cutout(image, x=xx, y=yy, ra=rr, dec=dd, h=h, w=w, pad=pad)

    def find_visit_detector(
        self,
        ra: Union[float, Sequence[float]],
        dec: Union[float, Sequence[float]],
        t: Union[datetime, str, Time, Sequence[datetime], Sequence[str], Sequence[Time]],
        *,
        dataset_type: str = "visit_image",
    ) -> tuple[Any, Any]:
        """Find matching ``(visit, detector)`` entries for sky position and time.

        Parameters
        ----------
        ra, dec : float or 1D sequence of float
            Sky coordinates in degrees.
        t : datetime | str | astropy.time.Time | 1D sequence
            Query times. A match requires ``begin <= t < end``.
        dataset_type : str
            Dataset type used to build the visit/detector index.

        Returns
        -------
        tuple of numpy.ndarray
            ``(visit, detector)`` arrays containing all matches across inputs,
            flattened in input order.
        """
        import numpy as np

        ra_values = _as_list(ra, "ra")
        dec_values = _as_list(dec, "dec")
        t_values = _as_list(t, "t")
        assert ra_values is not None and dec_values is not None and t_values is not None

        n = max(len(ra_values), len(dec_values), len(t_values))
        ra_values = _broadcast_to(ra_values, n, "ra")
        dec_values = _broadcast_to(dec_values, n, "dec")
        t_values = _broadcast_to(t_values, n, "t")
        t_values = [_to_astropy_time(v) for v in t_values]

        index = self._get_visit_detector_index(dataset_type)
        visits: list[int] = []
        detectors: list[int] = []

        for ra_i, dec_i, t_i in zip(ra_values, dec_values, t_values):
            point = UnitVector3d(LonLat.fromDegrees(float(ra_i), float(dec_i)))
            for row in index:
                if "timespan" in row:
                    if not row["timespan"].contains(t_i):
                        continue
                else:
                    if not (row["begin"] <= t_i < row["end"]):
                        continue
                if row["region"].contains(point):
                    visits.append(row["visit"])
                    detectors.append(row["detector"])

        return np.asarray(visits, dtype=np.int64), np.asarray(detectors, dtype=np.int64)

    def _extract_cutout(
        self,
        image: Any,
        x: Optional[float],
        y: Optional[float],
        ra: Optional[float],
        dec: Optional[float],
        h: Optional[int],
        w: Optional[int],
        pad: bool,
    ) -> Any:
        if not hasattr(image, "getBBox") or not hasattr(image, "Factory"):
            return image

        bbox = self._as_box2i(image.getBBox())
        if x is not None and y is not None:
            x_center = float(x)
            y_center = float(y)
        else:
            assert ra is not None and dec is not None
            if not hasattr(image, "getWcs") or image.getWcs() is None:
                raise ValueError("Image does not have a WCS; cannot use ra/dec center")
            sky = SpherePoint(ra * degrees, dec * degrees)
            pixel = image.getWcs().skyToPixel(sky)
            x_center = float(pixel.getX())
            y_center = float(pixel.getY())

        w_i = int(w) if w is not None else int(bbox.getMaxX() - bbox.getMinX() + 1)
        h_i = int(h) if h is not None else int(bbox.getMaxY() - bbox.getMinY() + 1)

        x0 = int(round(x_center - (w_i - 1) / 2.0))
        y0 = int(round(y_center - (h_i - 1) / 2.0))
        x1 = x0 + w_i - 1
        y1 = y0 + h_i - 1

        requested_box = Box2I(Point2I(x0, y0), Point2I(x1, y1))
        if pad:
            try:
                cutout = image.Factory(image, requested_box)
                if self._matches_requested_box(cutout, requested_box, h_i, w_i):
                    return cutout
            except Exception:
                pass

            return self._extract_padded_cutout(
                image=image,
                bbox=bbox,
                requested_box=requested_box,
                h=h_i,
                w=w_i,
            )

        cutout_box = Box2I(Point2I(x0, y0), Point2I(x1, y1))
        try:
            cutout_box.clip(bbox)
            if cutout_box.isEmpty():
                return None
            
        except Exception:
            pass

        return image.Factory(image, cutout_box)

    def _extract_padded_cutout(self, image: Any, bbox: Any, requested_box: Any, h: int, w: int) -> Any:
        import numpy as np

        bbox = self._as_box2i(bbox)
        clipped_box = Box2I(
            Point2I(requested_box.getMinX(), requested_box.getMinY()),
            Point2I(requested_box.getMaxX(), requested_box.getMaxY()),
        )
        clipped_box.clip(bbox)
        if clipped_box.isEmpty():
            return None

        source_cutout = image.Factory(image, clipped_box)

        source_array = self._get_primary_array(source_cutout)
        if source_array is None:
            return source_cutout

        padded_dtype = np.result_type(source_array.dtype, np.float32)
        padded_array = np.full((h, w), np.nan, dtype=padded_dtype)

        x_off = clipped_box.getMinX() - requested_box.getMinX()
        y_off = clipped_box.getMinY() - requested_box.getMinY()
        padded_array[y_off : y_off + source_array.shape[0], x_off : x_off + source_array.shape[1]] = source_array

        # Reject frames where the target centre is in the WCS-valid but
        # pixel-trimmed border strip.  The LSST pipeline trims border pixels
        # without updating the WCS, so skyToPixel() can place the target
        # outside the actual pixel data.  We detect this by checking whether
        # the centre region of the padded array contains any finite pixel:
        # if it is entirely NaN the target has no real data and the frame is
        # useless.  Checking the array directly is more reliable than a
        # geometric bbox.contains() test, which can fail by one pixel due to
        # rounding when the target is right on the boundary.
        check_h = max(3, min(20, h // 5))
        check_w = max(3, min(20, w // 5))
        y0_c = (h - check_h) // 2
        x0_c = (w - check_w) // 2
        if not np.any(np.isfinite(padded_array[y0_c : y0_c + check_h, x0_c : x0_c + check_w])):
            return None

        try:
            if hasattr(image, "getWcs") and image.getWcs() is not None:
                padded = image.Factory(requested_box, image.getWcs())
            else:
                padded = image.Factory(requested_box)
            self._propagate_exposure_metadata(image, padded)
            padded_target = self._get_primary_array(padded)
            if padded_target is not None:
                padded_target[:] = padded_array
                return padded
        except Exception:
            pass

        return padded_array

    @staticmethod
    def _propagate_exposure_metadata(source: Any, target: Any) -> None:
        """Copy exposure metadata (especially VisitInfo) from source to target."""
        try:
            if hasattr(source, "getInfo") and hasattr(target, "setInfo"):
                source_info = source.getInfo()
                if source_info is not None:
                    try:
                        import lsst.afw.image as afwImage

                        target.setInfo(afwImage.ExposureInfo(source_info))
                    except Exception:
                        target.setInfo(source_info)
        except Exception:
            pass

    @staticmethod
    def _matches_requested_box(cutout: Any, requested_box: Any, h: int, w: int) -> bool:
        arr = ButlerCutoutService._get_primary_array(cutout)
        if arr is not None:
            return tuple(arr.shape) == (h, w)

        if hasattr(cutout, "getBBox"):
            try:
                box = cutout.getBBox()
                return (
                    box.getMinX() == requested_box.getMinX()
                    and box.getMinY() == requested_box.getMinY()
                    and box.getMaxX() == requested_box.getMaxX()
                    and box.getMaxY() == requested_box.getMaxY()
                )
            except Exception:
                pass
        return True

    @staticmethod
    def _get_primary_array(obj: Any) -> Any:
        try:
            if hasattr(obj, "getArray"):
                return obj.getArray()
        except Exception:
            pass
        try:
            if hasattr(obj, "getImage") and hasattr(obj.getImage(), "getArray"):
                return obj.getImage().getArray()
        except Exception:
            pass
        return None

    @staticmethod
    def _as_box2i(bbox: Any) -> Any:
        if isinstance(bbox, Box2I):
            return bbox
        try:
            return Box2I(
                Point2I(int(bbox.getMinX()), int(bbox.getMinY())),
                Point2I(int(bbox.getMaxX()), int(bbox.getMaxY())),
            )
        except Exception:
            return bbox

    def _get_visit_detector_index(self, dataset_type: str) -> list[dict[str, Any]]:
        cached = self._visit_detector_index_cache.get(dataset_type)
        if cached is not None:
            return cached

        out: list[dict[str, Any]] = []
        query = self._butler.registry.queryDataIds(["visit", "detector"], datasets=dataset_type).expanded()
        for coord in query:
            if not coord.hasRecords() or coord.region is None:
                continue
            timespan = coord.records["visit"].timespan
            if timespan is None or timespan.begin is None or timespan.end is None:
                continue
            out.append(
                {
                    "visit": int(coord["visit"]),
                    "detector": int(coord["detector"]),
                    "region": coord.region,
                    "timespan": timespan,
                }
            )

        self._visit_detector_index_cache[dataset_type] = out
        return out


def cutouts_from_butler(
    repo: str,
    *,
    collections: Union[str, list[str]],
    butler: Optional[Any] = None,
) -> ButlerCutoutService:
    if butler is None:
        butler = Butler(repo, collections=collections)
    return ButlerCutoutService(butler=butler, repo=repo, collections=collections)


def _validate_request(
    *,
    ra: Optional[Union[float, Sequence[float]]],
    dec: Optional[Union[float, Sequence[float]]],
    x: Optional[Union[float, Sequence[float]]],
    y: Optional[Union[float, Sequence[float]]],
    h: Optional[int],
    w: Optional[int],
    visit: Optional[Union[int, Sequence[int]]],
    detector: Optional[Union[int, Sequence[int]]],
) -> None:
    xy_mode = _is_provided(x) or _is_provided(y)
    sky_center_mode = _is_provided(ra) or _is_provided(dec)

    if xy_mode and sky_center_mode:
        raise ValueError("Use either (x, y) or (ra, dec) for cutout center, not both")
    if not xy_mode and not sky_center_mode:
        raise ValueError("Provide either (x, y) or (ra, dec) for cutout center")
    if xy_mode and (x is None or y is None):
        raise ValueError("Both x and y must be provided together")
    if sky_center_mode and (ra is None or dec is None):
        raise ValueError("Both ra and dec must be provided together")

    if h is not None and h <= 0:
        raise ValueError("h must be > 0")
    if w is not None and w <= 0:
        raise ValueError("w must be > 0")

    visit_mode = visit is not None or detector is not None

    if visit_mode and (visit is None or detector is None):
        raise ValueError("Both visit and detector must be provided together")
    if not visit_mode:
        raise ValueError("Both visit and detector must be provided together")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_provided(value: Any) -> bool:
    if value is None:
        return False
    if _is_sequence(value):
        return len(value) > 0
    return True


def _as_list(value: Any, name: str) -> Optional[list[Any]]:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, dict)):
        try:
            out = list(value)
        except TypeError:
            out = None
        if out is not None:
            if not out:
                raise ValueError(f"{name} cannot be empty")
            return out
    return [value]


def _max_len(*values: Any) -> int:
    return max(len(v) for v in values if v is not None)


def _broadcast_to(values: list[Any], n: int, name: str) -> list[Any]:
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    raise ValueError(f"{name} must have length 1 or {n}")


def _to_astropy_time(value: Union[datetime, str, Time]) -> Time:
    if isinstance(value, Time):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return Time(value, scale="tai")
        return Time(value)
    return Time(value, scale="tai")


def _normalize_collections(collections: Union[str, Sequence[str]]) -> tuple[str, ...]:
    if isinstance(collections, str):
        return (collections,)
    return tuple(str(c) for c in collections)


def _resolve_ncores(ncores: Optional[int], n_items: int) -> int:
    if ncores is None:
        return max(1, min(os.cpu_count() or 1, n_items))
    if ncores < 1:
        raise ValueError("ncores must be >= 1")
    return ncores


_PROCESS_CUTOUT_SERVICE: Optional[ButlerCutoutService] = None


def _init_process_cutout_worker(repo: str, collections: tuple[str, ...]) -> None:
    global _PROCESS_CUTOUT_SERVICE
    butler = Butler(repo, collections=list(collections))
    _PROCESS_CUTOUT_SERVICE = ButlerCutoutService(butler=butler, repo=repo, collections=collections)


def _process_cutout_worker(args: tuple[str, tuple[Any, Any, Any, Any, Any, Any], Any, Any, bool]) -> Any:
    global _PROCESS_CUTOUT_SERVICE
    if _PROCESS_CUTOUT_SERVICE is None:
        raise RuntimeError("Process cutout worker service not initialized")
    dataset_type, item, h, w, pad = args
    return _PROCESS_CUTOUT_SERVICE._run_cutout_item(
        dataset_type=dataset_type, item=item, h=h, w=w, pad=pad
    )
