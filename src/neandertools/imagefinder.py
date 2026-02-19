import lsst.sphgeom as sphgeom
from lsst.daf.butler import Butler, Timespan
from astropy.time import Time
import lsst.geom as geom
import lsst.afw.image as afwimage
import numpy as np

DR = "dp1"
COLLECTION = "LSSTComCam/DP1"

def find_overlapping_images(polygons, bands, dr=DR, collection=COLLECTION):
    """
    Find visit_image datasets that overlap the search polygons.

    Parameters
    ----------
    polygons : list of dict
        From calculate_polygons(). Each has "time_start", "time_end", "polygon_corners".
    bands : list of str
        Filter bands to search, e.g. ["g", "r", "i"].
    dr : str
        Butler data release label.
    collection : str
        Butler collection name.

    Returns
    -------
    butler : Butler
        The initialized Butler instance (reused in Step 4).
    unique_refs : list of DatasetRef
        Deduplicated dataset references.
    """
    butler = Butler(dr, collections=collection)
    bands_clause = "band.name IN (" + ", ".join([f"'{b}'" for b in bands]) + ")"

    all_dataset_refs = []
    for item in polygons:
        time_start = Time(item["time_start"], format="jd", scale="tai")
        time_end = Time(item["time_end"], format="jd", scale="tai")
        timespan_search = Timespan(time_start, time_end)

        # Convert polygon corners to sphgeom Region (IVOA POS format)
        corners = item["polygon_corners"]
        polygon_string = "POLYGON " + " ".join([f"{c[0]} {c[1]}" for c in corners])
        region_search = sphgeom.Region.from_ivoa_pos(polygon_string)

        where_clause = (
            f"{bands_clause} AND visit.timespan OVERLAPS timespan "
            f"AND visit_detector_region.region OVERLAPS region"
        )

        try:
            refs = list(butler.query_datasets(
                "visit_image",
                where=where_clause,
                bind={"timespan": timespan_search, "region": region_search},
            ))
            all_dataset_refs.extend(refs)
        except Exception:
            continue  # empty result or error for this polygon

    # Deduplicate (same visit+detector can appear from multiple polygons)
    unique_refs = list(set(all_dataset_refs))
    return unique_refs

def create_cutout(exposure, ra_deg, dec_deg, cutout_size_px=100):
    """
    Extract a pixel cutout from a loaded ExposureF.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureF
        Full visit image loaded from Butler.
    ra_deg : float
        Target RA in degrees.
    dec_deg : float
        Target Dec in degrees.
    cutout_size_px : int
        Cutout side length in pixels.

    Returns
    -------
    lsst.afw.image.ExposureF or None
        Cutout sub-image with WCS preserved, or None if target is outside image.
    """
    wcs = exposure.getWcs()
    coord = geom.SpherePoint(np.radians(ra_deg), np.radians(dec_deg), geom.radians)
    pixel_coord = wcs.skyToPixel(coord)

    half = cutout_size_px // 2
    bbox = geom.Box2I()
    bbox.include(geom.Point2I(int(pixel_coord.getX() - half), int(pixel_coord.getY() - half)))
    bbox.include(geom.Point2I(int(pixel_coord.getX() + half), int(pixel_coord.getY() + half)))

    # Clip to image bounds; skip if no overlap
    bbox.clip(exposure.getBBox())
    if bbox.isEmpty():
        return None

    return exposure.Factory(exposure, bbox, origin=afwimage.PARENT, deep=False)

def get_obs_time(visit_id, detector_id):
    """Get observation midpoint time from visitInfo."""
    butler = Butler(DR, collections=COLLECTION)
    visit_info = butler.get("visit_image.visitInfo", visit=visit_id, detector=detector_id)
    t_mid = visit_info.date.toAstropy()
    if t_mid.scale != "tai":
        t_mid = t_mid.tai
    return t_mid

def interpolate_position(obs_time_mjd, ephem_times_mjd, ra_deg, dec_deg):
    """
    Linearly interpolate RA/Dec at a given MJD.

    Parameters
    ----------
    obs_time_mjd : float
        Observation time in MJD (TAI).
    ephem_times_mjd : np.ndarray
        Ephemeris grid times in MJD (TAI).
    ra_deg, dec_deg : np.ndarray
        Ephemeris RA/Dec in degrees.

    Returns
    -------
    (ra, dec) : tuple of float
        Interpolated position in degrees.
    """
    ra = np.interp(obs_time_mjd, ephem_times_mjd, ra_deg)
    dec = np.interp(obs_time_mjd, ephem_times_mjd, dec_deg)
    return float(ra), float(dec)