import numpy as np
import astropy.units as u

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy.coordinates import SkyCoord


def query_ephemeris(target, target_type, start, end, step, location="X05"):
    """
    Query JPL Horizons for ephemeris data.

    Parameters
    ----------
    target : str
        Asteroid name or designation (e.g. "Ceres", "2024 AA").
    target_type : str
        Horizons id_type: "smallbody", "designation", "asteroid_name", etc.
    start : str
        Start date, e.g. "2024-11-01".
    end : str
        End date, e.g. "2024-11-15".
    step : str
        Time step, e.g. "4h", "1d".
    location : str
        Observer location code. "X05" = Rubin Observatory.

    Returns
    -------
    dict with keys:
        times : astropy.time.Time array
        ra_deg : np.ndarray
        dec_deg : np.ndarray
        rss_3sigma : np.ndarray (arcsec)
        smaa_3sigma : np.ndarray (arcsec)
        smia_3sigma : np.ndarray (arcsec)
        theta_3sigma : np.ndarray (degrees, E of N)
    """
    obj = Horizons(
        id=target,
        id_type=target_type,
        location=location,
        epochs={"start": start, "stop": end, "step": step},
    )
    eph = obj.ephemerides(skip_daylight=False)

    return {
        "times": Time(eph["datetime_jd"], scale="utc", format="jd"),
        "ra_deg": np.array(eph["RA"]),
        "dec_deg": np.array(eph["DEC"]),
        "rss_3sigma": np.array(eph["RSS_3sigma"]),
        "smaa_3sigma": np.array(eph["SMAA_3sigma"]),
        "smia_3sigma": np.array(eph["SMIA_3sigma"]),
        "theta_3sigma": np.array(eph["Theta_3sigma"]),
    }

def calculate_polygons(times, ra_deg, dec_deg, time_interval_days=3.0, widening_arcsec=2.0):
    """
    Create sky polygons along an ephemeris track.

    Parameters
    ----------
    times : astropy.time.Time
        Array of ephemeris times.
    ra_deg : np.ndarray
        RA in degrees at each time.
    dec_deg : np.ndarray
        Dec in degrees at each time.
    time_interval_days : float
        Max duration of each polygon segment in days.
    widening_arcsec : float
        Width of the polygon on each side of the track, in arcseconds.

    Returns
    -------
    list of dict, each with:
        "time_start" : str 
        "time_end" : str 
        "polygon_corners" : list of (ra, dec) tuples in degrees
    """
    all_polygons = []
    extension = widening_arcsec * u.arcsec
    current_index = 0

    while current_index < len(times):
        start_time = times[current_index]
        target_end_time = start_time + time_interval_days * u.day

        # Find the last index within this time interval
        end_index = current_index
        for j in range(current_index, len(times)):
            if times[j] <= target_end_time:
                end_index = j
            else:
                break

        # Segment start and end sky positions
        a = SkyCoord(ra=ra_deg[current_index], dec=dec_deg[current_index], unit="deg", frame="icrs")
        b = SkyCoord(ra=ra_deg[end_index], dec=dec_deg[end_index], unit="deg", frame="icrs")

        separation = a.separation(b)
        if separation > 1e-10 * u.arcsec:
            pa_ab = a.position_angle(b)
        else:
            pa_ab = 0.0 * u.deg

        # Extend the path segment at both ends
        a_ext = a.directional_offset_by(pa_ab - 180 * u.deg, extension)
        b_ext = b.directional_offset_by(pa_ab, extension)

        # Create rectangle perpendicular to track
        pa_rect = a_ext.position_angle(b_ext)
        corner1 = a_ext.directional_offset_by(pa_rect + 90 * u.deg, extension)
        corner2 = a_ext.directional_offset_by(pa_rect - 90 * u.deg, extension)
        corner3 = b_ext.directional_offset_by(pa_rect - 90 * u.deg, extension)
        corner4 = b_ext.directional_offset_by(pa_rect + 90 * u.deg, extension)

        all_polygons.append({
            "time_start": times[current_index].jd,
            "time_end": times[end_index].jd,
            "polygon_corners": [
                (corner1.ra.deg, corner1.dec.deg),
                (corner4.ra.deg, corner4.dec.deg),
                (corner3.ra.deg, corner3.dec.deg),
                (corner2.ra.deg, corner2.dec.deg),
            ],
        })

        current_index = end_index + 1

    return all_polygons