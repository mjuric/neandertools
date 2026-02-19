# neandertools

<img src="logo.png" alt="neandertools logo" width="220" align="left" />

`neandertools`: Utilities for LSST users unable to appreciate complexity

<br clear="left" />

## Quick start

```python
import neandertools as nt

svc = nt.cutouts_from_butler(
    "~/lsst/dp1_subset",
    collections=["LSSTComCam/runs/DRP/DP1/DM-51335"],
)

# dataset_type defaults to "visit_image"
images = svc.cutout(visit=2024110800253, detector=5, x=2036, y=2000, h=201, w=201)

# center by sky coordinates (ra, dec), resolved via image WCS
images = svc.cutout(visit=2024110800253, detector=5, ra=62.1, dec=-31.2, h=201, w=201)

# vectorized input: one cutout per index
images = svc.cutout(
    visit=[2024110800253, 2024110800254],
    detector=[5, 0],
    ra=[62.1, 63.2],
    dec=[-31.2, -30.9],
    h=201,
    w=201,
)

# find all (visit, detector) containing sky coordinates at a given time
visit, detector = svc.find_visit_detector(ra=53.0, dec=-27.91, t="2024-11-09T06:12:10")

# vectorized lookup returns flattened matches in input order
visit_many, detector_many = svc.find_visit_detector(
    ra=[53.0, 53.1],
    dec=[-27.91, -27.95],
    t=["2024-11-09T06:12:10", "2024-11-09T06:13:10"],
)

# optional override
images = svc.cutout(
    visit=2024110800253,
    detector=5,
    x=2036,
    y=2000,
    h=201,
    w=201,
    dataset_type="preliminary_visit_image",
)
```

## Notes

- `collections` is required when creating the service.
- `dataset_type` is selected per `cutout(...)` call and defaults to `"visit_image"`.
- `cutout(...)` center must be specified as either (`x`, `y`) or (`ra`, `dec`).
- `cutout(...)` requires both `visit` and `detector`.
- If center is (`ra`, `dec`), WCS is used to convert to pixel coordinates.
- `visit`, `detector`, `x`, `y`, `ra`, and `dec` can be scalars or arrays. Arrays are paired by index.
- If `h`/`w` are omitted they default to full image size.
- `cutout(...)` pads edge-overlapping requests by default (`pad=True`) so the requested center stays at the center pixel; set `pad=False` to keep clipped behavior.
- `find_visit_detector(...)` returns two 1D numpy arrays `(visit, detector)` for all matches that contain `(ra, dec)` and satisfy `begin <= t < end`; it accepts scalar or 1D vector inputs.
