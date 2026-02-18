# neandertools

`neandertools` provides a simple API for generating image cutouts from Rubin Observatory LSST Butler repositories.

## Quick start

```python
import neandertools as nt

svc = nt.cutouts_from_butler(
    "~/lsst/dp1_subset",
    collections=["LSSTComCam/runs/DRP/DP1/DM-51335"],
)

# dataset_type defaults to "visit_image"
images = svc.cutout(visit=2024110800253, detector=5, x=2036, y=2000, h=201, w=201)

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
- `cutout(...)` takes center pixel coordinates (`x`, `y`) and size (`h`, `w`).
- If `x`/`y` are omitted they default to image center; if `h`/`w` are omitted they default to full image size.
- The sky-coordinate mode (`ra/dec/time`) is wired and needs a `sky_resolver` callback.
