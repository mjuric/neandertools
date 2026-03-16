# neandertools

<img src="logo.png" alt="neandertools logo" width="220" align="left" />

Utilities for generating asteroid cutout animations.

<br clear="left" />

---

## What it does

Given an asteroid name and a date range, **neandertools** will:

1. Query **JPL Horizons** for the asteroid's predicted track
2. Build sky-search polygons along that track and find all matching **Rubin/LSST visit images** via the Butler
3. Extract a postage-stamp **cutout** centered on the asteroid at each epoch
4. Export an **animated GIF** (and optionally a grid image) with the asteroid fixed at the centre

The package runs on the **Rubin Science Platform** (RSP), where the LSST software stack is available.

---

## Requirements

- Python ≥ 3.10
- `numpy`, `astropy`, `astroquery`, `matplotlib` (installed automatically)
- **LSST software stack** — only available on the Rubin Science Platform

---

## Installation

```bash
pip install .          # from the repo root
# or in editable / development mode:
pip install -e .[dev]
```

---

## Quick start — command line

`run_pipeline.py` is the recommended entry point. It runs the full pipeline and writes a GIF (and optionally a contact-sheet grid) without needing a Jupyter kernel.

```bash
# Minimal — all defaults (dp1, LSSTComCam/DP1, all bands, 100 px cutouts):
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30

# Also save a contact-sheet grid alongside the GIF:
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30 --grid

# r-band only, WCS-aligned frames, 6-column grid, custom output paths:
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30 \
    --output tn57.gif --bands r --warp \
    --grid --grid-output tn57_grid.png --grid-ncols 6

# Larger cutouts, slower GIF:
python run_pipeline.py Ceres 2024-11-01 2024-11-15 \
    --cutout-size 200 --frame-duration 800

# Show all pipeline log messages:
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30 -v

# Other example with brighter target:
python run_pipeline.py "1991 SJ" 2024-11-25 2024-11-28 --output 1991SJ.gif --warp --grid --cutout-size 200 --match-noise --show-ne --step 12h
```

### Full flag reference

| Flag | Default | Description |
|---|---|---|
| `target` | *(required)* | Asteroid name or designation |
| `start` | *(required)* | Start date `YYYY-MM-DD` |
| `end` | *(required)* | End date `YYYY-MM-DD` |
| `--dr` | `dp1` | Butler data release label |
| `--collection` | `LSSTComCam/DP1` | Butler collection |
| `--bands` | all ugrizy | Filter bands to search (space-separated) |
| `--step` | `12h` | Ephemeris time step |
| `--polygon-interval` | `3.0` | Max days per search polygon |
| `--polygon-widening` | `2.0` | Search corridor half-width (arcsec) |
| `--location` | `X05` | Observer location code (X05 = Rubin) |
| `--cutout-size` | `100` | Cutout side length in pixels |
| `--output` | `asteroid.gif` | Output GIF path |
| `--frame-duration` | `500` | Frame duration in milliseconds |
| `--match-background` / `--no-match-background` | on | Subtract per-frame background |
| `--match-noise` | off | Divide by per-frame RMS (SNR display) |
| `--show-ne` | off | Draw North/East compass on each frame |
| `--warp` | off | Warp frames onto a common sky grid |
| `--cmap` | `gray` | Matplotlib colormap |
| `--grid` | off | Also save a contact-sheet grid image |
| `--grid-output` | `<gif-stem>_grid.png` | Grid image path |
| `--grid-ncols` | `5` | Number of columns in the grid |
| `--grid-dpi` | `150` | Grid image resolution (DPI) |
| `-v` / `--verbose` | off | Enable DEBUG logging |

The script prints the output path(s) on success so they can be captured in shell scripts:

```bash
gif=$(python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30)
```

---

## Quick start — Python API

### High-level: full pipeline

```python
from neandertools import AsteroidCutoutPipeline

pipeline = AsteroidCutoutPipeline(
    target="2024 TN57",
    start="2024-12-01",
    end="2024-12-30",
)

# Run and write GIF
gif_path = pipeline.run(
    output_path="asteroid.gif",
    warp_common_grid=True,   # align frames on sky
    show_ne=True,            # North/East compass
    match_background=True,   # normalise background level
)

# Display a contact-sheet grid (in a notebook)
fig, axes = pipeline.grid(ncols=6, warp_common_grid=True)
```

`AsteroidCutoutPipeline` parameters:

| Parameter | Default | Description |
|---|---|---|
| `target` | *(required)* | Asteroid name or designation |
| `start`, `end` | *(required)* | Date range strings |
| `dr` | `"dp1"` | Butler data release label |
| `collection` | `"LSSTComCam/DP1"` | Butler collection |
| `bands` | all ugrizy | List of filter bands |
| `step` | `"12h"` | Ephemeris time step |
| `cutout_size` | `100` | Cutout side in pixels |
| `polygon_interval_days` | `3.0` | Days per search polygon segment |
| `polygon_widening_arcsec` | `2.0` | Search corridor half-width |
| `location` | `"X05"` | JPL Horizons observer location |

`run()` parameters:

| Parameter | Default | Description |
|---|---|---|
| `output_path` | `"asteroid.gif"` | Output GIF path |
| `frame_duration_ms` | `500` | Frame duration (ms) |
| `match_background` | `True` | Subtract per-frame background |
| `match_noise` | `False` | Divide by per-frame RMS |
| `show_ne` | `False` | Draw North/East compass |
| `warp_common_grid` | `False` | Align frames on a common sky grid |

### Low-level: direct cutout extraction

```python
from neandertools import cutouts_from_butler

svc = cutouts_from_butler(
    "dp1",
    collections=["LSSTComCam/DP1"],
)

# By pixel coordinates
images = svc.cutout(visit=2024110800253, detector=5, x=2036, y=2000, h=201, w=201)

# By sky coordinates (resolved via image WCS)
images = svc.cutout(visit=2024110800253, detector=5, ra=62.1, dec=-31.2, h=201, w=201)

# Vectorised — one cutout per row
images = svc.cutout(
    visit=[2024110800253, 2024110800254],
    detector=[5, 0],
    ra=[62.1, 63.2],
    dec=[-31.2, -30.9],
    h=201,
    w=201,
)

# Find which (visit, detector) pairs contain a sky position at a given time
visit, detector = svc.find_visit_detector(ra=53.0, dec=-27.91, t="2024-11-09T06:12:10")

# Vectorised lookup
visit_many, detector_many = svc.find_visit_detector(
    ra=[53.0, 53.1],
    dec=[-27.91, -27.95],
    t=["2024-11-09T06:12:10", "2024-11-09T06:13:10"],
)
```

`cutout()` notes:

- Center must be either `(x, y)` **or** `(ra, dec)`, not both.
- `visit` and `detector` are always required.
- `h` / `w` default to the full image size if omitted.
- Edge-overlapping requests are padded with NaN by default (`pad=True`) so the output shape is always `(h, w)` and the target stays at the centre pixel. Frames where the target falls on an image border strip are automatically discarded. Set `pad=False` for clipped (no-padding) behaviour.
- All coordinate arguments accept scalars or 1-D arrays; arrays are paired by index.

---

## Package layout

```
src/neandertools/
├── __init__.py        # Public API: AsteroidCutoutPipeline, cutouts_from_butler
├── pipeline.py        # AsteroidCutoutPipeline — full end-to-end orchestration
├── butler.py          # ButlerCutoutService — image loading and cutout extraction
├── trackbuilder.py    # JPL Horizons query + search polygon construction
├── imagefinder.py     # Butler image search + position interpolation
└── visualization.py   # GIF and grid rendering

run_pipeline.py        # Command-line entry point
tests/
└── test_butler_service.py
```

---

## Known platform constraints

- The **LSST stack** (`lsst.daf.butler`, `lsst.geom`, `lsst.afw.*`) is only available on the Rubin Science Platform. The package will import but the pipeline will fail outside RSP.
- WCS warping (`--warp` / `warp_common_grid=True`) additionally requires `lsst.afw.geom` and `lsst.afw.math`. A clear error is raised if they are absent.
- Multiprocessing uses the `fork` start method (Linux / macOS). On platforms without `fork` the service falls back to serial execution automatically.
