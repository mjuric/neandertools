#!/usr/bin/env python
"""Run the AsteroidCutoutPipeline from the command line.

Examples
--------
# Minimal — GIF only, all defaults (dp1, LSSTComCam/DP1, all bands):
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30

# GIF + grid side-by-side:
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30 --grid

# Custom paths, r-band only, WCS warping, 6-column grid:
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30 \
    --output my_asteroid.gif --grid --grid-output my_grid.png \
    --bands r --warp --grid-ncols 6

# Larger cutouts, slower GIF, no background matching:
python run_pipeline.py Ceres 2024-11-01 2024-11-15 \
    --cutout-size 200 --frame-duration 800 --no-match-background

# Verbose logging to see every step:
python run_pipeline.py "2024 TN57" 2024-12-01 2024-12-30 -v
"""

import argparse
import logging
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate an animated GIF (and optionally a grid image) of asteroid cutouts from Rubin/LSST data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional — the three required inputs
    p.add_argument("target", help="Asteroid name or designation, e.g. '2024 TN57' or 'Ceres'")
    p.add_argument("start",  help="Start date, e.g. 2024-12-01")
    p.add_argument("end",    help="End date,   e.g. 2024-12-30")

    # Butler / data
    p.add_argument("--dr",         default="dp1",              help="Butler data release label")
    p.add_argument("--collection", default="LSSTComCam/DP1",   help="Butler collection name")
    p.add_argument("--bands",      nargs="+", default=None,
                   metavar="BAND",
                   help="Filter bands to search (default: all ugrizy)")

    # Ephemeris / search
    p.add_argument("--step",               default="12h",  help="Ephemeris time step")
    p.add_argument("--polygon-interval",   default=3.0,    type=float,
                   metavar="DAYS", help="Max duration of each search polygon")
    p.add_argument("--polygon-widening",   default=2.0,    type=float,
                   metavar="ARCSEC", help="Search polygon half-width")
    p.add_argument("--location",           default="X05",
                   help="Observer location code (X05 = Rubin Observatory)")
    p.add_argument("--target-type",        default="smallbody",
                   help="Horizons id_type")

    # Cutout / display (shared by both GIF and grid)
    p.add_argument("--cutout-size",     default=100,  type=int,
                   metavar="PX", help="Cutout side length in pixels")
    p.add_argument("--cmap",            default="gray",
                   help="Matplotlib colormap name")

    p.add_argument("--match-background", dest="match_background",
                   action="store_true",  default=True,
                   help="Subtract per-cutout background (recommended)")
    p.add_argument("--no-match-background", dest="match_background",
                   action="store_false",
                   help="Disable background subtraction")

    p.add_argument("--match-noise", action="store_true", default=False,
                   help="Divide by per-cutout RMS (SNR-like display)")
    p.add_argument("--show-ne",     action="store_true", default=False,
                   help="Draw a North/East compass indicator on each frame/panel")
    p.add_argument("--warp",        action="store_true", default=False,
                   help="Warp all cutouts onto a common sky grid (requires LSST warp modules)")

    # GIF-specific
    p.add_argument("--output",          default="asteroid.gif",
                   help="Output GIF file path")
    p.add_argument("--frame-duration",  default=500,  type=int,
                   metavar="MS", help="Duration of each GIF frame in milliseconds")

    # Grid
    p.add_argument("--grid",            action="store_true", default=False,
                   help="Also save a grid image of all cutout frames")
    p.add_argument("--grid-output",     default=None,
                   metavar="PATH",
                   help="Grid image output path (default: <gif-stem>_grid.png)")
    p.add_argument("--grid-ncols",      default=5,  type=int,
                   metavar="N", help="Number of columns in the grid")
    p.add_argument("--grid-dpi",        default=150, type=int,
                   metavar="DPI", help="Resolution of the saved grid image")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG logging")

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here so import errors are shown cleanly after arg parsing
    try:
        from neandertools import AsteroidCutoutPipeline
    except ImportError as exc:
        logging.error("Cannot import neandertools: %s", exc)
        logging.error("Make sure the package is installed and the LSST stack is active.")
        sys.exit(1)

    pipeline = AsteroidCutoutPipeline(
        target=args.target,
        start=args.start,
        end=args.end,
        dr=args.dr,
        collection=args.collection,
        target_type=args.target_type,
        location=args.location,
        bands=args.bands,
        step=args.step,
        cutout_size=args.cutout_size,
        polygon_interval_days=args.polygon_interval,
        polygon_widening_arcsec=args.polygon_widening,
    )

    logging.info(
        "Running pipeline for '%s'  %s -> %s",
        args.target, args.start, args.end,
    )

    gif_path = pipeline.run(
        output_path=args.output,
        frame_duration_ms=args.frame_duration,
        match_background=args.match_background,
        match_noise=args.match_noise,
        show_ne=args.show_ne,
        warp_common_grid=args.warp,
    )

    n = len(pipeline.cutouts)
    if n == 0:
        logging.warning("No frames were produced — check target name, date range, and collection.")
        sys.exit(1)

    logging.info("Done. %d frame(s) -> %s", n, gif_path)
    print(gif_path)

    if args.grid:
        import matplotlib
        matplotlib.use("Agg")  # ensure no display is required
        import matplotlib.pyplot as plt

        # Derive default grid output path from the GIF path
        grid_path = Path(args.grid_output) if args.grid_output else Path(args.output).with_name(
            Path(args.output).stem + "_grid.png"
        )

        logging.info("Saving cutout grid (%d cols) -> %s", args.grid_ncols, grid_path)

        fig, _ = pipeline.grid(
            ncols=args.grid_ncols,
            match_background=args.match_background,
            match_noise=args.match_noise,
            show_ne=args.show_ne,
            warp_common_grid=args.warp,
            cmap=args.cmap,
            show=False,
        )

        if fig is not None:
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(grid_path, dpi=args.grid_dpi, bbox_inches="tight")
            plt.close(fig)
            logging.info("Grid saved -> %s", grid_path)
            print(grid_path)
        else:
            logging.warning("Grid could not be generated.")


if __name__ == "__main__":
    main()
