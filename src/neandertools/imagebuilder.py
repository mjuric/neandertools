import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
from lsst.daf.butler import Butler

butler = Butler("dp1", collections= "LSSTComCam/DP1")

def get_images(butler, visit_id, detector_id):
    """Get the exposure from RSP via Butler"""
    try:
        exposure = butler.get(
            "visit_image",
            dataId={"visit": visitid, "detector": detector},
        )
    except Exception as e:
        print(f"Skipping visit={visitid} det={detector}: {e}")


def cutout_to_png(cutout_exposure, output_path, title=""):
    """Render an LSST ExposureF cutout to a PNG file."""
    image_array = cutout_exposure.image.array
    norm = ImageNormalize(image_array, interval=ZScaleInterval())

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image_array, origin="lower", cmap="gray", norm=norm)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved in {output_path}")
    plt.close(fig) 

def create_gif(png_dir, output_path, duration=500):
    """
    Create an animated GIF from a directory of PNG frames.

    Parameters
    ----------
    png_dir : str
        Directory containing PNG frames, sorted by filename.
    output_path : str
        Output GIF file path.
    duration : int
        Duration of each frame in milliseconds.
    """
    png_files = sorted(glob.glob(f"{png_dir}/*.png"))
    if not png_files:
        raise ValueError(f"No PNG files found in {png_dir}")

    frames = [Image.open(f) for f in png_files]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,  # 0 = loop forever
    )
    print(f"GIF saved to {output_path} ({len(frames)} frames)")