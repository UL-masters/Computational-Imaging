import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_phantom(path: str):
    phantom = np.load(path)
    filename = os.path.basename(path)
    phantom_number = filename.split("_")[1].split(".")[0]
    print(f"Loaded phantom_{phantom_number}  shape: {phantom.shape}")
    return phantom, phantom_number

def show_slices(volume: np.ndarray, title: str = "Volume", save_path: str | None = None) -> None:
    mid_z = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    mid_x = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    views = [
        (volume[mid_z],       f"Axial  (z={mid_z})"),
        (volume[:, mid_y, :], f"Coronal (y={mid_y})"),
        (volume[:, :, mid_x], f"Sagittal (x={mid_x})"),
    ]

    vmin, vmax = volume.min(), volume.max()
    for ax, (img, label) in zip(axes, views):
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize a phantom volume.")
    parser.add_argument(
        "--phantom",
        default=os.path.join(SCRIPT_DIR, "phantoms/phantom_090.npy"),  # anchored to script
        help="Path to phantom .npy file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,  # always save when running from VSCode
        help="Save figure to results_phantoms/",
    )
    args = parser.parse_args()

    phantom, phantom_number = load_phantom(args.phantom)

    save_path = os.path.join(SCRIPT_DIR, f"results_phantoms/phantom_{phantom_number}_overview.png")

    show_slices(phantom, f"Phantom {phantom_number}", save_path=save_path)


if __name__ == "__main__":
    main()