import argparse
import os
import astra
import matplotlib.pyplot as plt
import numpy as np
from reconstruct import sinogram_volume, reconstruct_volume_fbp, reconstruct_volume_sirt, rmse, SIRT_ITERATIONS


# angle sets to compare
ANGLE_SETS = {
    "many_angles_180": np.linspace(0, 2 * np.pi, 180, endpoint=False),
    "few_angles_30":   np.linspace(0, 2 * np.pi,  30, endpoint=False),
    "few_angles_10":   np.linspace(0, 2 * np.pi,  10, endpoint=False),
}

# noise levels (standard deviation as fraction of maximum sinogram value)
NOISE_LEVELS = {
    "no_noise":     0.00,
    "low_noise":    0.01,
    "high_noise":   0.05,
}

OUTPUT_DIR = "results"

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

# extract the middle slice from a 3D volume for visualisation
def _middle_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]

# helper to save a figure to OUTPUT_DIR with a consistent naming scheme, and display it
def show_and_save(fig, tag: str, phantom_number: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"phantom_{phantom_number}_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.show()
    plt.close(fig)


# plots comparing sinograms for different conditions, and a grid comparing original phantom slice, FBP reconstruction, and SIRT reconstruction for multiple conditions
def plot_sinogram_comparison(sinograms_dict: dict, phantom_number: str) -> None:
    
    n = len(sinograms_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, sinos) in zip(axes, sinograms_dict.items()):
        mid_z = sinos.shape[0] // 2
        ax.imshow(sinos[mid_z], cmap="gray", aspect="auto")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Detector pixel")
        ax.set_ylabel("Angle index")

    fig.suptitle(f"Parallel-beam sinograms — Phantom {phantom_number}", fontsize=11)
    fig.tight_layout()
    show_and_save(fig, "sinograms", phantom_number)

# make a grid plot comparing original phantom slice, FBP reconstruction, and SIRT reconstruction for multiple experimental conditions
def plot_reconstruction_grid(phantom: np.ndarray, results: dict, phantom_number: str) -> None:

    settings = list(results.keys())
    n_rows = len(settings)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    orig_slice = _middle_slice(phantom)
    vmin, vmax = orig_slice.min(), orig_slice.max()

    for row, setting in enumerate(settings):
        res = results[setting]

        # Original
        axes[row, 0].imshow(orig_slice, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        # FBP
        fbp_slice = _middle_slice(res["fbp"])
        axes[row, 1].imshow(fbp_slice, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(
            f"FBP  RMSE={res['rmse_fbp']:.4f}\n{setting}", fontsize=8
        )
        axes[row, 1].axis("off")

        # SIRT
        sirt_slice = _middle_slice(res["sirt"])
        axes[row, 2].imshow(sirt_slice, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(
            f"SIRT ({SIRT_ITERATIONS} it)  RMSE={res['rmse_sirt']:.4f}\n{setting}", fontsize=8
        )
        axes[row, 2].axis("off")

    fig.suptitle(f"Reconstruction comparison — Phantom {phantom_number}", fontsize=12)
    fig.tight_layout()
    show_and_save(fig, "reconstructions", phantom_number)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# full experiment pipeline for phantom: compute sinograms for all conditions, reconstruct with FBP and SIRT, compute RMSE, and generate figures
# Conditions tested (6 total):
# - many_angles + no_noise
# - many_angles + high_noise
# - few_angles_30 + no_noise
# - few_angles_30 + low_noise
# - few_angles_10 + no_noise
# - few_angles_10 + high_noise

def run_experiment(phantom: np.ndarray, phantom_number: str) -> None:
    
    conditions = [
        ("many_angles_180", "no_noise"),
        ("many_angles_180", "high_noise"),
        ("few_angles_30",   "no_noise"),
        ("few_angles_30",   "low_noise"),
        ("few_angles_10",   "no_noise"),
        ("few_angles_10",   "high_noise"),
    ]

    sinograms_for_plot = {}
    results = {}

    # loop through conditions, compute sinograms, reconstructions, RMSE, and store results for plotting
    for angle_key, noise_key in conditions:
        label = f"{angle_key} | {noise_key}"
        angles = ANGLE_SETS[angle_key]
        noise  = NOISE_LEVELS[noise_key]

        print(f"\n[{label}]")
        print(f"  Forward projecting {phantom.shape[0]} slices …")
        sinos = sinogram_volume(phantom, angles, noise_fraction=noise)

        print("  Reconstructing with FBP …")
        fbp_vol = reconstruct_volume_fbp(sinos, angles)

        print(f"  Reconstructing with SIRT ({SIRT_ITERATIONS} iterations) …")
        sirt_vol = reconstruct_volume_sirt(sinos, angles)

        err_fbp  = rmse(phantom, fbp_vol)
        err_sirt = rmse(phantom, sirt_vol)
        print(f"  RMSE  FBP={err_fbp:.4f}  SIRT={err_sirt:.4f}")

        sinograms_for_plot[label] = sinos
        results[label] = {
            "fbp": fbp_vol,
            "sirt": sirt_vol,
            "rmse_fbp": err_fbp,
            "rmse_sirt": err_sirt,
        }

    print("\nGenerating sinogram figure …")
    plot_sinogram_comparison(sinograms_for_plot, phantom_number)

    print("Generating reconstruction grid …")
    plot_reconstruction_grid(phantom, results, phantom_number)

# main function to load a phantom and run the experiment pipeline
def main():
    # specify phantom
    phantom_path = "phantoms/phantom_089.npy"   # path to phantom .npy file

    # load phantom 
    print(f"Loading {phantom_path} …")
    phantom = np.load(phantom_path)

    filename = os.path.basename(phantom_path)
    phantom_number = filename.split("_")[1].split(".")[0]

    print(f"Shape: {phantom.shape}")

    # run full experiment pipeline 
    run_experiment(phantom, phantom_number)



if __name__ == "__main__":
    main()