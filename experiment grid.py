import os
import glob
import numpy as np
import pandas as pd
from reconstruct import (
    sinogram_volume,
    reconstruct_volume_fbp,
    reconstruct_volume_sirt,
    reconstruct_volume_sirt_tv,
    rmse, ssim_score, dice_score,
    SIRT_ITERATIONS,
)


PHANTOMS_DIR  = "phantoms"
OUTPUT_DIR    = "results"
OUTPUT_CSV    = os.path.join(OUTPUT_DIR, "grid_results.csv")

N_PHANTOMS = 20

ANGLE_COUNTS  = [10, 30, 90, 180]
NOISE_LEVELS  = [0.00, 0.01, 0.05]
ALGORITHMS    = ["FBP", "SIRT", "SIRT-TV"]

SIRT_TV_LAM   = 0.01



def has_stone(phantom: np.ndarray, stone_threshold: float = 0.3) -> bool:
    """Return True if the phantom contains at least one stone-like voxel."""
    return bool(phantom.max() > stone_threshold)


def load_phantoms(phantoms_dir: str, n: int):
    paths = sorted(glob.glob(os.path.join(phantoms_dir, "phantom_*.npy")))
    phantoms = []
    for p in paths:
        vol = np.load(p)
        if has_stone(vol):
            phantoms.append((os.path.basename(p), vol))
        if len(phantoms) == n:
            break
    print(f"Loaded {len(phantoms)} phantoms with stones.")
    return phantoms


def reconstruct(algorithm: str, sinograms: np.ndarray,
                angles: np.ndarray) -> np.ndarray:
    if algorithm == "FBP":
        return reconstruct_volume_fbp(sinograms, angles)
    elif algorithm == "SIRT":
        return reconstruct_volume_sirt(sinograms, angles, SIRT_ITERATIONS)
    elif algorithm == "SIRT-TV":
        return reconstruct_volume_sirt_tv(sinograms, angles, SIRT_ITERATIONS, SIRT_TV_LAM)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phantoms = load_phantoms(PHANTOMS_DIR, N_PHANTOMS)
    rows = []

    total = len(phantoms) * len(ANGLE_COUNTS) * len(NOISE_LEVELS) * len(ALGORITHMS)
    done  = 0

    for phantom_name, phantom in phantoms:
        for n_angles in ANGLE_COUNTS:
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

            for noise in NOISE_LEVELS:
                # compute sinogram once per (phantom, angles, noise), reused by all algorithms
                sinos = sinogram_volume(phantom, angles, noise_fraction=noise)

                for algorithm in ALGORITHMS:
                    recon = reconstruct(algorithm, sinos, angles)

                    row = {
                        "phantom":   phantom_name,
                        "algorithm": algorithm,
                        "n_angles":  n_angles,
                        "noise":     noise,
                        "rmse":      rmse(phantom, recon),
                        "ssim":      ssim_score(phantom, recon),
                        "dice":      dice_score(phantom, recon),
                    }
                    rows.append(row)
                    done += 1
                    print(
                        f"[{done}/{total}] {phantom_name} | {algorithm} | "
                        f"{n_angles} angles | noise={noise:.2f} | "
                        f"RMSE={row['rmse']:.4f}  SSIM={row['ssim']:.4f}  Dice={row['dice']:.4f}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()