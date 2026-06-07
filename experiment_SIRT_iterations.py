import os
import glob
import numpy as np
import pandas as pd
from reconstruct import (
    sinogram_volume,
    reconstruct_volume_sirt,
    rmse, ssim_score, dice_score,
)
from experiment_grid import has_stone, load_phantoms

PHANTOMS_DIR = "phantoms"
OUTPUT_DIR   = "results"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "sirt_iterations_results.csv")

N_PHANTOMS   = 20 # only a subset of phantoms to speed up the experiments 

# Best-performing condition from the grid: many angles, no noise
N_ANGLES     = 180
NOISE        = 0.00

# Iteration counts to evaluate: sparse at first, then finer near convergence
ITERATION_COUNTS = [5, 10, 25, 50, 100, 200, 500]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phantoms = load_phantoms(PHANTOMS_DIR, N_PHANTOMS)
    rows = []

    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

    total = len(phantoms) * len(ITERATION_COUNTS)
    done  = 0

    for phantom_name, phantom in phantoms:
        # sinogram computed once per phantom, reused across all iteration counts
        sinos = sinogram_volume(phantom, angles, noise_fraction=NOISE)

        for n_iter in ITERATION_COUNTS:
            recon = reconstruct_volume_sirt(sinos, angles, n_iter)

            row = {
                "phantom":    phantom_name,
                "n_angles":   N_ANGLES,
                "noise":      NOISE,
                "n_iter":     n_iter,
                "rmse":       rmse(phantom, recon),
                "ssim":       ssim_score(phantom, recon),
                "dice":       dice_score(phantom, recon),
            }
            rows.append(row)
            done += 1
            print(
                f"[{done}/{total}] {phantom_name} | {n_iter} iterations | "
                f"RMSE={row['rmse']:.4f}  SSIM={row['ssim']:.4f}  Dice={row['dice']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()