import os
import glob
import numpy as np
import pandas as pd
from reconstruct import (
    sinogram_volume,
    reconstruct_volume_fbp,
    reconstruct_volume_sirt,
    rmse, ssim_score, dice_score,
    SIRT_ITERATIONS,
)
from experiment_grid import has_stone, load_phantoms

PHANTOMS_DIR = "phantoms"
OUTPUT_DIR   = "results"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "angle_sweep_results.csv")

N_PHANTOMS   = 20 # only a subset of phantoms to speed up the experiments
NOISE        = 0.00   # noiseless: isolates the effect of angle count only
ALGORITHMS   = ["FBP", "SIRT"]

# Fine-grained sweep as specified in the experimental plan
ANGLE_COUNTS = [5, 10, 20, 30, 60, 90, 120, 180]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phantoms = load_phantoms(PHANTOMS_DIR, N_PHANTOMS)
    rows = []

    total = len(phantoms) * len(ANGLE_COUNTS) * len(ALGORITHMS)
    done  = 0

    for phantom_name, phantom in phantoms:
        for n_angles in ANGLE_COUNTS:
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

            # sinogram computed once per (phantom, angles) and shared by both algorithms
            sinos = sinogram_volume(phantom, angles, noise_fraction=NOISE)

            for algorithm in ALGORITHMS:
                if algorithm == "FBP":
                    recon = reconstruct_volume_fbp(sinos, angles)
                else:
                    recon = reconstruct_volume_sirt(sinos, angles, SIRT_ITERATIONS)

                row = {
                    "phantom":   phantom_name,
                    "algorithm": algorithm,
                    "n_angles":  n_angles,
                    "noise":     NOISE,
                    "rmse":      rmse(phantom, recon),
                    "ssim":      ssim_score(phantom, recon),
                    "dice":      dice_score(phantom, recon),
                }
                rows.append(row)
                done += 1
                print(
                    f"[{done}/{total}] {phantom_name} | {algorithm} | "
                    f"{n_angles} angles | "
                    f"RMSE={row['rmse']:.4f}  SSIM={row['ssim']:.4f}  Dice={row['dice']:.4f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()