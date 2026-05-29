"""
analyse_sweep_and_iterations.py
--------------------------------
Produces plots for Subquestions 1 and 3:

  Q1 — Angle sweep 
       - Line plot: RMSE vs number of angles (FBP vs SIRT)
       - Line plot: Dice score vs number of angles (FBP vs SIRT)

  Q3 — SIRT iteration convergence 
       - Line plot: RMSE vs iteration count (log x-axis)
       - Line plot: Dice score vs iteration count (log x-axis)

All lines show the mean across phantoms; shaded bands show ±1 std.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANGLE_SWEEP_CSV  = "results/angle_sweep_results.csv"
ITERATIONS_CSV   = "results/sirt_iterations_results.csv"
OUTPUT_DIR       = "figures"

ALG_COLORS = {
    "FBP":  "#e6194b",
    "SIRT": "#3cb44b",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_metric_with_std(ax, df, x_col, metric, group_col, colors,
                         x_label, y_label, title, log_x=False):
    """
    Plot mean +/- std of `metric` against `x_col`, grouped by `group_col`.
    Each group gets its own line and shaded band.
    """
    for group, grp_df in df.groupby(group_col):
        stats = (
            grp_df.groupby(x_col)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        x    = stats[x_col].values
        mean = stats["mean"].values
        std  = stats["std"].values
        color = colors.get(group, "#555555")

        ax.plot(x, mean, marker="o", color=color, linewidth=2, label=group)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    if log_x:
        ax.set_xscale("log")
        # show the actual iteration values as tick labels
        ax.set_xticks(df[x_col].unique())
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Algorithm", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)


# ----- Angle Sweep ------ 
df_sweep = pd.read_csv(ANGLE_SWEEP_CSV)
print(f"Loaded {len(df_sweep)} rows from {ANGLE_SWEEP_CSV}")

for metric, y_label, title, fname in [
    ("rmse", "RMSE",       "Reconstruction error vs number of angles\n(noiseless, mean +/- 1 std)", "angle_sweep_rmse.png"),
    ("dice", "Dice score", "Foreign object detectability vs number of angles\n(noiseless, mean +/- 1 std)", "angle_sweep_dice.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_metric_with_std(
        ax, df_sweep,
        x_col     = "n_angles",
        metric    = metric,
        group_col = "algorithm",
        colors    = ALG_COLORS,
        x_label   = "Number of angles $N_\\theta$",
        y_label   = y_label,
        title     = title,
        log_x     = False,
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----- SIRT Iterations ------
df_iter = pd.read_csv(ITERATIONS_CSV)
print(f"Loaded {len(df_iter)} rows from {ITERATIONS_CSV}")

# add a dummy group column so plot_metric_with_std can be reused as-is
df_iter["algorithm"] = "SIRT"

for metric, y_label, title, fname in [
    ("rmse", "RMSE",       "SIRT convergence: reconstruction error\n(180 angles, noiseless, mean +/- 1 std)", "sirt_iterations_rmse.png"),
    ("dice", "Dice score", "SIRT convergence: foreign object detectability\n(180 angles, noiseless, mean +/- 1 std)", "sirt_iterations_dice.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_metric_with_std(
        ax, df_iter,
        x_col     = "n_iter",
        metric    = metric,
        group_col = "algorithm",
        colors    = ALG_COLORS,
        x_label   = "Number of SIRT iterations",
        y_label   = y_label,
        title     = title,
        log_x     = True,   # iteration counts are logarithmically spaced
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


print("\nAll figures saved to figures/")