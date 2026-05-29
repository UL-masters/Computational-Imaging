import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

INPUT_CSV  = "results/grid_results.csv"
OUTPUT_DIR = "figures"

ALG_COLORS = {
    "FBP":     "#e6194b",
    "SIRT":    "#3cb44b",
    "SIRT-TV": "#4363d8",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# read the CSV file with results and print some basic info about the dataset
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")
print(df[["algorithm", "n_angles", "noise"]].drop_duplicates().to_string(index=False))



fig, ax = plt.subplots(figsize=(7, 5))

# plot all points with some transparency to show density, and use different colors for each algorithm
for alg, grp in df.groupby("algorithm"):
    ax.scatter(
        grp["rmse"], grp["dice"],
        color=ALG_COLORS[alg],
        alpha=0.55, s=30, label=alg, edgecolors="none",
    )

# highlight the "hard" cases with a star marker and black edge
hard = df[(df["n_angles"] == 10) & (df["noise"] == 0.05)]
for alg, grp in hard.groupby("algorithm"):
    ax.scatter(
        grp["rmse"], grp["dice"],
        color=ALG_COLORS[alg],
        s=80, marker="*", edgecolors="black", linewidths=0.5,
        zorder=5,
    )

ax.set_xlabel("RMSE", fontsize=12)
ax.set_ylabel("Dice score", fontsize=12)
ax.set_title("RMSE vs Dice score\n(* = 10 angles, high noise)", fontsize=12)
ax.legend(title="Algorithm", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "scatter_rmse_vs_dice.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved {path}")


# compute Pearson correlation between RMSE and Dice score for each algorithm and condition
def condition_group(row):
    if row["n_angles"] >= 90 and row["noise"] <= 0.01:
        return "easy"
    elif row["n_angles"] <= 10 or row["noise"] >= 0.05:
        return "hard"
    else:
        return "medium"

df["condition"] = df.apply(condition_group, axis=1)

# compute correlation for each algorithm and condition, and store results in a new DataFrame for display
rows = []
for alg in df["algorithm"].unique():
    for cond in ["easy", "medium", "hard", "all"]:
        subset = df[df["algorithm"] == alg] if cond == "all" else \
                 df[(df["algorithm"] == alg) & (df["condition"] == cond)]
        if len(subset) < 3:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(subset["rmse"], subset["dice"])
        rows.append({
            "Algorithm": alg,
            "Condition": cond,
            "Pearson r(RMSE, Dice)": f"{r:.3f}",
            "p-value": f"{p:.3f}" if not np.isnan(p) else "—",
        })

# create a DataFrame for the correlation results and print it in a nice format
corr_df = pd.DataFrame(rows)
print("\nCorrelation table:")
print(corr_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis("off")
table = ax.table(
    cellText=corr_df.values,
    colLabels=corr_df.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# style the header row with a dark background and white bold text
for j in range(len(corr_df.columns)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# color the rows based on the algorithm, alternating between two light colors for better readability
alg_list = corr_df["Algorithm"].unique()
palette  = ["#f0f4f8", "#dbe8f5"]
for i, row in corr_df.iterrows():
    color = palette[list(alg_list).index(row["Algorithm"]) % 2]
    for j in range(len(corr_df.columns)):
        table[i + 1, j].set_facecolor(color)

ax.set_title(
    "Pearson correlation between RMSE and Dice score",
    fontsize=11, pad=12,
)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "correlation_table.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path}")



angle_counts = sorted(df["n_angles"].unique())
noise_levels = sorted(df["noise"].unique())

# helper function to create 2D arrays for heatmaps of RMSE and Dice score by angle count and noise level
def make_heatmap_arrays(df_sub, metric):
    arr = np.zeros((len(angle_counts), len(noise_levels)))
    for i, na in enumerate(angle_counts):
        for j, nl in enumerate(noise_levels):
            vals = df_sub[(df_sub["n_angles"] == na) & (df_sub["noise"] == nl)][metric]
            arr[i, j] = vals.mean() if len(vals) > 0 else np.nan
    return arr


# create heatmaps comparing SIRT and SIRT-TV for RMSE and Dice score across all conditions
for metric, cmap, title_suffix in [
    ("rmse", "Reds",   "RMSE (lower is better)"),
    ("dice", "Greens", "Dice score (higher is better)"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    vmin = min(
        df[df["algorithm"] == "SIRT"][metric].min(),
        df[df["algorithm"] == "SIRT-TV"][metric].min(),
    )
    vmax = max(
        df[df["algorithm"] == "SIRT"][metric].max(),
        df[df["algorithm"] == "SIRT-TV"][metric].max(),
    )

    for ax, alg in zip(axes, ["SIRT", "SIRT-TV"]):
        arr = make_heatmap_arrays(df[df["algorithm"] == alg], metric)
        im  = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="auto", origin="lower")
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{n:.2f}" for n in noise_levels])
        ax.set_yticks(range(len(angle_counts)))
        ax.set_yticklabels(angle_counts)
        ax.set_xlabel("Noise fraction $\\eta$", fontsize=11)
        ax.set_ylabel("Number of angles $N_\\theta$", fontsize=11)
        ax.set_title(alg, fontsize=12, fontweight="bold")

        for i in range(len(angle_counts)):
            for j in range(len(noise_levels)):
                val = arr[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if val > (vmin + vmax) / 2 else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"SIRT vs SIRT-TV — {title_suffix}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"heatmap_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

print("\nAll figures saved to figures/")
