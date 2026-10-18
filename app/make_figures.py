"""Draw the two result figures from the published per-row predictions.

Reads:  results/ensemble_predictions_constant_allpc_e3.csv
        results/baselines_disjoint_molecules.csv
Writes: results/predicted_vs_measured.png
        results/rmse_by_fold.png

Run:    python app/make_figures.py

Nothing here chooses anything; it is the Results table drawn twice. Both figures
are regenerated from the published files, so they cannot drift from the numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.metrics import evaluate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BLEND_WEIGHT = 0.5
NOISE_FLOOR = 0.55


def load():
    rows = pd.read_csv(REPO_ROOT / "results/ensemble_predictions_constant_allpc_e3.csv")
    baselines = pd.read_csv(REPO_ROOT / "results/baselines_disjoint_molecules.csv")
    baselines = baselines[(baselines["split"] == "constant")
                          & (baselines["censoring"] == "exact")
                          & (baselines["series"] == "loose")
                          & (baselines["molecules"] == "disjoint")]
    return rows, baselines


def blended(frame):
    return BLEND_WEIGHT * frame["ensemble"] + (1 - BLEND_WEIGHT) * frame["shrunk_target_mean"]


def predicted_vs_measured(rows, path):
    truth = rows["truth"].to_numpy(dtype=float)
    predicted = blended(rows).to_numpy(dtype=float)
    scores = evaluate(rows["truth"], pd.Series(predicted))
    # The headline is the MEAN OF THE FIVE per-fold RMSEs, which is not the same
    # as one RMSE over the pooled rows: the pool weights a fold by its size.
    # Both are printed so the figure can never be read as contradicting the table.
    per_fold = [evaluate(sub["truth"], blended(sub))["rmse"]
                for _, sub in rows.groupby("fold")]
    headline = float(np.mean(per_fold))

    figure, axes = plt.subplots(figsize=(6.4, 6.0))
    extent = 5.0
    axes.hexbin(truth, predicted, gridsize=55, bins="log", cmap="Blues",
                extent=(-extent, extent, -extent, extent))
    axes.plot([-extent, extent], [-extent, extent], color="0.3", linewidth=1,
              label="perfect prediction")
    for sign in (1, -1):
        axes.plot([-extent, extent],
                  [-extent + sign * headline, extent + sign * headline],
                  color="0.3", linewidth=0.8, linestyle="--",
                  label=f"± headline RMSE ({headline:.2f} log units)" if sign == 1 else None)
    axes.set_xlim(-extent, extent)
    axes.set_ylim(-extent, extent)
    axes.set_xlabel("measured Δp (log units)")
    axes.set_ylabel("predicted Δp (log units)")
    axes.set_title(f"Held-out predictions, all five folds pooled\n"
                   f"{len(rows):,} comparisons — headline RMSE {headline:.4f} "
                   f"(mean of the five folds); {scores['rmse']:.4f} pooled")
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def rmse_by_fold(rows, baselines, path):
    folds = sorted(rows["fold"].unique())
    series = {"blend (headline)": [], "ensemble alone": [], "shrunk_target_mean": [],
              "transformation median": [], "predict no change": []}
    base = baselines.pivot(index="fold", columns="model", values="rmse")
    for fold in folds:
        sub = rows[rows["fold"] == fold]
        blended = BLEND_WEIGHT * sub["ensemble"] + (1 - BLEND_WEIGHT) * sub["shrunk_target_mean"]
        series["blend (headline)"].append(evaluate(sub["truth"], blended)["rmse"])
        series["ensemble alone"].append(evaluate(sub["truth"], sub["ensemble"])["rmse"])
        series["shrunk_target_mean"].append(base.loc[fold, "shrunk_target_mean"])
        series["transformation median"].append(base.loc[fold, "transformation_median"])
        series["predict no change"].append(base.loc[fold, "zero"])

    figure, axes = plt.subplots(figsize=(8.4, 4.6))
    width = 0.16
    positions = np.arange(len(folds))
    for index, (label, values) in enumerate(series.items()):
        axes.bar(positions + (index - 2) * width, values, width, label=label)
    axes.axhline(NOISE_FLOOR, color="crimson", linewidth=1, linestyle=":",
                 label=f"between-lab noise floor ({NOISE_FLOOR})")
    axes.set_xticks(positions)
    axes.set_xticklabels([f"fold {fold}\n{(rows['fold'] == fold).sum():,} pairs"
                          for fold in folds])
    axes.set_ylabel("RMSE (log units), lower is better")
    axes.set_title("Held-out RMSE per fold, model against the required baselines")
    axes.legend(fontsize=8, ncol=2)
    axes.set_ylim(0, 1.35)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    rows, baselines = load()
    first = REPO_ROOT / "results/predicted_vs_measured.png"
    second = REPO_ROOT / "results/rmse_by_fold.png"
    predicted_vs_measured(rows, first)
    rmse_by_fold(rows, baselines, second)
    print(f"{len(rows):,} comparisons read from the published predictions")
    print(f"wrote {first.relative_to(REPO_ROOT)}")
    print(f"wrote {second.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
