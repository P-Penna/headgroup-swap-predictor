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
from matplotlib.colors import LogNorm  # noqa: E402
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

    inside = float(np.mean(np.abs(predicted - truth) <= headline))

    figure, axes = plt.subplots(figsize=(7.6, 7.0))
    # Clipped to where the data actually is: 99.5% of comparisons sit inside
    # +-4, and a wider frame is mostly white space.
    extent = 4.0
    mesh = axes.hexbin(truth, predicted, gridsize=46, cmap="Blues", mincnt=1,
                       norm=LogNorm(vmin=1, vmax=None),
                       extent=(-extent, extent, -extent, extent))
    axes.fill_between([-extent, extent],
                      [-extent - headline, extent - headline],
                      [-extent + headline, extent + headline],
                      color="0.55", alpha=0.12, zorder=0,
                      label=f"within ± one RMSE ({headline:.2f}) — holds {inside:.0%} of points")
    axes.plot([-extent, extent], [-extent, extent], color="0.15", linewidth=1.4,
              label="perfect prediction", zorder=3)
    for sign in (1, -1):
        axes.plot([-extent, extent],
                  [-extent + sign * headline, extent + sign * headline],
                  color="0.15", linewidth=1.0, linestyle="--", zorder=3)
    axes.axhline(0, color="0.75", linewidth=0.7, zorder=1)
    axes.axvline(0, color="0.75", linewidth=0.7, zorder=1)
    axes.set_xlim(-extent, extent)
    axes.set_ylim(-extent, extent)
    axes.set_xticks(range(-4, 5))
    axes.set_yticks(range(-4, 5))
    axes.set_xlabel("measured Δp (log units)   →  B more potent")
    axes.set_ylabel("predicted Δp (log units)   →  B more potent")
    axes.set_title("Held-out predictions against measurement\n"
                   f"{len(rows):,} comparisons, all five test folds pooled",
                   fontsize=12)
    axes.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    axes.set_aspect("equal")

    bar = figure.colorbar(mesh, ax=axes, shrink=0.82, pad=0.02)
    bar.set_label("comparisons per hexagon (log scale)", fontsize=9)

    # The two RMSEs are different statistics and both get named, so the figure
    # can never be read as contradicting the tables.
    axes.text(0.985, 0.015,
              f"headline RMSE {headline:.4f} (mean of the five folds)\n"
              f"pooled RMSE {scores['rmse']:.4f} (all rows at once)\n"
              f"three-class MCC {scores['mcc_three_class']:+.3f}",
              transform=axes.transAxes, ha="right", va="bottom", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor="0.7", alpha=0.92))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def rmse_by_fold(rows, baselines, path):
    folds = sorted(rows["fold"].unique())
    # All THREE required baselines are shown, not a selection of them: predict
    # no change, the transformation median, and that median conditioned on the
    # local environment. shrunk_target_mean is the stronger reference beside them.
    series = {"blend (headline)": [], "ensemble alone": [], "shrunk_target_mean": [],
              "transformation median": [], "transformation × environment median": [],
              "predict no change": []}
    base = baselines.pivot(index="fold", columns="model", values="rmse")
    for fold in folds:
        sub = rows[rows["fold"] == fold]
        blended = BLEND_WEIGHT * sub["ensemble"] + (1 - BLEND_WEIGHT) * sub["shrunk_target_mean"]
        series["blend (headline)"].append(evaluate(sub["truth"], blended)["rmse"])
        series["ensemble alone"].append(evaluate(sub["truth"], sub["ensemble"])["rmse"])
        series["shrunk_target_mean"].append(base.loc[fold, "shrunk_target_mean"])
        series["transformation median"].append(base.loc[fold, "transformation_median"])
        series["transformation × environment median"].append(
            base.loc[fold, "transformation_environment_median"])
        series["predict no change"].append(base.loc[fold, "zero"])

    # The sixth group is the headline itself: the mean of the five folds, which
    # is the number every table quotes.
    for label in series:
        series[label].append(float(np.mean(series[label])))
    groups = [f"fold {fold}\n{(rows['fold'] == fold).sum():,} pairs" for fold in folds]
    groups.append(f"MEAN OF 5\n{len(rows):,} pairs")

    figure, axes = plt.subplots(figsize=(10.0, 5.4))
    width = 0.135
    positions = np.arange(len(groups), dtype=float)
    positions[-1] += 0.45          # set the headline group apart
    for index, (label, values) in enumerate(series.items()):
        bars = axes.bar(positions + (index - 2.5) * width, values, width, label=label,
                        edgecolor="white", linewidth=0.4)
        for bar_patch, value in zip(bars, values):
            axes.text(bar_patch.get_x() + bar_patch.get_width() / 2, value + 0.012,
                      f"{value:.3f}", ha="center", va="bottom", fontsize=5.8,
                      rotation=90, color="0.25")
    axes.axhline(NOISE_FLOOR, color="crimson", linewidth=1.2, linestyle=":",
                 zorder=4, label=f"between-lab noise floor ({NOISE_FLOOR}) — "
                                 f"no model on this data can beat it")
    axes.axvline(positions[-1] - 3.4 * width, color="0.8", linewidth=1)
    axes.set_xticks(positions)
    axes.set_xticklabels(groups, fontsize=9)
    axes.set_ylabel("RMSE (log units) — lower is better")
    axes.set_title("Held-out RMSE per test fold: the model against every required "
                   "baseline\nsame comparisons for every bar in a group",
                   fontsize=12)
    axes.set_ylim(0, 1.32)
    axes.set_yticks(np.arange(0, 1.31, 0.2))
    axes.grid(axis="y", color="0.9", linewidth=0.7, zorder=0)
    axes.set_axisbelow(True)
    # Outside the plot area, so it can never sit on top of a bar, and in the
    # order the bars appear rather than the order matplotlib collected them.
    handles, labels = axes.get_legend_handles_labels()
    order = [labels.index(name) for name in series] + \
            [i for i, name in enumerate(labels) if name not in series]
    axes.legend([handles[i] for i in order], [labels[i] for i in order],
                fontsize=8.5, ncol=4, loc="upper center",
                bbox_to_anchor=(0.5, -0.13), frameon=False)
    figure.savefig(path, dpi=150, bbox_inches="tight")
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
