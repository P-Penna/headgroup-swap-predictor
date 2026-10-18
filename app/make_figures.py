"""Draw the result figures from the published per-row predictions.

Reads:  results/ensemble_predictions_constant_allpc_e3.csv
        results/baselines_disjoint_molecules.csv
Writes: results/predicted_vs_measured.{png,pdf}
        results/rmse_by_fold.{png,pdf}

Run:    python app/make_figures.py

Nothing here chooses anything; it is the published table drawn twice. Both
figures are regenerated from the published files, so they cannot drift from the
numbers, and both are written as PNG at 400 dpi for the README and as vector
PDF for print.

STYLE
-----
Figures are sized to journal columns (89 mm single, 183 mm double), set in
Helvetica at 7-9 pt, and drawn with hairline rules. Explanation belongs in the
caption, not inside the axes, so there are no titles and no prose on the plot.
The categorical palette is Okabe-Ito, which stays distinguishable under all
common forms of colour blindness and in greyscale print.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from app.metrics import evaluate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BLEND_WEIGHT = 0.5
NOISE_FLOOR = 0.55

MM = 1 / 25.4
SINGLE_COLUMN = 89 * MM
DOUBLE_COLUMN = 183 * MM

# Okabe-Ito, colour-blind safe. Blue carries the headline everywhere.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GREY = "#999999"
VERMILLION = "#D55E00"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,          # editable text in the vector output
    "ps.fonttype": 42,
}


def save(figure, stem):
    """Write the raster for the README and the vector for print."""
    for suffix in ("png", "pdf"):
        figure.savefig(REPO_ROOT / f"results/{stem}.{suffix}")
    plt.close(figure)


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


def headline_rmse(rows):
    """The mean of the five per-fold RMSEs, which is what every table quotes.

    Not the same as one RMSE over the pooled rows, which would weight a fold by
    its size.
    """
    return float(np.mean([evaluate(sub["truth"], blended(sub))["rmse"]
                          for _, sub in rows.groupby("fold")]))


def predicted_vs_measured(rows, stem):
    truth = rows["truth"].to_numpy(dtype=float)
    predicted = blended(rows).to_numpy(dtype=float)
    rmse = headline_rmse(rows)
    inside = float(np.mean(np.abs(predicted - truth) <= rmse))
    slope, intercept = np.polyfit(truth, predicted, 1)

    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN * 1.45, SINGLE_COLUMN * 1.34))
    limit = 4.0
    mesh = axes.hexbin(truth, predicted, gridsize=44, cmap="Blues", mincnt=1,
                       linewidths=0.0, norm=LogNorm(vmin=1),
                       extent=(-limit, limit, -limit, limit))
    axes.fill_between([-limit, limit], [-limit - rmse, limit - rmse],
                      [-limit + rmse, limit + rmse], color="0.5", alpha=0.10,
                      linewidth=0, zorder=0,
                      label=f"$\\pm$1 RMSE ({rmse:.2f}); {inside:.0%} of comparisons")
    axes.plot([-limit, limit], [-limit, limit], color="0.15", linewidth=0.9,
              zorder=3, label="Identity ($y = x$)")
    for sign in (1, -1):
        axes.plot([-limit, limit], [-limit + sign * rmse, limit + sign * rmse],
                  color="0.15", linewidth=0.5, linestyle=(0, (4, 3)), zorder=3)
    axes.plot([-limit, limit], [slope * -limit + intercept, slope * limit + intercept],
              color=VERMILLION, linewidth=1.3, zorder=4,
              label=f"Least-squares fit (slope {slope:.2f})")
    axes.axhline(0, color="0.85", linewidth=0.4, zorder=1)
    axes.axvline(0, color="0.85", linewidth=0.4, zorder=1)

    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.xaxis.set_major_locator(MultipleLocator(2))
    axes.yaxis.set_major_locator(MultipleLocator(2))
    axes.xaxis.set_minor_locator(MultipleLocator(1))
    axes.yaxis.set_minor_locator(MultipleLocator(1))
    axes.tick_params(which="minor", length=1.4, width=0.5)
    axes.set_xlabel("Measured $\\Delta p$ (log units)")
    axes.set_ylabel("Predicted $\\Delta p$ (log units)")
    axes.set_aspect("equal")
    for side in ("top", "right"):
        axes.spines[side].set_visible(True)
        axes.spines[side].set_linewidth(0.6)

    # Potency on the top edge, so a log unit does not have to be converted.
    fold = axes.secondary_xaxis("top")
    fold.set_xticks([-2, -1, 0, 1, 2])
    fold.set_xticklabels(["0.01$\\times$", "0.1$\\times$", "1$\\times$",
                          "10$\\times$", "100$\\times$"], fontsize=6.5, color="0.35")
    fold.tick_params(length=0, pad=1.5)
    fold.set_xlabel("Fold-change in potency", fontsize=7, color="0.35", labelpad=3)
    fold.spines["top"].set_visible(False)

    axes.legend(loc="upper left", handlelength=1.6, borderpad=0.2,
                labelspacing=0.35, borderaxespad=0.4)
    axes.text(0.975, 0.03, f"$n$ = {len(rows):,}", transform=axes.transAxes,
              ha="right", va="bottom", fontsize=7, color="0.35")

    bar = figure.colorbar(mesh, ax=axes, shrink=0.70, pad=0.025, aspect=22)
    bar.set_label("Comparisons per bin", fontsize=7.5, labelpad=4)
    bar.ax.tick_params(labelsize=6.5, width=0.6, length=2.2)
    bar.outline.set_linewidth(0.6)
    save(figure, stem)


def rmse_by_fold(rows, baselines, stem):
    folds = sorted(rows["fold"].unique())
    base = baselines.pivot(index="fold", columns="model", values="rmse")
    # All THREE required baselines, not a selection of them, with the stronger
    # shrunk_target_mean reference beside them.
    series = {
        "Blend (reported)": (BLUE, []),
        "Ensemble alone": (SKY, []),
        "Shrunk target mean": (GREEN, []),
        "Transformation median": (ORANGE, []),
        "Transformation $\\times$ environment median": (PURPLE, []),
        "No change ($\\Delta p = 0$)": (GREY, []),
    }
    for fold in folds:
        sub = rows[rows["fold"] == fold]
        series["Blend (reported)"][1].append(evaluate(sub["truth"], blended(sub))["rmse"])
        series["Ensemble alone"][1].append(evaluate(sub["truth"], sub["ensemble"])["rmse"])
        series["Shrunk target mean"][1].append(base.loc[fold, "shrunk_target_mean"])
        series["Transformation median"][1].append(base.loc[fold, "transformation_median"])
        series["Transformation $\\times$ environment median"][1].append(
            base.loc[fold, "transformation_environment_median"])
        series["No change ($\\Delta p = 0$)"][1].append(base.loc[fold, "zero"])
    for _, values in series.values():
        values.append(float(np.mean(values)))

    labels = [f"Fold {fold}\n$n$ = {(rows['fold'] == fold).sum():,}" for fold in folds]
    labels.append(f"Mean\n$n$ = {len(rows):,}")

    figure, axes = plt.subplots(figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.40))
    width = 0.13
    positions = np.arange(len(labels), dtype=float)
    positions[-1] += 0.40
    for index, (label, (colour, values)) in enumerate(series.items()):
        axes.bar(positions + (index - 2.5) * width, values, width, label=label,
                 color=colour, linewidth=0, zorder=2)
    axes.axhline(NOISE_FLOOR, color=VERMILLION, linewidth=0.8,
                 linestyle=(0, (4, 3)), zorder=3)
    axes.text(positions[-1] + 3.4 * width, NOISE_FLOOR + 0.03,
              f"Experimental noise floor ({NOISE_FLOOR})", ha="right", va="bottom",
              fontsize=6.5, color=VERMILLION)

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlim(positions[0] - 4 * width, positions[-1] + 3.8 * width)
    axes.set_ylabel("RMSE (log units)")
    axes.set_ylim(0, 1.30)
    axes.yaxis.set_major_locator(MultipleLocator(0.25))
    axes.yaxis.set_minor_locator(MultipleLocator(0.125))
    axes.tick_params(axis="x", length=0, pad=4)
    axes.tick_params(axis="y", which="minor", length=1.4, width=0.5)
    axes.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    axes.set_axisbelow(True)
    axes.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                handlelength=1.1, handleheight=0.9, columnspacing=1.6,
                labelspacing=0.4)
    save(figure, stem)


def main():
    rows, baselines = load()
    with plt.rc_context(STYLE):
        predicted_vs_measured(rows, "predicted_vs_measured")
        rmse_by_fold(rows, baselines, "rmse_by_fold")
    print(f"{len(rows):,} comparisons read from the published predictions")
    print(f"headline RMSE {headline_rmse(rows):.4f} (mean of the five folds)")
    for stem in ("predicted_vs_measured", "rmse_by_fold"):
        for suffix in ("png", "pdf"):
            print(f"wrote results/{stem}.{suffix}")


if __name__ == "__main__":
    main()
