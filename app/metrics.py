"""The one place metrics are defined, so baselines and models cannot drift apart.

Every number in every results table comes from `evaluate()` here. If a model
reports an RMSE computed its own way, the comparison with the baselines is
worthless, so nothing else is allowed to compute these.

WHAT EACH NUMBER IS FOR
-----------------------
  rmse            the headline. In log units, directly comparable with the
                  noise floor in rule 3 (0.57 on a difference). An RMSE below
                  that is a leakage suspect, not a triumph.
  mae             less sensitive to a few badly wrong pairs than RMSE, so the
                  gap between them says how heavy the error tail is.
  spearman_rho    does it get the ORDER right - would it rank a good swap
                  above a bad one? Undefined (NaN) when a model predicts the
                  same value for everything, which is exactly what the zero
                  baseline does.
  sign_accuracy   on pairs where something really happened (|delta| > 0.1),
                  does it get the DIRECTION right? This is the question a
                  chemist actually asks: will this swap help or hurt?
  mcc_three_class worse / same / better, cut at -1 and +1 log units. A tenfold
                  change in potency is the kind of difference that decides
                  whether a compound is worth making. MCC rather than accuracy
                  because the classes are very unbalanced: most swaps do
                  little, so predicting "same" for everything scores well on
                  accuracy and zero on MCC.
  rmse_<bin>      RMSE split by how big the true change was. A model can look
                  good overall purely by being right about the many small
                  changes while missing every large one, which is the opposite
                  of useful.

SIGN CONVENTIONS THAT MATTER
----------------------------
delta = pIC50(B) - pIC50(A), so positive means B is MORE potent than A.
A predictor of exactly 0 has no direction, so sign_accuracy counts it as
wrong rather than as half right: sign(0) matches no real sign. That makes the
zero baseline score 0.0 here, which is the honest reading - it has no
directional information at all - and is why spearman_rho is NaN for it too
rather than 0, since "no ordering" is not the same as "ordering uncorrelated".
"""

import math

import numpy as np
import pandas as pd

# Class edges in log units: a tenfold change either way.
WORSE_BELOW = -1.0
BETTER_ABOVE = 1.0
CLASS_NAMES = ["worse", "same", "better"]

# Pairs below this |delta| are treated as "nothing happened" for sign accuracy,
# because the experimental noise floor swamps them (rule 3).
SIGN_THRESHOLD = 0.1

# |delta| bins for stratified RMSE.
DELTA_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf)]


def bin_label(low, high):
    return f"{low:g}-{high:g}" if high != math.inf else f"{low:g}+"


def three_class(values):
    """worse (< -1) / same / better (> +1) as integers -1, 0, 1."""
    values = np.asarray(values, dtype=float)
    classes = np.zeros(len(values), dtype=int)
    classes[values < WORSE_BELOW] = -1
    classes[values > BETTER_ABOVE] = 1
    return classes


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_rho(y_true, y_pred):
    """Rank correlation. NaN when either side is constant - a model that
    predicts one value for everything has produced no ordering to score."""
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3:
        return float("nan")
    if np.all(y_pred == y_pred[0]) or np.all(y_true == y_true[0]):
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def sign_accuracy(y_true, y_pred, threshold=SIGN_THRESHOLD):
    """Fraction of real changes whose direction is predicted correctly.

    Only pairs with |true delta| > threshold count: below that the measurement
    noise is larger than the effect, so the sign is not a meaningful target.
    A prediction of exactly 0 has no sign and is counted as wrong.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    real = np.abs(y_true) > threshold
    if not real.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[real]) == np.sign(y_true[real])))


def mcc_three_class(y_true, y_pred):
    """Matthews correlation over worse / same / better.

    0 means no better than guessing the class distribution, which is what a
    constant predictor gets however well it does on RMSE.
    """
    from sklearn.metrics import matthews_corrcoef

    if len(y_true) == 0:
        return float("nan")
    true_classes = three_class(y_true)
    pred_classes = three_class(y_pred)
    if len(set(true_classes)) == 1 and len(set(pred_classes)) == 1:
        return float("nan")
    return float(matthews_corrcoef(true_classes, pred_classes))


def stratified_rmse(y_true, y_pred, bins=DELTA_BINS):
    """RMSE within each |true delta| band, plus how many pairs are in each.

    Returns {label: {"rmse": ..., "n": ...}}. A band with no pairs gets NaN
    rather than being left out, so every results table has the same columns.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    magnitude = np.abs(y_true)
    out = {}
    for low, high in bins:
        inside = (magnitude >= low) & (magnitude < high)
        label = bin_label(low, high)
        out[label] = {"rmse": rmse(y_true[inside], y_pred[inside]),
                      "n": int(inside.sum())}
    return out


# How many standard errors of improvement count as real. 2 is the usual
# two-sigma convention and is deliberately not tunable per run.
SIGNIFICANT_SIGMA = 2.0


def transferable_between_cell_variance(train_values, test_values):
    """How much cell structure SURVIVES a split, as a variance component.

    WHY THIS EXISTS, AND WHAT IT REPLACES
    ----------------------------------------------------
    The usual between-group variance component - ANOVA on one pool of data -
    answers "how much of the variance lies between cells here". Offered as a
    ceiling for a model it is wrong whenever the cell key is correlated with
    the thing the split holds out, because it counts variation that will not
    be there at test time. That mistake was made once on this project: the local
    environment scored a 37.2% component in-pool, and across held-out series
    the real figure was near zero.

    This estimator cannot make that mistake, because it never uses the same
    rows twice. For each cell present on both sides it takes the cell's
    deviation from its class mean in TRAINING and the same deviation in TEST,
    and averages their product over test rows:

        tau^2 = sum_c n_test(c) * dev_train(c) * dev_test(c) / sum_c n_test(c)

    The two deviations carry independent noise, so the noise contributes zero
    on average instead of inflating the estimate: a cell structure that is
    pure sampling noise gives about 0, and one that transfers gives its real
    variance. It can come out NEGATIVE, which is informative rather than a
    bug - it means training cell deviations point the wrong way at test time.

    Each argument is {cell key: (class key, values)}. Returns the component,
    the share of test variance it represents, and how many cells matched.
    """
    train_by_class, test_by_class = {}, {}
    for store, source in ((train_by_class, train_values),
                          (test_by_class, test_values)):
        for cell, (class_key, values) in source.items():
            store.setdefault(class_key, []).extend(values)
    train_class_mean = {key: float(np.mean(values))
                        for key, values in train_by_class.items()}
    test_class_mean = {key: float(np.mean(values))
                       for key, values in test_by_class.items()}

    all_test = [value for _cell, (_key, values) in test_values.items()
                for value in values]
    if len(all_test) < 2:
        return {"tau": 0.0, "share": 0.0, "cells_matched": 0, "n_test": len(all_test)}
    total = float(np.var(np.asarray(all_test, dtype=float), ddof=0))

    numerator, weight, matched = 0.0, 0.0, 0
    for cell, (class_key, test_cell_values) in test_values.items():
        if cell not in train_values:
            continue
        train_class_key, train_cell_values = train_values[cell]
        if class_key not in train_class_mean or class_key not in test_class_mean:
            continue
        train_deviation = float(np.mean(train_cell_values)) - \
            train_class_mean[class_key]
        test_deviation = float(np.mean(test_cell_values)) - test_class_mean[class_key]
        n_test = len(test_cell_values)
        numerator += n_test * train_deviation * test_deviation
        weight += n_test
        matched += 1
    tau = numerator / weight if weight else 0.0
    return {"tau": tau,
            "share": tau / total if total > 0 else 0.0,
            "cells_matched": matched,
            "n_test": len(all_test),
            "test_rows_in_matched_cells": int(weight),
            "total_variance": total}


BOOTSTRAP_DRAWS = 1000


def _cluster_bootstrap_se(difference, groups, draws, seed):
    """Standard error of mean(difference) when rows come in clusters.

    Resamples CLUSTERS with replacement, not rows. A cluster is
    one comparison - the same two molecules measured on several targets - whose
    rows share both molecules and are therefore correlated. Treating them as
    independent understates the standard error.

    Done on per-cluster sums and counts rather than rows, so a draw costs one
    pass over the clusters however many rows they hold.
    """
    order = np.argsort(groups, kind="stable")
    sorted_groups = np.asarray(groups)[order]
    sorted_difference = np.asarray(difference, dtype=float)[order]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    sums = np.add.reduceat(sorted_difference, starts)
    counts = np.diff(np.r_[starts, len(sorted_difference)]).astype(float)
    n_clusters = len(sums)
    if n_clusters < 2:
        return float("nan"), n_clusters

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(draws, n_clusters))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(means.std(ddof=1)), n_clusters


def beats_paired(y_true, y_pred, y_reference, sigma=SIGNIFICANT_SIGMA,
                 groups=None, draws=BOOTSTRAP_DRAWS, seed=42):
    """Is y_pred's error RELIABLY lower than y_reference's on these same rows?

    WHY A PAIRED TEST AND NOT JUST A LOWER RMSE: a model can
    post a lower RMSE than the constant it is being compared against and still
    be indistinguishable from it. That happened - `nearest_swap_class_median`
    scored 0.8029 on transformation fold 0 against the zero baseline's 0.8029,
    identical to four decimals, and was announced as the split's best
    non-degenerate baseline. Comparing two aggregate numbers cannot tell that
    apart from a real improvement.

    Both models predict the same test rows, so the comparison is paired: take
    the per-row difference of squared errors, d_i = e_model,i^2 -
    e_reference,i^2, and ask whether its mean is below zero by more than
    `sigma` standard errors. Paired, so the (large) row-to-row variation in how
    hard each pair is cancels instead of swamping the difference.

    With `groups` - one comparison id per row - the standard error comes from a
    CLUSTER BOOTSTRAP over comparisons instead of the independent-rows formula. The rows of one comparison share both molecules, so the
    naive standard error is optimistic and the 2-sigma bar is softer than it
    looks. Without `groups` the independent-rows formula is used and the result
    says so in `standard_error_method`.

    Returns a dict rather than a bare bool so a results table can carry the
    numbers that justify the verdict.
    """
    y_true = np.asarray(y_true, dtype=float)
    model_error = (np.asarray(y_pred, dtype=float) - y_true) ** 2
    reference_error = (np.asarray(y_reference, dtype=float) - y_true) ** 2
    difference = model_error - reference_error
    n = len(difference)
    if n < 2:
        return {"mse_gain": 0.0, "standard_error": float("nan"),
                "sigmas": 0.0, "beats_reference": False, "n_compared": n,
                "standard_error_method": "none", "n_clusters": n}
    mean = float(difference.mean())
    method, n_clusters = "independent_rows", n
    if groups is not None:
        standard_error, n_clusters = _cluster_bootstrap_se(difference, groups,
                                                           draws, seed)
        method = "comparison_bootstrap"
        if not np.isfinite(standard_error):
            standard_error = float(difference.std(ddof=1) / math.sqrt(n))
            method = "independent_rows"
    else:
        standard_error = float(difference.std(ddof=1) / math.sqrt(n))
    # All-zero difference (identical predictions) has no standard error, and is
    # exactly the case this function exists to reject.
    sigmas = 0.0 if standard_error == 0 else -mean / standard_error
    return {"mse_gain": -mean,
            "standard_error": standard_error,
            "sigmas": float(sigmas),
            "beats_reference": bool(sigmas >= sigma),
            "n_compared": n,
            "standard_error_method": method,
            "n_clusters": int(n_clusters)}


def evaluate(y_true, y_pred, bins=DELTA_BINS):
    """Every metric for one set of predictions, as a flat dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows, y_pred has {len(y_pred)}")
    scores = {
        "n": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "sign_accuracy": sign_accuracy(y_true, y_pred),
        "mcc_three_class": mcc_three_class(y_true, y_pred),
    }
    for label, values in stratified_rmse(y_true, y_pred, bins).items():
        scores[f"rmse_{label}"] = values["rmse"]
        scores[f"n_{label}"] = values["n"]
    return scores


def metrics_table(named_predictions, y_true=None):
    """One row per model, columns in a fixed order, for the results table.

    named_predictions: {name: (y_true, y_pred)}, or {name: y_pred} together
    with a shared y_true.
    """
    rows = []
    for name, value in named_predictions.items():
        if y_true is None:
            truth, prediction = value
        else:
            truth, prediction = y_true, value
        rows.append({"model": name, **evaluate(truth, prediction)})
    table = pd.DataFrame(rows)
    ordered = ["model", "n", "rmse", "mae", "spearman_rho", "sign_accuracy",
               "mcc_three_class"]
    ordered += [f"rmse_{bin_label(low, high)}" for low, high in DELTA_BINS]
    ordered += [f"n_{bin_label(low, high)}" for low, high in DELTA_BINS]
    return table[[column for column in ordered if column in table.columns]]


def format_table(table, float_format="{:.3f}"):
    """Readable fixed-width text, with NaN shown as 'n/a' rather than blank."""
    shown = table.copy()
    for column in shown.columns:
        if shown[column].dtype.kind == "f":
            shown[column] = [float_format.format(v) if pd.notna(v) else "n/a"
                             for v in shown[column]]
    return shown.to_string(index=False)
