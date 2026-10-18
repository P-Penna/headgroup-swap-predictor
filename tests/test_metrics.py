"""Tests for app/metrics.py, with answers worked out by hand.

Every later model is scored by these functions, so a wrong metric would make
every comparison in the project wrong in the same direction and be invisible.
"""

import math

import numpy as np
import pytest

from app.metrics import (
    BETTER_ABOVE,
    DELTA_BINS,
    WORSE_BELOW,
    evaluate,
    format_table,
    mae,
    mcc_three_class,
    metrics_table,
    rmse,
    sign_accuracy,
    spearman_rho,
    stratified_rmse,
    three_class,
)


# --- the simple two -------------------------------------------------------

def test_rmse_and_mae_by_hand():
    true = [0.0, 1.0, 2.0]
    pred = [0.0, 0.0, 0.0]
    assert rmse(true, pred) == pytest.approx(math.sqrt((0 + 1 + 4) / 3))
    assert mae(true, pred) == pytest.approx((0 + 1 + 2) / 3)


def test_perfect_prediction_scores_zero_error():
    true = [-2.0, 0.3, 1.7]
    assert rmse(true, true) == pytest.approx(0.0)
    assert mae(true, true) == pytest.approx(0.0)


def test_empty_input_is_nan_not_an_error():
    assert math.isnan(rmse([], []))
    assert math.isnan(mae([], []))


# --- three-class edges ----------------------------------------------------

def test_three_class_edges():
    values = [-2.0, WORSE_BELOW, -0.5, 0.0, 0.5, BETTER_ABOVE, 2.0]
    assert list(three_class(values)) == [-1, 0, 0, 0, 0, 0, 1]


def test_exactly_one_log_unit_counts_as_same():
    """The cut is strictly beyond +-1, so exactly 1.0 is not 'better'."""
    assert list(three_class([-1.0, 1.0])) == [0, 0]
    assert list(three_class([-1.001, 1.001])) == [-1, 1]


# --- Spearman -------------------------------------------------------------

def test_spearman_is_one_for_a_monotone_prediction():
    true = [0.1, 0.5, 1.0, 2.0]
    pred = [1.0, 2.0, 3.0, 9.0]            # same order, different values
    assert spearman_rho(true, pred) == pytest.approx(1.0)


def test_spearman_is_nan_for_a_constant_prediction():
    """The zero baseline produces no ordering, so there is nothing to correlate.
    NaN, not 0 - 'no ordering' is not 'uncorrelated ordering'."""
    assert math.isnan(spearman_rho([0.1, 0.5, 1.0], [0.0, 0.0, 0.0]))


def test_spearman_needs_three_points():
    assert math.isnan(spearman_rho([1.0, 2.0], [1.0, 2.0]))


# --- sign accuracy --------------------------------------------------------

def test_sign_accuracy_ignores_tiny_changes():
    """Only |true| > 0.1 counts: below that the noise floor swamps the effect."""
    true = [0.05, -0.05, 0.9, -0.9]        # first two are below threshold
    pred = [-9.0, 9.0, 0.5, -0.5]          # wrong on both ignored ones
    assert sign_accuracy(true, pred) == pytest.approx(1.0)


def test_sign_accuracy_counts_a_zero_prediction_as_wrong():
    """A prediction of 0 has no direction. The zero baseline scores 0.0."""
    assert sign_accuracy([1.0, -1.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_sign_accuracy_is_nan_when_nothing_happened():
    assert math.isnan(sign_accuracy([0.01, -0.02], [1.0, -1.0]))


def test_sign_accuracy_half_right():
    assert sign_accuracy([1.0, -1.0], [1.0, 1.0]) == pytest.approx(0.5)


# --- MCC ------------------------------------------------------------------

def test_mcc_is_one_for_perfect_three_class_prediction():
    true = [-2.0, 0.0, 2.0, -3.0, 0.1, 1.5]
    assert mcc_three_class(true, true) == pytest.approx(1.0)


def test_mcc_is_zero_for_a_constant_predictor():
    """Predicting 'same' for everything gets high accuracy and no MCC, which is
    the whole reason MCC is reported instead of accuracy."""
    true = [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    assert mcc_three_class(true, [0.0] * 6) == pytest.approx(0.0)


def test_mcc_is_nan_when_both_sides_are_one_class():
    assert math.isnan(mcc_three_class([0.0, 0.1], [0.0, 0.0]))


# --- stratified RMSE ------------------------------------------------------

def test_stratified_rmse_puts_pairs_in_the_right_bins():
    true = [0.2, 0.7, 1.5, 3.0]
    pred = [0.0, 0.0, 0.0, 0.0]
    out = stratified_rmse(true, pred)
    assert [out[label]["n"] for label in ["0-0.5", "0.5-1", "1-2", "2+"]] == [1, 1, 1, 1]
    assert out["0-0.5"]["rmse"] == pytest.approx(0.2)
    assert out["2+"]["rmse"] == pytest.approx(3.0)


def test_stratified_rmse_bin_edges_are_half_open():
    """0.5 belongs to 0.5-1, not 0-0.5."""
    out = stratified_rmse([0.5, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert out["0-0.5"]["n"] == 0
    assert out["0.5-1"]["n"] == 1
    assert out["1-2"]["n"] == 1
    assert out["2+"]["n"] == 1


def test_empty_bins_are_reported_as_nan_not_dropped():
    out = stratified_rmse([0.1], [0.0])
    assert set(out) == {"0-0.5", "0.5-1", "1-2", "2+"}
    assert math.isnan(out["2+"]["rmse"])
    assert out["2+"]["n"] == 0


# --- evaluate and the table ----------------------------------------------

def test_evaluate_returns_every_metric():
    scores = evaluate([0.2, 1.5, -2.0], [0.1, 1.0, -1.0])
    for key in ["n", "rmse", "mae", "spearman_rho", "sign_accuracy",
                "mcc_three_class"]:
        assert key in scores
    for low, high in DELTA_BINS:
        label = f"{low:g}-{high:g}" if high != math.inf else f"{low:g}+"
        assert f"rmse_{label}" in scores
        assert f"n_{label}" in scores


def test_evaluate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="rows"):
        evaluate([1.0, 2.0], [1.0])


def test_metrics_table_has_one_row_per_model_in_order():
    true = [0.2, 1.5, -2.0]
    table = metrics_table({"zero": (true, [0.0, 0.0, 0.0]),
                           "perfect": (true, true)})
    assert list(table["model"]) == ["zero", "perfect"]
    assert table.loc[table["model"] == "perfect", "rmse"].iloc[0] == pytest.approx(0.0)
    assert list(table.columns)[:3] == ["model", "n", "rmse"]


def test_metrics_table_accepts_a_shared_truth():
    true = [0.2, 1.5, -2.0]
    table = metrics_table({"zero": [0.0, 0.0, 0.0]}, y_true=true)
    assert len(table) == 1


def test_format_table_shows_nan_as_na():
    true = [0.2, 1.5, -2.0]
    text = format_table(metrics_table({"zero": (true, [0.0, 0.0, 0.0])}))
    assert "n/a" in text           # spearman is NaN for a constant prediction


def test_zero_baseline_metric_signature():
    """The combination that identifies a no-information predictor: finite RMSE,
    NaN Spearman, zero sign accuracy, zero MCC. Documented so nobody reads the
    RMSE alone and thinks it did well."""
    true = [0.2, -0.9, 1.5, -2.0, 0.05]
    scores = evaluate(true, np.zeros(len(true)))
    assert scores["rmse"] > 0
    assert math.isnan(scores["spearman_rho"])
    assert scores["sign_accuracy"] == pytest.approx(0.0)
    assert scores["mcc_three_class"] == pytest.approx(0.0)


# --- the paired comparison against a reference ----------------

def test_beats_paired_rejects_identical_predictions():
    """The exact case this exists for: nearest_swap_class_median scored 0.8029
    against the zero baseline's 0.8029 and was reported as a split's best."""
    from app.metrics import beats_paired
    truth = np.array([1.0, -1.0, 0.5, -0.5, 2.0])
    same = np.zeros(5)
    verdict = beats_paired(truth, same, same)
    assert verdict["beats_reference"] is False
    assert verdict["sigmas"] == 0.0
    assert verdict["mse_gain"] == 0.0


def test_beats_paired_accepts_a_real_improvement():
    from app.metrics import beats_paired
    rng = np.random.default_rng(42)
    truth = rng.normal(size=400)
    good = truth + rng.normal(scale=0.2, size=400)
    constant = np.zeros(400)
    verdict = beats_paired(truth, good, constant)
    assert verdict["beats_reference"] is True
    assert verdict["sigmas"] > 2.0
    assert verdict["mse_gain"] > 0


def test_beats_paired_rejects_a_tiny_gain_on_few_rows():
    """A barely-lower RMSE on a handful of rows is not evidence."""
    from app.metrics import beats_paired
    truth = np.array([1.0, -1.0, 0.5, -0.5])
    nearly = np.array([0.01, -0.01, 0.0, 0.0])
    constant = np.zeros(4)
    verdict = beats_paired(truth, nearly, constant)
    assert verdict["beats_reference"] is False


def test_beats_paired_handles_one_row():
    from app.metrics import beats_paired
    verdict = beats_paired([1.0], [0.9], [0.0])
    assert verdict["beats_reference"] is False
    assert verdict["n_compared"] == 1


# --- the comparison-level cluster bootstrap -------------------

def test_the_cluster_bootstrap_widens_the_error_bar():
    """The whole point: five rows of one comparison are not five independent
    observations, so clustering them must give a LARGER standard error than
    the independent-rows formula, and therefore fewer sigmas."""
    from app.metrics import beats_paired
    rng = np.random.default_rng(42)
    # 40 comparisons, each measured on 5 targets, with the error shared
    # within a comparison - exactly the correlation the clustering is for.
    per_comparison = rng.normal(scale=0.5, size=40)
    truth, model, groups = [], [], []
    for index, shared in enumerate(per_comparison):
        for _target in range(5):
            truth.append(shared + rng.normal(scale=0.05))
            model.append(shared * 0.5)
            groups.append(index)
    truth = np.asarray(truth)
    model = np.asarray(model)
    reference = np.zeros_like(truth)
    naive = beats_paired(truth, model, reference)
    clustered = beats_paired(truth, model, reference, groups=groups)
    assert clustered["standard_error"] > naive["standard_error"]
    assert clustered["sigmas"] < naive["sigmas"]
    assert clustered["standard_error_method"] == "comparison_bootstrap"
    assert naive["standard_error_method"] == "independent_rows"
    assert clustered["n_clusters"] == 40


def test_the_bootstrap_is_deterministic():
    from app.metrics import beats_paired
    rng = np.random.default_rng(7)
    truth = rng.normal(size=100)
    model = truth * 0.5
    groups = np.repeat(np.arange(20), 5)
    first = beats_paired(truth, model, np.zeros(100), groups=groups)
    second = beats_paired(truth, model, np.zeros(100), groups=groups)
    assert first["standard_error"] == second["standard_error"]


def test_the_bootstrap_still_rejects_identical_predictions():
    from app.metrics import beats_paired
    truth = np.array([1.0, -1.0, 0.5, -0.5, 2.0, 0.3])
    same = np.zeros(6)
    verdict = beats_paired(truth, same, same, groups=[0, 0, 1, 1, 2, 2])
    assert verdict["beats_reference"] is False


def test_one_cluster_falls_back_to_the_row_formula():
    """A single comparison cannot be resampled, so the clustered standard error
    is undefined and the row formula is used, with the method saying so."""
    from app.metrics import beats_paired
    verdict = beats_paired([1.0, 2.0, 3.0], [0.9, 1.9, 2.9], [0.0, 0.0, 0.0],
                           groups=[5, 5, 5])
    assert verdict["standard_error_method"] == "independent_rows"
    assert np.isfinite(verdict["standard_error"])


# --- the out-of-sample variance component ---------------------

def cells(spec):
    """{cell: (class, values)} from {cell: (class, values list)}."""
    return {key: (class_key, list(values)) for key, (class_key, values) in spec.items()}


def test_pure_noise_cells_score_about_zero():
    """THE property the in-pool estimator lacks. Cell structure that is only
    sampling noise must not be credited: train and test deviations are
    independent, so their product averages to nothing."""
    from app.metrics import transferable_between_cell_variance
    rng = np.random.default_rng(42)
    train, test = {}, {}
    for index in range(300):
        # Every cell has the SAME true mean; only noise distinguishes them.
        train[index] = ("g", rng.normal(scale=1.0, size=4).tolist())
        test[index] = ("g", rng.normal(scale=1.0, size=4).tolist())
    result = transferable_between_cell_variance(cells(train), cells(test))
    assert abs(result["share"]) < 0.05
    assert result["cells_matched"] == 300


def test_a_real_cell_effect_is_recovered():
    """Cells with genuinely different means, measured on both sides, must give
    back roughly their true between-cell variance."""
    from app.metrics import transferable_between_cell_variance
    rng = np.random.default_rng(42)
    train, test = {}, {}
    true_effects = rng.normal(scale=1.0, size=200)
    for index, effect in enumerate(true_effects):
        train[index] = ("g", (effect + rng.normal(scale=0.5, size=8)).tolist())
        test[index] = ("g", (effect + rng.normal(scale=0.5, size=8)).tolist())
    result = transferable_between_cell_variance(cells(train), cells(test))
    # True between variance is 1.0 against a total of about 1.25.
    assert result["tau"] == pytest.approx(1.0, abs=0.25)
    assert 0.6 < result["share"] < 1.0


def test_an_anticorrelated_cell_effect_goes_negative():
    """A negative component is informative, not an error: it says the training
    cell deviations point the wrong way at test time."""
    from app.metrics import transferable_between_cell_variance
    train = {0: ("g", [2.0, 2.0]), 1: ("g", [-2.0, -2.0])}
    test = {0: ("g", [-2.0, -2.0]), 1: ("g", [2.0, 2.0])}
    result = transferable_between_cell_variance(cells(train), cells(test))
    assert result["tau"] < 0


def test_deviations_are_taken_within_the_class():
    """A cell effect that is really a CLASS effect must not be counted, since
    the class mean is what the cell is being measured against."""
    from app.metrics import transferable_between_cell_variance
    train = {("a", 0): ("a", [5.0, 5.0]), ("b", 0): ("b", [-5.0, -5.0])}
    test = {("a", 0): ("a", [5.0, 5.0]), ("b", 0): ("b", [-5.0, -5.0])}
    result = transferable_between_cell_variance(cells(train), cells(test))
    assert result["tau"] == pytest.approx(0.0)


def test_cells_absent_from_training_are_skipped_not_counted():
    from app.metrics import transferable_between_cell_variance
    train = {0: ("g", [1.0, 1.0])}
    test = {0: ("g", [1.0, 1.0]), 1: ("g", [-1.0, -1.0])}
    result = transferable_between_cell_variance(cells(train), cells(test))
    assert result["cells_matched"] == 1
    assert result["test_rows_in_matched_cells"] == 2
    assert result["n_test"] == 4
