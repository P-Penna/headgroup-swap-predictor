"""Tests for app/losses.py, above all the censoring direction.

Which way a censored measurement bounds Delta p is measurement logic, it is
easy to get backwards, and getting it backwards trains the model to move
predictions the WRONG side of a bound while every number in the log still
looks reasonable. So each of the sixteen (censored_a, censored_b)
combinations is pinned by name, and the hinge is checked to punish only
violations.
"""

import pytest
import torch

from app.join import CENSOR_LABELS
from app.losses import (
    CLASS_WEIGHT, DeltaLoss, EXACT, LOWER, MIXED, UPPER, assert_one_direction,
    bound_directions, class_mask, inverse_frequency_weights, regression_loss,
)
from app.model import BETTER, SAME, WORSE

ALL = (EXACT, UPPER, LOWER, MIXED)

# Worked out by hand from Delta p = pIC50(B) - pIC50(A), where `upper` means the
# recorded pchembl is an upper bound on the truth.
EXPECTED = {
    ("exact", "exact"): "exact",
    ("exact", "upper"): "above",    # B at most this potent -> delta at most c
    ("lower", "exact"): "above",    # A at least this potent -> delta at most c
    ("lower", "upper"): "above",
    ("upper", "exact"): "below",
    ("exact", "lower"): "below",
    ("upper", "lower"): "below",
    ("upper", "upper"): "none",     # both weak bounds: delta unconstrained
    ("lower", "lower"): "none",
}
for first in CENSOR_LABELS:
    EXPECTED.setdefault((first, "mixed"), "none")
    EXPECTED.setdefault(("mixed", first), "none")


def labels(pairs):
    a = torch.tensor([CENSOR_LABELS.index(first) for first, _ in pairs])
    b = torch.tensor([CENSOR_LABELS.index(second) for _, second in pairs])
    return a, b


# --- the censoring direction ------------------------------------------------

@pytest.mark.parametrize("combination,expected", sorted(EXPECTED.items()))
def test_every_censoring_combination_bounds_the_expected_way(combination, expected):
    censor_a, censor_b = labels([combination])
    exact, above, below = bound_directions(censor_a, censor_b)
    got = {"exact": exact, "above": above, "below": below}
    for name, mask in got.items():
        assert bool(mask[0]) == (name == expected), \
            f"{combination} should be {expected}, got {name}={bool(mask[0])}"


def test_all_sixteen_combinations_are_covered():
    assert len(EXPECTED) == len(CENSOR_LABELS) ** 2 == 16


def test_no_combination_is_bounded_both_ways():
    pairs = [(first, second) for first in CENSOR_LABELS for second in CENSOR_LABELS]
    censor_a, censor_b = labels(pairs)
    assert assert_one_direction(censor_a, censor_b)


def test_the_directions_mirror_each_other():
    """Swapping the two sides must swap `above` and `below` - the file holds
    both directions of every comparison, so an asymmetry here would train the
    two mirrors inconsistently."""
    pairs = [(first, second) for first in CENSOR_LABELS for second in CENSOR_LABELS]
    censor_a, censor_b = labels(pairs)
    _e, above, below = bound_directions(censor_a, censor_b)
    _e2, above_swapped, below_swapped = bound_directions(censor_b, censor_a)
    assert torch.equal(above, below_swapped)
    assert torch.equal(below, above_swapped)


# --- the regression loss ----------------------------------------------------

def test_exact_rows_use_huber_not_squared_error():
    censor_a, censor_b = labels([("exact", "exact")])
    prediction = torch.tensor([5.0])
    target = torch.tensor([0.0])
    per_row, weight = regression_loss(prediction, target, censor_a, censor_b)
    assert weight.item() == 1.0
    # Huber at error 5 with delta 1 is 5 - 0.5 = 4.5; squared error would be 25.
    assert per_row.item() == pytest.approx(4.5)


def test_a_bounded_above_row_is_free_below_its_bound_and_punished_above():
    censor_a, censor_b = labels([("exact", "upper")] * 3)
    target = torch.tensor([0.0, 0.0, 0.0])
    prediction = torch.tensor([-2.0, 0.0, 2.0])
    per_row, weight = regression_loss(prediction, target, censor_a, censor_b)
    assert torch.equal(weight, torch.ones(3))
    assert per_row[0].item() == 0.0, "predicting below an upper bound is allowed"
    assert per_row[1].item() == 0.0
    assert per_row[2].item() == pytest.approx(4.0), "2 above the bound, squared"


def test_a_bounded_below_row_is_the_mirror_image():
    censor_a, censor_b = labels([("upper", "exact")] * 2)
    target = torch.tensor([0.0, 0.0])
    per_row, _w = regression_loss(torch.tensor([2.0, -2.0]), target,
                                  censor_a, censor_b)
    assert per_row[0].item() == 0.0
    assert per_row[1].item() == pytest.approx(4.0)


def test_rows_with_no_usable_bound_get_zero_loss_and_zero_weight():
    censor_a, censor_b = labels([("upper", "upper"), ("lower", "lower"),
                                 ("mixed", "exact")])
    per_row, weight = regression_loss(torch.tensor([9.0, -9.0, 9.0]),
                                      torch.zeros(3), censor_a, censor_b)
    assert torch.equal(weight, torch.zeros(3))
    assert torch.equal(per_row, torch.zeros(3))


def test_an_unusable_row_cannot_change_the_batch_loss():
    """The whole point of weighting rather than filtering: adding a row that
    says nothing must not move the number."""
    weights = torch.ones(3)
    loss = DeltaLoss(weights)
    exact_a, exact_b = labels([("exact", "exact")])
    batch = {"delta": torch.tensor([0.5]), "censor_a": exact_a, "censor_b": exact_b,
             "label_class": torch.tensor([SAME])}
    output = {"delta": torch.tensor([0.1]), "logits": torch.zeros(1, 3)}

    padded_a, padded_b = labels([("exact", "exact"), ("upper", "upper")])
    padded_batch = {"delta": torch.tensor([0.5, 7.0]),
                    "censor_a": padded_a, "censor_b": padded_b,
                    "label_class": torch.tensor([SAME, WORSE])}
    padded_output = {"delta": torch.tensor([0.1, -7.0]), "logits": torch.zeros(2, 3)}

    assert loss(output, batch)["regression"].item() == pytest.approx(
        loss(padded_output, padded_batch)["regression"].item())


# --- which rows may train the classifier ------------------------------------

def test_class_mask_takes_every_exact_row():
    censor_a, censor_b = labels([("exact", "exact")] * 2)
    assert torch.equal(class_mask(torch.tensor([0.0, -3.0]), censor_a, censor_b),
                       torch.tensor([True, True]))


def test_a_bound_that_proves_the_class_is_used():
    """Bounded above by -2: the truth is at most -2, so `worse` is certain."""
    censor_a, censor_b = labels([("exact", "upper"), ("exact", "upper")])
    mask = class_mask(torch.tensor([-2.0, -0.5]), censor_a, censor_b)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False, "bounded above by -0.5 proves nothing"


def test_a_bound_that_proves_better_is_used():
    censor_a, censor_b = labels([("upper", "exact"), ("upper", "exact")])
    mask = class_mask(torch.tensor([2.0, 0.5]), censor_a, censor_b)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False


def test_an_unusable_row_never_trains_the_classifier():
    censor_a, censor_b = labels([("upper", "upper"), ("mixed", "mixed")])
    assert not class_mask(torch.tensor([-5.0, 5.0]), censor_a, censor_b).any()


# --- class weights ----------------------------------------------------------

def test_inverse_frequency_weights_favour_the_rare_classes():
    labels_tensor = torch.tensor([SAME] * 74 + [WORSE] * 13 + [BETTER] * 12)
    weights = inverse_frequency_weights(labels_tensor)
    assert weights[SAME] < weights[WORSE] < weights[BETTER]
    assert weights.mean().item() == pytest.approx(1.0)
    # Inverse frequency: the ratio of weights is the inverse ratio of counts.
    assert (weights[WORSE] / weights[SAME]).item() == pytest.approx(74 / 13)


def test_a_missing_class_does_not_give_an_infinite_weight():
    weights = inverse_frequency_weights(torch.tensor([SAME, SAME, WORSE]))
    assert torch.isfinite(weights).all()


# --- the combined loss ------------------------------------------------------

def test_total_is_regression_plus_weighted_classification():
    torch.manual_seed(0)
    censor_a, censor_b = labels([("exact", "exact")] * 4)
    batch = {"delta": torch.tensor([0.2, -1.5, 2.0, 0.0]),
             "censor_a": censor_a, "censor_b": censor_b,
             "label_class": torch.tensor([SAME, WORSE, BETTER, SAME])}
    output = {"delta": torch.randn(4), "logits": torch.randn(4, 3)}
    parts = DeltaLoss(torch.ones(3))(output, batch)
    assert parts["loss"].item() == pytest.approx(
        parts["regression"].item() + CLASS_WEIGHT * parts["classification"].item())
    assert parts["n_regression"].item() == 4.0
    assert parts["n_class"].item() == 4.0


def test_a_batch_with_nothing_usable_is_finite_and_zero():
    censor_a, censor_b = labels([("upper", "upper")] * 2)
    batch = {"delta": torch.tensor([1.0, -1.0]), "censor_a": censor_a,
             "censor_b": censor_b, "label_class": torch.tensor([SAME, SAME])}
    output = {"delta": torch.tensor([3.0, -3.0]), "logits": torch.zeros(2, 3)}
    parts = DeltaLoss(torch.ones(3))(output, batch)
    assert parts["loss"].item() == 0.0
    assert parts["n_regression"].item() == 0.0
    assert parts["n_class"].item() == 0.0


def test_the_loss_backpropagates_to_both_heads():
    censor_a, censor_b = labels([("exact", "exact"), ("exact", "upper")])
    batch = {"delta": torch.tensor([0.5, -2.0]), "censor_a": censor_a,
             "censor_b": censor_b, "label_class": torch.tensor([SAME, WORSE])}
    delta = torch.tensor([0.0, 0.0], requires_grad=True)
    logits = torch.zeros(2, 3, requires_grad=True)
    DeltaLoss(torch.ones(3))({"delta": delta, "logits": logits}, batch)["loss"].backward()
    assert delta.grad.abs().sum().item() > 0.0
    assert logits.grad.abs().sum().item() > 0.0


def test_run_tag_keeps_variant_runs_apart_from_the_standard_run():
    """A class weight or model seed other than the standard one must change the
    file names, or a variant run would overwrite the 0.3 / seed-42 results."""
    from app.losses import CLASS_WEIGHT, run_tag
    assert run_tag(CLASS_WEIGHT, 42, 42) == ""
    assert run_tag(0.1, 42, 42) == "_cw0.1"
    assert run_tag(0.0, 42, 42) == "_cw0"
    assert run_tag(CLASS_WEIGHT, 42, 3) == "_ms3"
    assert run_tag(0.0, 42, 3) == "_cw0_ms3"
    tags = {run_tag(w, 42, s) for w in (0.3, 0.1, 0.0) for s in (42, 1, 2)}
    assert len(tags) == 9
