"""The training objective for SiameseDelta.

Three pieces: a robust regression loss, a one-sided loss for censored
measurements, and a class-weighted cross-entropy. The subtle part is the
censoring, so it is worked out here in full.

WHY HUBER AND NOT MSE
---------------------
9.3% of bioisostere comparisons move more than 2 log units. Squared error
makes one mislabelled activity cliff worth more gradient than a hundred
ordinary pairs, and the cliffs are exactly where the data is least
trustworthy. Huber (smooth L1, delta = 1.0) is quadratic for small errors and
linear beyond, so a 5-log outlier pulls like a 1-log one.

WHICH WAY A CENSORED MEASUREMENT BOUNDS Delta p
-----------------------------------------------
This is NOT "A or B is censored". Delta p = pIC50(B) - pIC50(A), and a bound
on the difference needs the two sides bounded in OPPOSITE directions.

`upper` means the recorded pchembl is an upper bound on the truth (a record
like "IC50 > 1000 nM": the compound is at best this potent). `lower` is the
other way. So, writing c for the recorded Delta p:

    Delta p <= c   needs   pIC50(B) <= recorded   AND   pIC50(A) >= recorded
                   i.e.    B is exact or upper    AND   A is exact or lower

    Delta p >= c   needs   B is exact or lower    AND   A is exact or upper

with at least one side actually censored (or it is simply exact). The two
conditions cannot both hold unless both sides are exact, so a row is bounded
in at most one direction - `assert_one_direction` checks that on real data
rather than trusting the argument.

What this costs, counted on the 57,184 bioisostere rows:

    both exact                       42,340   74.0%   Huber
    one-sided bound                   6,638   11.6%   squared hinge
    no usable bound                   8,206   14.4%   no regression loss

The last group is the surprise and it is worth stating plainly: 26% of rows
carry a bound somewhere, but MORE THAN HALF OF THOSE SAY NOTHING about the
difference. (upper, upper) is 2,730 rows and (lower, lower) 3,430 - if both
molecules are "IC50 > 1000 nM", either could be arbitrarily weaker and Delta p
is unconstrained in both directions. Every `mixed` combination (2,046 rows) is
unusable too, because the median of some exact and some bounded records is
neither a value nor a clean bound (app/join.py). Those rows stay in the
dataset and in the batch counts with weight zero, so the waste is visible in
the log rather than hidden by a filter (rule 5).

THE CLASSIFICATION HEAD ON CENSORED ROWS
----------------------------------------
A censored row's recorded class can be wrong, so the class loss is applied
only where the bound PROVES the class: bounded above by c < -1 is certainly
"worse", bounded below by c > +1 is certainly "better". Measured: 3,054 of the
6,638 one-sided rows (46.0%) qualify, on top of all 42,340 exact rows.

A NOTE ON SCALE, UNRESOLVED
---------------------------
The hinge is squared while Huber is linear in the tail, so a censored row
whose prediction sits 3 log units the wrong side of its bound contributes 9
while an exact row with the same error contributes about 2.5. Censored
violations are therefore punished harder than exact ones. That is what was
asked for and it is defensible - crossing a bound is a definite contradiction
where an exact error is only a miss - but it is an asymmetry, not a neutral
choice, and if censored rows end up dominating the gradient this is the first
thing to look at.
"""

import torch
from torch import nn

from app.join import CENSOR_LABELS
from app.model import BETTER, SAME, WORSE

# Indices into CENSOR_LABELS, which the loader stores as integers.
EXACT = CENSOR_LABELS.index("exact")
UPPER = CENSOR_LABELS.index("upper")
LOWER = CENSOR_LABELS.index("lower")
MIXED = CENSOR_LABELS.index("mixed")

HUBER_DELTA = 1.0
CLASS_WEIGHT = 0.3          # the classification head against the regression head's 1.0


def run_tag(class_weight, seed, model_seed):
    """The part of a run's file names that says how it differs from the standard
    run, so a variant can never overwrite it: "" for class weight 0.3 with the
    model seed equal to the fold seed, else e.g. "_cw0.1" or "_cw0.1_ms7"."""
    tag = "" if class_weight == CLASS_WEIGHT else f"_cw{class_weight:g}"
    if model_seed != seed:
        tag += f"_ms{model_seed}"
    return tag
CLASS_THRESHOLD = 1.0       # the same boundary as app/pair_dataset.three_class


def bound_directions(censor_a, censor_b):
    """(exact, above, below): three disjoint boolean masks over the batch.

    `above` means the true Delta p is at most the recorded one, `below` at
    least. Rows in none of the three have no usable bound.
    """
    both_exact = (censor_a == EXACT) & (censor_b == EXACT)
    above = ((censor_b == EXACT) | (censor_b == UPPER)) \
        & ((censor_a == EXACT) | (censor_a == LOWER)) & ~both_exact
    below = ((censor_b == EXACT) | (censor_b == LOWER)) \
        & ((censor_a == EXACT) | (censor_a == UPPER)) & ~both_exact
    return both_exact, above, below


def assert_one_direction(censor_a, censor_b):
    """No row may be bounded both ways. Cheap, and run on the real batches."""
    _exact, above, below = bound_directions(censor_a, censor_b)
    both = above & below
    if bool(both.any()):
        index = int(torch.nonzero(both)[0])
        raise AssertionError(
            f"{int(both.sum())} row(s) bounded above AND below, e.g. row {index} "
            f"with censoring ({CENSOR_LABELS[int(censor_a[index])]}, "
            f"{CENSOR_LABELS[int(censor_b[index])]})")
    return True


def regression_loss(prediction, target, censor_a, censor_b, huber_delta=HUBER_DELTA):
    """(loss per row, weight per row) for the regression head.

    Weight is 1 for a row that carries information and 0 for one that does not,
    so the caller can both average correctly and report how many rows were
    idle. Rows with no usable bound get loss 0 and weight 0.
    """
    exact, above, below = bound_directions(censor_a, censor_b)
    per_row = torch.zeros_like(prediction)

    if bool(exact.any()):
        per_row = torch.where(
            exact,
            nn.functional.huber_loss(prediction, target, reduction="none",
                                     delta=huber_delta),
            per_row)
    # Bounded above: only a prediction ABOVE the bound is a contradiction.
    per_row = torch.where(above, torch.clamp(prediction - target, min=0.0) ** 2,
                          per_row)
    per_row = torch.where(below, torch.clamp(target - prediction, min=0.0) ** 2,
                          per_row)
    weight = (exact | above | below).to(prediction.dtype)
    return per_row, weight


def class_mask(target, censor_a, censor_b, threshold=CLASS_THRESHOLD):
    """Rows whose three-class label is certainly right.

    Every exact row, plus the censored rows whose bound already decides the
    class: bounded above by c < -threshold must be `worse`, bounded below by
    c > +threshold must be `better`.
    """
    exact, above, below = bound_directions(censor_a, censor_b)
    proven = (above & (target < -threshold)) | (below & (target > threshold))
    return exact | proven


def inverse_frequency_weights(labels, n_classes=3):
    """Weights proportional to 1 / count, scaled to average 1 over the classes.

    Unweighted cross-entropy on a 74/13/12 split learns to answer "same" and
    stop. Averaging to 1 keeps the loss on the same scale as an unweighted one,
    so CLASS_WEIGHT still means what it says. A class absent from the training
    fold gets weight 1 rather than infinity.
    """
    counts = torch.bincount(labels, minlength=n_classes).to(torch.float32)
    weights = torch.where(counts > 0, 1.0 / torch.clamp(counts, min=1.0),
                          torch.ones_like(counts))
    present = counts > 0
    if bool(present.any()):
        weights = weights / weights[present].mean()
    return weights


class DeltaLoss(nn.Module):
    """Huber-or-hinge regression plus class-weighted cross-entropy.

    Returns the total and every part, because a combined number that moves
    tells you nothing about which head moved it.
    """

    def __init__(self, class_weights, class_weight=CLASS_WEIGHT,
                 huber_delta=HUBER_DELTA):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.class_weight = class_weight
        self.huber_delta = huber_delta

    def forward(self, output, batch):
        target = batch["delta"]
        censor_a, censor_b = batch["censor_a"], batch["censor_b"]
        assert_one_direction(censor_a, censor_b)

        per_row, weight = regression_loss(output["delta"], target, censor_a,
                                          censor_b, self.huber_delta)
        n_regression = weight.sum()
        # Mean over the rows that carry information, not over the batch: a
        # batch full of unusable rows must not look like a batch with no error.
        regression = (per_row * weight).sum() / torch.clamp(n_regression, min=1.0)

        mask = class_mask(target, censor_a, censor_b)
        n_class = mask.sum()
        if bool(n_class > 0):
            per_class = nn.functional.cross_entropy(
                output["logits"], batch["label_class"],
                weight=self.class_weights.to(output["logits"].dtype),
                reduction="none")
            classification = (per_class * mask).sum() / n_class
        else:
            classification = torch.zeros((), device=target.device,
                                         dtype=target.dtype)

        total = regression + self.class_weight * classification
        return {"loss": total, "regression": regression.detach(),
                "classification": classification.detach(),
                "n_regression": n_regression.detach(),
                "n_class": n_class.detach().to(target.dtype),
                "n_rows": torch.tensor(float(len(target)), device=target.device)}
