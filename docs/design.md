# Design notes

Why the model is shaped the way it is, and which of those choices cost something.

## The problem is a delta, not a QSAR

A training example is a triple: molecule A, molecule B (identical except the headgroup),
and target T, labelled Δp = pIC50(B) − pIC50(A). Predicting absolute potency and
subtracting would be far harder and would throw away the thing that makes this tractable:
within a matched pair, most of the systematic error — the assay, the lab, the cell line —
cancels.

## Antisymmetry is built in, not learned

Δp is antisymmetric: if A→B is +0.8 then B→A is −0.8. Rather than hope the network learns
this, the trunk runs on (A, B) and on (B, A) and the heads subtract the two passes. That
makes f(A, B) = −f(B, A) and f(A, A) = 0 **for any weights**, including random ones.
Swapping A and B exchanges "worse" and "better" exactly.

`tests/test_model.py` checks this to zero error on the CPU. Apple's MPS backend breaks it
at about 1e-8, so the default device is CUDA if available and CPU otherwise.

## Splitting on molecules, never on pairs

One molecule appears in many pairs. Splitting pairs at random puts the same molecule in
training and test and the model scores far better than it deserves. So molecules are
assigned to folds first — by the scaffold that did not change — and pairs are formed only
within a fold. Every mirror of a test pair stays with it.

This is the single most important thing in the project, and it is checked rather than
trusted: `tests/test_disjoint_series.py` and `tests/test_headgroup_split.py` assert that
no molecule and no mirrored pair straddles a fold.

## Three baselines, always

Every result is reported against predicting no change, the training median Δp for that
transformation, and that median conditioned on the local environment. Most headgroup
swaps do very little, so "predict zero" is genuinely hard to beat on RMSE and a model
that cannot beat it has learned nothing.

The stronger reference, `shrunk_target_mean`, is the average Δp for that swap on that
protein, shrunk toward the swap's overall average by an empirical-Bayes factor. It is the
baseline that encodes the project's main finding: **a lot of what matters is which protein
the swap is made on.** It scores 1.0396, and beating it is the real test.

## Training on all pair classes, testing on bioisosteres only

The test set is always bioisostere pairs. But the training data can include headgroup
changes that are not bioisosteric, deletions, and same-headgroup pairs — about 1.15M rows
rather than 46k. Doing that improved held-out RMSE from 1.0038 to 0.9243 on the same
comparisons, better on all five folds. The model learns more about what headgroups do
from seeing changes that are not the ones being tested.

Three epochs, not more. Six epochs fit the training data substantially better and did
**not** transfer: the 3-epoch choice was made by a rule fixed before the sweep ran, and
it is the weakest-supported choice in the project — the folds disagreed, and 3 epochs was
partly a compute decision.

## Why the headline is a blend, when the ensemble alone scores better

The reported prediction is `0.5 × ensemble + 0.5 × shrunk_target_mean`, which scores
0.9339. The ensemble alone scores 0.9243 — better.

The ensemble alone is not the headline because choosing it *after seeing the test scores*
is selecting a setting on test data, which is the one thing this project does not do. The
0.5 weight was registered in advance. To check it honestly, a sweep over
w ∈ {0, 0.25, 0.5, 0.75, 1.0} was run on validation folds carved out of the training
folds, with the test folds never read. It picked **w = 0.5**: mean validation RMSE 1.0765,
1.0193, **0.9925**, 0.9986, 1.0370. Four of five folds preferred 0.5 and none preferred
1.0.

That sweep used one seed while the published model uses five, which biases the chosen
weight downward — a stronger model wants more weight on itself. So the result does not
prove 0.5 is optimal for the five-seed ensemble; it proves 0.5 is defensible and was not
chosen on test data. The honest summary is that the better-scoring configuration exists,
is reported beside the headline, and was not selectable under the project's own rule. The
two differ by 0.19 on a potency ratio.

`results/blend_weight_validation_constant.csv` holds that sweep, so the argmin can be
rechecked.

## Why an ensemble of five seeds

One training run of this model is noisy: individual seeds on the same fold range from
0.9429 to 0.9887 RMSE. Averaging five seeds gives 0.9243, better than every individual
seed on every fold. The seeds differ only in initialisation, batch order and dropout; the
folds are identical.

The side benefit is the one the interface uses: the spread across seeds is a free,
per-prediction measure of how much the model disagrees with itself, which the single
headline RMSE cannot give you.

## The classification head is dead weight

The architecture has a three-class head, but the published models were trained with
`--class-weight 0`, so it contributed nothing to the loss and learned nothing. Its logits
are ignored everywhere, and the worse / same / better call is the predicted Δp thresholded
at ±1 log unit — the same definition the reported MCC uses. Keeping the head in the
architecture while never training it is untidy; it is kept because removing it would
invalidate the published checkpoints.

## Censoring is handled, not dropped

Many ChEMBL values are bounds: "IC50 > 10 µM". Dropping those rows would bias the data
toward compounds that worked. Instead the loss is Huber on exactly measured pairs, a
hinge on pairs where the bound actually constrains Δp, and zero weight on pairs whose
bounds say nothing about the difference. The headline is still reported on exactly
measured pairs only, one row per comparison.

## Evaluation discipline

- No setting is ever chosen on a test fold. Anything needing selection is chosen on a
  validation fold inside the training folds. Leave-one-fold-out over other folds' test
  scores does not count, and an earlier blend weight chosen that way was replaced.
- The five test folds are scored once, at the end.
- Significance is a paired 2σ bootstrap clustered by comparison, because rows sharing a
  comparison id are the same two molecules on different targets and are not independent.
- Any held-out RMSE below the 0.55 noise floor is treated as a suspected leak until shown
  otherwise.

## What would most improve this

1. **Matched-pair verification in the interface.** It checks the headgroup and not the
   rest of the molecule; see `docs/limits.md` for the two approaches that failed
   calibration and why mmpdb fragmentation is the real answer.
2. **A five-seed blend-weight sweep.** About 55 hours of compute to settle whether the
   optimal weight for the five-seed ensemble is above 0.5. It was declined deliberately:
   the answer changes 8.6-fold to at best 8.4-fold.
3. **More non-acid chemistry.** The data is dominated by carboxylic acid replacements
   because the literature is.
