# Limits

What this model cannot do, in the order that is likely to bite you.

## 1. The typical error is 8.6-fold in potency

RMSE **0.9339** log units on 16,116 held-out comparisons. A log unit is a tenfold change
in potency, so 0.93 log units is a factor of about 8.6 either way.

Read that as: a predicted Δp of −0.4 and a predicted Δp of +0.4 are not meaningfully
different claims. **A prediction smaller in magnitude than about 0.93 is inside the
model's own noise.** The interface says so next to every number.

What the model is reasonable for is *ranking*: given six candidate bioisosteres for one
compound on one target, the ordering carries some signal. The three-class MCC of **+0.369**
says the worse / same / better call is substantially better than chance (0.0) and a long
way from reliable (1.0).

## 2. It gets the direction of a change right far better than the size

The model systematically **under-calls how big a change is**. Fit a straight line
through predicted against measured and its slope is **0.34**, where a model that got
sizes right would give 1.00.

In concrete terms, from the 16,116 held-out comparisons:

| what the lab measured | what the model predicted, on average |
|---|---|
| a gain above +2 log units (>100× more potent), 807 cases | **+1.18** |
| a loss below −2 log units (>100× less potent), 553 cases | **−0.81** |

So a swap that really gained 100-fold is typically called as a 15-fold gain. The sign
is usually right; the magnitude is squashed toward zero by roughly a factor of three.

This is ordinary behaviour for a model predicting under uncertainty — guessing near the
average is how you minimise squared error when you are unsure — and it is partly a real
property of the data, where 68% of measured changes are under one log unit. But it has a
practical consequence worth stating plainly: **if you are screening for a dramatic
improvement, this model will under-sell the winners.** Use it to rank candidates, not to
estimate how much you will gain.

The effect is visible in `results/predicted_vs_measured.png` as the gap between the
orange line (what the model does) and the diagonal (what a perfect model would do).

## 3. There is a floor it cannot go below, and it is not far away

Two different papers measuring the same molecule pair on the same target disagree by
about **0.55 log units** on the difference — 3.5-fold — measured on this project's own
data after removing duplicates and near-copies (`results/noise_floor.csv`, row
`all pairs` / `group` / `cleaned`, column `iqr_sigma`).

That is the ceiling on any model trained on this data. The gap between 0.93 and 0.55 is
real room for improvement, but the headline cannot approach zero no matter what, and a
model reporting much below 0.55 on this data should be assumed to have a leak rather than
a breakthrough.

## 4. No skill on proteins it never saw

Much of the signal is *which protein* the swap is made on: an acid to hydroxamic acid swap
is worth a great deal on a zinc metalloenzyme and little elsewhere. The model learns one
embedding per target from the training data. A target that was never in training falls
back to a reserved "unknown" slot that carries no information about that protein.

This is why the interface offers a **fixed dropdown of trained targets and no free-text
entry**. On a protein outside that list, the honest prediction is the transformation
median, not this model, and the interface will not pretend otherwise. Even inside the
list, a target with 4 training pairs is far weaker than one with 900, which is why the
dropdown shows each target's training count.

## 5. It does not check that your two molecules are otherwise identical

The model was trained on matched pairs: two molecules identical apart from one headgroup.
The interface checks that **exactly one headgroup differs** — one removed from A, one
added to B, counted by copies so a diacid losing one acid still counts — and refuses to
predict otherwise.

It does **not** verify that the rest of the molecule is unchanged. If you give it two
molecules that differ at the headgroup *and* somewhere else, it will answer, and the
answer will be meaningless.

This is a real gap and not a cheap one to close. Two obvious approaches were tried against
this project's own data and both failed:

- **Counting connected regions outside a maximum common substructure.** On 250 genuine
  training pairs it reported more than one differing region 39% of the time, so it would
  refuse real pairs constantly — while reporting aspirin versus caffeine, two unrelated
  molecules, as a clean single-site change. It also took 1.3 seconds a pair.
- **Morgan fingerprint similarity.** Worse on small molecules, where a headgroup is a
  large fraction of the structure. A genuine 4-chlorobenzoic acid → tetrazole swap scores
  Tanimoto 0.250, below the 0.5th percentile of real training pairs, and a swap carrying
  an unrelated Cl → F change scores 0.081 — below aspirin versus caffeine.

Deciding that two molecules differ *only* at the headgroup needs the fragmentation the
training pipeline used (mmpdb), not a similarity score. Until that is wired into the
interface, the honest position is to say it has not been checked. The one check that did
survive calibration is a size guard: no training pair differed by more than **45.8%** of
its heavy atoms, so the interface warns past that.

## 6. The ensemble alone scores better than what is reported

The five-seed ensemble without the blend scores 0.9243 (8.4-fold) against the reported
0.9339 (8.6-fold). The better number is not the headline because the blend weight was
fixed in advance and may not be re-chosen after seeing test scores; a sweep done properly
on validation folds picked 0.5 rather than 1.0. The difference is 0.19 on a potency ratio
and changes nothing anyone would decide. See `docs/design.md`.

## 7. The five seeds can disagree more than the error bar

The headline is an average over five training runs that differ only in initialisation.
On some pairs those five agree closely; on others they span more than 1.5 log units. The
interface shows the spread for every prediction.

**When the spread is wider than the 0.93 error bar, the mean is not a prediction** — it is
five models failing to agree, averaged. That usually means the pair is unlike anything in
training. The published 8.6-fold figure is an average over held-out data and does not
cover this case.

## 8. Three-class calls come from the number, not from a classifier

The network has a classification head, but the published models were trained with
`--class-weight 0`, so it never learned anything and its outputs are ignored. The
worse / same / better call is the predicted Δp put through ±1 log unit thresholds — the
same thresholds the reported MCC is computed on. The number and the call therefore can
never contradict each other, and the call inherits every limit above.

## 9. What the data is and is not

ChEMBL 37, IC50 and similar endpoints, pooled across papers. Each molecule's potency is
the median of its measurements on that target and assay type, and Δp is only taken within
one assay type. Values that were only bounds ("> 10 µM") are handled by a censoring-aware
loss rather than dropped, but placeholder values that were really missing data were
censored out of the modelling set.

The data is heavily weighted toward carboxylic acid chemistry on well-studied targets,
because that is what the literature contains. A swap class with three examples in training
is being predicted from three examples.
