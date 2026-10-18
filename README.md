---
title: Headgroup Swap Potency Model
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app/app.py
pinned: false
license: mit
---

# Headgroup swap potency model

Predicts how much a drug's potency changes when its acidic headgroup is swapped for a
bioisostere — a carboxylic acid replaced by a tetrazole, say — on a given protein target.

It is a matched-molecular-pair **delta** model: it never predicts absolute potency, only
the change caused by the swap. One input is a triple — molecule A, molecule B (identical
to A except for the headgroup), and a protein target T — and the output is

- **Δp = pIC50(B) − pIC50(A)** in log units, where 1.0 is a tenfold change, and
- a three-way call: **worse** (Δp < −1), **about the same**, or **better** (Δp > +1).

## The number you should judge it by

**Typical error 8.6-fold in potency** — RMSE 0.9339 log units, three-class MCC +0.369,
measured on 16,116 held-out comparisons whose molecules were never seen in training.

That is good enough to rank a handful of candidate swaps and nowhere near good enough to
tell you a compound is worth making. Two different labs measuring *the same* pair disagree
by about 0.55 log units (3.5-fold), so no model built on this data can do much better.

| | RMSE (log units) | typical error | three-class MCC |
|---|---|---|---|
| **this model** (5-seed ensemble blended 50/50 with the baseline) | **0.9339** | **8.6×** | **+0.369** |
| the ensemble alone, without the blend | 0.9243 | 8.4× | +0.373 |
| best baseline (`shrunk_target_mean`) | 1.0396 | 11.0× | +0.328 |
| transformation median | 1.0817 | 12.1× | +0.010 |
| transformation × environment median | 1.1419 | 13.9× | +0.181 |
| predict no change at all | 1.1213 | 13.2× | 0.000 |
| *between-lab noise floor* | *0.55* | *3.5×* | — |

Each row is the mean of five molecule-disjoint test folds, on the same 16,116
comparisons. The model beats both `zero` and `shrunk_target_mean` by more than
2 standard errors on all five folds, paired and clustered by comparison.

![Predicted against measured Δp for 16,116 held-out comparisons, as a log-scaled
hexagonal density plot with the perfect-prediction diagonal and a ±RMSE band](results/predicted_vs_measured.png)

*Every held-out comparison, all five test folds pooled. Each hexagon is shaded by
how many comparisons fell in it, on a log scale. The shaded band is one typical
error either side of perfect and holds 73% of them. The orange line is what the
model actually does: its slope of 0.34, against the diagonal's 1.00, is the model
squashing big changes toward zero — it reads the direction of a swap far better
than the size. Limit 2 in `docs/limits.md` gives the numbers.*

![Held-out RMSE for every fold, comparing the model with all three required
baselines and the shrunk_target_mean reference](results/rmse_by_fold.png)

*The same comparisons score every bar within a group, so the bars are directly
comparable. The rightmost group is the headline. The model beats every baseline
on every fold, and no bar comes near the noise floor — the limit set by labs
disagreeing with each other, not by the model.*

Both figures are redrawn from the published per-row predictions by
`python app/make_figures.py`, so they cannot drift from the tables.

## Run the interface

```
conda env create -f config/environment.yml
conda activate headgroup-swap
python app/app.py
```

Then open the local URL it prints. First prediction takes a second or two while the five
checkpoints load; after that it is immediate.

The same repository is a Hugging Face Space with no changes — the front matter at the top
of this file points `app_file` at `app/app.py`, and `requirements.txt` is at the root
where Spaces expects it.

`docs/user-guide.md` explains what each part of the output means.

## How it works

- **Encoder:** one shared chemprop v2 D-MPNN reads both molecules.
- **Conditioning:** learned embeddings for the swap (which headgroup replaced which), the
  target's protein family and the target itself.
- **Exactly antisymmetric by construction:** the trunk runs on (A, B) and on (B, A) and
  the heads subtract, so f(A, B) = −f(B, A) and f(A, A) = 0 for any weights. Swapping A
  and B exchanges "worse" and "better" exactly. `tests/test_model.py` checks this to zero
  error on the CPU.
- **Censoring-aware loss:** Huber on exactly measured pairs; on pairs where a value is
  only a bound ("IC50 > 10 µM") a hinge that applies only when the bound actually
  constrains Δp; zero weight on pairs whose bounds say nothing.
- **The published prediction is a blend:** `0.5 × ensemble + 0.5 × shrunk_target_mean`.
  The weight was chosen on validation folds carved out of the training folds, never on a
  test fold.

`docs/design.md` has the rest, including why the ensemble alone is not the headline even
though it scores better.

## Evaluation

Five-fold **molecule-disjoint series split**: molecules are assigned to folds first, by
the scaffold that did not change, and pairs are formed only within a fold, so no test
molecule is ever seen in training. Pairs are never split at random, because one molecule
appears in many pairs — a random split over pairs leaks badly and flatters the model.

## Data

- `data/pairs_bioisostere.csv` — 57,184 rows, i.e. 20,210 distinct comparisons each
  stored in both directions, from ChEMBL 37. This is the slice the interface and the
  baseline use, and it is in the repository (23 MB).
- `data/target_properties.csv` — ChEMBL protein family for every target.
- **The full `pairs.csv`** — 1.29M rows covering every pair class, which is what the
  published model was trained on — is too large for git and is on Zenodo. See
  `docs/data.md` for the DOI and how to use it.

## Retraining

```
python app/train.py --split constant --fold 0 --no-val --class-weight 0 \
    --pair-classes all --epochs 3 --fit-every 2000 --seed 42 --model-seed 1
```

for each fold 0–4 and each `--model-seed` in 42, 1, 2, 3, 4 — 25 runs, about 2.5 hours
each on a laptop CPU. `--pair-classes all` needs the full `pairs.csv` from Zenodo.
`docs/user-guide.md` covers the shorter bioisostere-only run, which works from this
repository alone.

## Layout

```
app/        the model, the Gradio interface, training and scoring
  app.py            the interface
  predict.py        loading, the ensemble, the blend, the input checks
  model.py          the network
  pair_dataset.py   pairs -> molecule graphs, folds, vocabularies
  train.py          train one fold
  score_test.py     score one fold on its held-out test set
  make_figures.py   redraws the two figures in results/
tests/      541 tests covering everything in app/ except the interface
docs/       user guide, design notes, limits, data
config/     environment.yml, .env.example
data/       the bioisostere slice; the full file is on Zenodo
results/    the result tables and the two figures above
models/     25 checkpoints: five folds x five seeds
```

## Licence

MIT for the code. The data is derived from ChEMBL 37 and carries CC BY-SA 3.0, which the
trained weights inherit. See `LICENSE` for what that means in practice and how to cite
ChEMBL.

## Limits worth reading before you use it

The full list is `docs/limits.md`. The four that matter most:

1. **8.6-fold typical error.** A predicted Δp smaller than about 0.93 log units is inside
   the noise and should not be read as a real change.
2. **Directions are far more trustworthy than sizes.** The model squashes big changes
   toward zero: where the lab measured a gain above +2 log units it predicted +1.18 on
   average. Rank candidates with it; do not use it to estimate how much you will gain.
3. **No skill on proteins it never saw.** Much of what the model knows is which protein
   the swap is made on. The interface therefore offers a fixed list of trained targets
   and no free-text entry.
4. **It does not check that your two molecules are otherwise identical.** It checks that
   exactly one headgroup differs and refuses otherwise, but if the molecules also differ
   elsewhere, the prediction is meaningless and nothing will tell you so.
