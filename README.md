# Headgroup swap potency model

### ▶ [Try it: headgroup-swap.streamlit.app](https://headgroup-swap.streamlit.app)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://headgroup-swap.streamlit.app)
[![Dataset on Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.23456998.svg)](https://doi.org/10.5281/zenodo.23456998)
[![Licence](https://img.shields.io/badge/licence-MIT%20code%20%7C%20CC--BY--SA--3.0%20data-blue)](LICENSE)

Paste two SMILES, pick a protein target, get a prediction. Nothing to install.

---

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

| | RMSE (log units) | typical error | three-class MCC |
|---|---|---|---|
| **this model** (5-seed ensemble blended 50/50 with the baseline) | **0.9339** | **8.6×** | **+0.369** |
| the ensemble alone — better, but not selectable (see below) | 0.9243 | 8.4× | +0.373 |
| best baseline (`shrunk_target_mean`) | 1.0396 | 11.0× | +0.328 |
| transformation median | 1.0817 | 12.1× | +0.010 |
| transformation × environment median | 1.1419 | 13.9× | +0.181 |
| predict no change at all | 1.1213 | 13.2× | 0.000 |
| *between-lab noise floor* | *0.55* | *3.5×* | — |

Each row is the mean of five molecule-disjoint test folds, on the same 16,116
comparisons. The model beats both `zero` and `shrunk_target_mean` by more than
2 standard errors on all five folds, paired and clustered by comparison. Rows are not ordered by performance: `transformation × environment median` has a
higher RMSE than predicting no change, but a higher MCC.

![Predicted against measured Δp for 16,116 held-out comparisons, as a log-scaled
hexagonal density plot with the perfect-prediction diagonal and a ±RMSE band](results/predicted_vs_measured.png)

**Figure 1.** Predicted against measured Δp for all 16,116 held-out comparisons,
pooled over the five molecule-disjoint test folds. Hexagons are shaded by the
number of comparisons they contain (log scale). The grey band spans ±1 RMSE
(0.93 log units) about the identity line and contains 73% of comparisons. The
least-squares fit has slope 0.34 rather than 1.00: the model recovers the
direction of a swap far more reliably than its magnitude, compressing large
changes toward zero (`docs/limits.md`, limit 2). The upper axis gives the
corresponding fold-change in potency.

![Held-out RMSE for every fold, comparing the model with all three required
baselines and the shrunk_target_mean reference](results/rmse_by_fold.png)

**Figure 2.** Held-out RMSE for each test fold and for the mean over folds, which
is the reported figure. Every bar within a group is scored on the same
comparisons, so bars are directly comparable. All three required baselines are
shown alongside the stronger `shrunk_target_mean` reference. The blend improves
on every baseline in every fold; no configuration approaches the experimental
noise floor, which is set by disagreement between laboratories rather than by
the model.

Both figures are regenerated from the published per-row predictions by
`python app/make_figures.py`, as 400 dpi PNG and as vector PDF, so they cannot
drift from the tables.

## Run the interface

```
conda env create -f config/environment.yml
conda activate headgroup-swap
streamlit run app/streamlit_app.py
```

Then open the local URL it prints. Startup takes about ten seconds while the five
checkpoints load and the baseline is fitted; predictions after that are immediate.

There are two interfaces over the same backend. `app/streamlit_app.py` is the one that
deploys to [Streamlit Community Cloud](https://streamlit.io/cloud) and the one
`requirements.txt` installs. `app/app.py` is a Gradio version of the same thing for
local use — `pip install gradio==6.30.0`, then `python app/app.py`. Both are thin
presentation layers over `app/predict.py`, so neither can drift from the other or from
the published numbers.

The hosted copy runs on Streamlit Community Cloud, which installs `requirements.txt`
from the repository root and runs `app/streamlit_app.py`. Nothing in the repository
needs changing to deploy it.

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

- `data/pairs_bioisostere.csv.gz` — 57,184 rows, i.e. 20,210 distinct comparisons each
  stored in both directions, from ChEMBL 37. This is the slice the interface and the
  baseline use, and it is in the repository (1.6 MB gzipped; pandas reads it without
  unpacking).
- `data/target_properties.csv` — ChEMBL protein family for every target.
- **The full `pairs.csv`** — 1,291,070 rows covering every pair class, which is what the
  published model was trained on — is too large for git and lives on Zenodo:
  **[10.5281/zenodo.23456998](https://doi.org/10.5281/zenodo.23456998)** (42.7 MB
  gzipped). Download it into `data/` and `app/train.py` finds it; `docs/data.md`
  has the details.

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
app/        the model, both interfaces, training and scoring
  streamlit_app.py  the interface (this is what gets deployed)
  app.py            the same interface in Gradio, for local use
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
