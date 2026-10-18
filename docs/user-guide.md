# User guide

## What the model does

You give it three things:

- **Molecule A** as SMILES — the parent, usually the one bearing the carboxylic acid.
- **Molecule B** as SMILES — identical to A except that the headgroup has been swapped.
- **A protein target**, chosen from a list.

It predicts **Δp = pIC50(B) − pIC50(A)**: how much potency changes when you make that
swap on that protein. Positive means B is more potent. One log unit is a tenfold change.

It does not predict how potent either molecule is. A delta model only ever answers "what
does this change do", which is the question a medicinal chemist choosing between
bioisosteres actually has.

The headgroups it knows are the seventeen in `app/headgroups.py`: carboxylic acid,
tetrazole, acylsulfonamide, hydroxamic acid, sulfonic acid, phosphonic acid, the
hydroxy-azoles, the acidic diones, squaramide and the rest.

## Running the interface

```
conda env create -f config/environment.yml
conda activate headgroup-swap
python app/app.py
```

It prints a local URL. The first prediction takes a second or two while five checkpoints
load and the baseline is fitted; everything after that is immediate.

To change which fold's five seeds are loaded, set `HEADGROUP_FOLD` (see
`config/.env.example`). Running two different folds on the same pair is a cheap sanity
check: they are genuinely different models.

On Hugging Face Spaces nothing needs changing. The YAML at the top of `README.md` points
`app_file` at `app/app.py`, and `requirements.txt` sits at the repository root where
Spaces looks for it.

## Reading the output

**The big number** is the prediction, in log units, after blending. Beside it is the
fold-change it corresponds to and the three-class call — worse, about the same, or better
— which is that same number put through ±1 log unit thresholds.

**The seed spread** is five models' worth of disagreement about your specific pair. The
headline 8.6-fold error is an average over held-out data; the spread is specific to what
you typed. If the spread is wider than 0.93 log units, treat the mean as "the model does
not know" rather than as an answer.

**The table** shows every number behind the answer: each seed, their mean, the baseline,
and the blend. The prediction is always

```
0.5 x (mean of the five seeds) + 0.5 x shrunk_target_mean
```

where `shrunk_target_mean` is the average measured Δp for that swap on that protein,
pulled toward the swap's overall average in proportion to how little data supports it.
A swap-target cell with one measurement is barely trusted; one with fifty is trusted
almost completely.

**Warnings** appear when the input is unusual — for instance when the two molecules
differ in size by more than any training pair did.

## When it refuses

It refuses rather than guessing when:

- either SMILES will not parse;
- **more than one headgroup differs** — two acids becoming two tetrazoles is two changes,
  not one, and nothing in training looked like that;
- no acidic headgroup differs at all, so there is no swap to predict;
- a headgroup was removed without another being added, or added without one being
  removed. That is a deletion or an addition, not a bioisostere swap, and the published
  error bar does not cover it.

The refusal always says which case it is.

## Why the target list is finite

The model learns one embedding per protein target. A target not in training gets a
reserved "unknown" slot carrying no information about that protein, so predictions there
are worth no more than the transformation median — and the interface would be lying by
presenting one. Free-text entry is therefore not offered. The dropdown shows each
target's training-pair count, because a target with four pairs behind it is much weaker
than one with nine hundred.

## Retraining

The full model needs `pairs.csv` from Zenodo (`docs/data.md`), because it trains on all
pair classes:

```
python app/train.py --split constant --fold 0 --no-val --class-weight 0 \
    --pair-classes all --epochs 3 --fit-every 2000 --seed 42 --model-seed 1
```

`--fold` is the TEST fold, never trained on. `--no-val` trains on the other four folds and
keeps the final epoch. `--model-seed` changes initialisation, batch order and dropout
while `--seed` keeps the folds fixed — that is what an ensemble varies. The published
ensemble is folds 0–4 × model seeds 42, 1, 2, 3, 4, which is 25 runs at roughly 2.5 hours
each on a laptop CPU.

A smaller run that works from this repository alone, with no Zenodo download, trains on
the bioisostere slice only:

```
python app/train.py --split constant --fold 0 --no-val --epochs 60 --seed 42
```

Scoring one fold on its held-out test set:

```
python app/score_test.py --split constant --fold 0 --no-val --seed 42
```

Non-default options change the output file names, so a variant can never overwrite the
standard results. Pass the same options to the scorer that you passed to the trainer.

## Redrawing the figures

```
python app/make_figures.py
```

Both figures in `results/` are regenerated from the published per-row predictions, so
they cannot drift away from the numbers in the tables.

## Running the tests

```
pytest
```

541 tests, covering the chemistry (SMARTS matching, headgroup counting, swap checking),
the splitting rules, the loss, the metrics and the model's antisymmetry. They do not
cover the Gradio interface.
