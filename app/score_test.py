"""Score a trained SiameseDelta checkpoint on its TEST fold, once.

Reads:  models/<split>_fold<k>[_4fold].pt            from app/train.py
        data/pairs_bioisostere.csv
        data/target_properties.csv
        results/baselines_disjoint_molecules.csv     the baselines' test rows and scores
Writes: results/model_<split>_fold<k>_<n>fold[<tag>].csv   baselines' columns plus
                                                     mcc_head_three_class, one row per
                                                     censoring subset; <tag> as in
                                                     12_train.py (_cw<w>, _ms<n>)

Two kinds of checkpoint are kept apart by name, so nothing ever averages
them: siamese_delta_3fold (trained on three folds, a fourth held out for
early stopping) and siamese_delta_4fold (--no-val: the baselines' four folds).

THE SAME ROWS AS THE BASELINES
------------------------------
Rule 2 only means something if the model and the baselines are scored on the
same rows. So this script rebuilds the test rows exactly as the baseline
scorer does: the molecule-disjoint split with the same seed,
the test fold, one row per comparison, and then the `exact`
subset (the headline) and `all` beside it. It checks the row count against
the baselines file and stops if they differ.

The model's vocabularies come from the checkpoint - they were built on the
training rows - so a protein that never trained gets index 0 here exactly as
it did in validation.

WHAT "beats_constant" MEANS HERE
--------------------------------
The same paired test as the baselines (app/metrics.beats_paired), clustered
by comparison, against the better of `zero` and
`training_mean` on these rows. training_mean is computed from the model's own
training rows, which are three folds, not the baselines' four.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.baselines import DELTA, comparison_ids  # noqa: E402
from app.losses import CLASS_WEIGHT, run_tag  # noqa: E402
from app.metrics import beats_paired, evaluate, mcc_three_class, rmse  # noqa: E402
from app.model import SiameseDelta  # noqa: E402
from app.pair_dataset import (  # noqa: E402
    PairDataset, _split_result, collate_pairs, load_bioisostere_pairs,
    pair_class_tag, parse_pair_classes,
)
from app.splits import one_row_per_comparison  # noqa: E402

CENSORING_SUBSETS = ("exact", "all")


def exact_only(pairs):
    """Rows where neither side's value is a censored bound."""
    return (pairs["censored_a"] == "exact") & (pairs["censored_b"] == "exact")


def predict(model, dataset):
    """(predicted Delta p, head class 0/1/2) for every item, in dataset order, on CPU."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_pairs)
    model.eval()
    deltas, classes = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            deltas.append(output["delta"].numpy())
            classes.append(output["logits"].argmax(dim=1).numpy())
    return np.concatenate(deltas), np.concatenate(classes)


def head_mcc(truth, head_class):
    """Three-class MCC of the classification HEAD's call, binned like a delta
    (worse -> -2, same -> 0, better -> +2) so the same metric code applies. At
    --class-weight 0 the head was never trained and this number is meaningless."""
    as_delta = np.where(head_class == 0, -2.0, np.where(head_class == 2, 2.0, 0.0))
    return mcc_three_class(np.asarray(truth, dtype=float), as_delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp", default=str(REPO_ROOT / "data/pairs_bioisostere.csv"))
    parser.add_argument("--target-properties",
                        default=str(REPO_ROOT / "data/target_properties.csv"))
    parser.add_argument("--baselines",
                        default=str(REPO_ROOT / "results/baselines_disjoint_molecules.csv"))
    parser.add_argument("--split", default="constant")
    parser.add_argument("--fold", type=int, default=0, help="the TEST fold to score")
    parser.add_argument("--no-val", action="store_true",
                        help="score the four-fold checkpoint (12_train.py --no-val)")
    parser.add_argument("--class-weight", type=float, default=CLASS_WEIGHT,
                        help="score the checkpoint trained at this --class-weight")
    parser.add_argument("--model-seed", type=int, default=None,
                        help="score the checkpoint trained with this --model-seed")
    parser.add_argument("--pair-classes", default="bioisostere",
                        help="score the checkpoint trained on these pair classes "
                             "(12_train.py --pair-classes); the test rows are bioisostere "
                             "either way")
    parser.add_argument("--epochs", type=int, default=60,
                        help="score the checkpoint trained for this many epochs "
                             "(12_train.py names a non-60 run _e<n>)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pair_classes = parse_pair_classes(args.pair_classes)

    torch.manual_seed(args.seed)
    model_seed = args.seed if args.model_seed is None else args.model_seed
    tag = (run_tag(args.class_weight, args.seed, model_seed) + pair_class_tag(pair_classes)
           + ("" if args.epochs == 60 else f"_e{args.epochs}"))
    suffix = ("_4fold" if args.no_val else "") + tag
    checkpoint_path = REPO_ROOT / f"models/{args.split}_fold{args.fold}{suffix}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["split"] != args.split or checkpoint["fold"] != args.fold \
            or checkpoint["seed"] != args.seed:
        raise SystemExit(f"{checkpoint_path} is split {checkpoint['split']} fold "
                         f"{checkpoint['fold']} seed {checkpoint['seed']}, not what was asked")
    print(f"checkpoint {checkpoint_path.relative_to(REPO_ROOT)}: epoch {checkpoint['epoch']}, "
          f"validation RMSE {checkpoint['val_rmse']:.4f}, fusion {checkpoint['fusion']}")

    # Checkpoints from before --no-val existed carry no train_folds: they are 3.
    train_folds = checkpoint.get("train_folds", 3)
    if train_folds != (4 if args.no_val else 3):
        raise SystemExit(f"{checkpoint_path} was trained on {train_folds} folds")
    # Checkpoints from before --class-weight / --model-seed carry neither.
    if checkpoint.get("class_weight", CLASS_WEIGHT) != args.class_weight \
            or checkpoint.get("model_seed", checkpoint["seed"]) != model_seed:
        raise SystemExit(f"{checkpoint_path} does not record class weight "
                         f"{args.class_weight} and model seed {model_seed}")
    model_name = f"siamese_delta_{train_folds}fold{tag}"
    vocabs = checkpoint["vocabs"]
    if checkpoint.get("pair_classes", ["bioisostere"]) != pair_classes:
        raise SystemExit(f"{checkpoint_path} was trained on pair classes "
                         f"{checkpoint.get('pair_classes')}, not {pair_classes}")
    model = SiameseDelta(len(vocabs["swap"]) + 1, len(vocabs["family"]) + 1,
                         len(vocabs["target"]) + 1, fusion=checkpoint["fusion"],
                         n_pair_classes=checkpoint.get("n_pair_classes", 0))
    model.load_state_dict(checkpoint["model"])

    pairs = load_bioisostere_pairs(args.inp, args.target_properties)
    usable = pd.to_numeric(pairs[DELTA], errors="coerce").notna()
    if not usable.all():
        raise SystemExit(f"{int((~usable).sum())} rows have no {DELTA}; the baselines drop "
                         "them, so the row sets would differ")
    result = _split_result(pairs, args.split, args.n_folds, args.seed)
    kept = result.kept()
    test_mask = kept & (result.folds == args.fold)
    train_mask = kept & (result.folds != args.fold)
    # training_mean from the rows the model trained on: every non-test fold, or
    # for a 3-fold model not the validation fold either (12_train.py's default,
    # the fold after the test one).
    model_train_mask = train_mask
    if train_folds == 3:
        folds = sorted(set(result.folds[kept]))
        val_fold = folds[(folds.index(args.fold) + 1) % len(folds)]
        model_train_mask = train_mask & (result.folds != val_fold)
    training_mean = float(pairs.loc[model_train_mask, DELTA].mean())
    print(f"  model training rows ({train_folds} folds): "
          f"{int(model_train_mask.sum()):,}, mean {DELTA} {training_mean:+.4f}")

    baselines = pd.read_csv(args.baselines)
    rows = []
    scored = test_mask & one_row_per_comparison(pairs)
    for subset in CENSORING_SUBSETS:
        mask = scored & exact_only(pairs) if subset == "exact" else scored
        test = pairs[mask].reset_index(drop=True)
        expected = baselines[(baselines["split"] == args.split)
                             & (baselines["fold"] == args.fold)
                             & (baselines["censoring"] == subset)
                             & (baselines["model"] == "zero")]["n"]
        print(f"  {subset}: {int(mask.sum()):,} test rows (one per comparison) "
              f"of {int(test_mask.sum()):,} in fold {args.fold}; baselines scored "
              f"{int(expected.iloc[0]):,}")
        if len(expected) != 1 or int(expected.iloc[0]) != len(test):
            raise SystemExit("row count differs from the baselines file - not the same rows")

        dataset = PairDataset(test, vocabs, antisymmetry=False, identity_fraction=0.0,
                              seed=args.seed, name=f"test[{subset}]")
        if len(dataset) != len(test):
            raise SystemExit("PairDataset changed the row count")
        item_deltas = np.array([item["delta"] for item in dataset.items])
        if not np.allclose(item_deltas, test[DELTA].to_numpy()):
            raise SystemExit("dataset rows are not in the same order as the test rows")

        truth = test[DELTA]
        deltas, head_classes = predict(model, dataset)
        predicted = pd.Series(deltas, index=test.index)
        constants = {"zero": pd.Series(0.0, index=test.index),
                     "training_mean": pd.Series(training_mean, index=test.index)}
        reference_name = min(constants, key=lambda name: rmse(truth, constants[name]))
        reference = constants[reference_name]
        verdict = beats_paired(truth, predicted, reference, groups=comparison_ids(test))
        scores = evaluate(truth, predicted)
        rows.append({"split": args.split, "fold": args.fold, "held_out": "",
                     "series": "loose", "molecules": "disjoint", "censoring": subset,
                     "model": model_name, "n_train": int(model_train_mask.sum()),
                     "collapsed_to_global": False,
                     "beats_constant": verdict["beats_reference"],
                     "sigmas_vs_constant": round(verdict["sigmas"], 2),
                     "constant_rmse": rmse(truth, reference),
                     "n_comparisons": verdict["n_clusters"], **scores,
                     "mcc_head_three_class": head_mcc(truth, head_classes)})

    table = pd.DataFrame(rows)
    # The baselines' columns in their order, then the head's MCC, which only a
    # model with a classification head has.
    table = table[[c for c in baselines.columns if c in table.columns]
                  + ["mcc_head_three_class"]]
    out_path = REPO_ROOT / f"results/model_{args.split}_fold{args.fold}_{train_folds}fold{tag}.csv"
    table.to_csv(out_path, index=False)

    # The same-fold bars, so the number is read against its own rows.
    print(f"\nTest fold {args.fold}, {args.split} split, exact rows:")
    same = baselines[(baselines["split"] == args.split) & (baselines["fold"] == args.fold)
                     & (baselines["censoring"] == "exact")]
    exact_row = table[table["censoring"] == "exact"].iloc[0]
    print(f"  {'model':<36} {'rmse':>7} {'mcc':>7}")
    print(f"  {model_name:<36} {exact_row['rmse']:>7.4f} {exact_row['mcc_three_class']:>+7.3f}"
          f"   beats constant: {exact_row['beats_constant']} "
          f"({exact_row['sigmas_vs_constant']:+.2f} sigma)")
    print(f"  {'  MCC from the classification head':<36} {'':>7} "
          f"{exact_row['mcc_head_three_class']:>+7.3f}"
          + ("   (no head trained at class weight 0)" if args.class_weight == 0 else ""))
    for name in ("zero", "shrunk_target_mean", "headgroup_property_ridge"):
        row = same[same["model"] == name]
        if len(row):
            print(f"  {name:<36} {row['rmse'].iloc[0]:>7.4f} "
                  f"{row['mcc_three_class'].iloc[0]:>+7.3f}")
    print(f"\nSummary: {len(table)} rows written to {out_path.relative_to(REPO_ROOT)} "
          f"(exact n={int(exact_row['n']):,}).")


if __name__ == "__main__":
    main()
