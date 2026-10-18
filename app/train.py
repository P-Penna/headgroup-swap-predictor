"""Train one fold of SiameseDelta and write its learning curve.

Reads:  data/pairs_bioisostere.csv
        data/target_properties.csv
Writes: results/train_<split>_fold<k>.csv    one row per epoch
        models/<split>_fold<k>.pt            the best checkpoint by val RMSE
        (with --no-val: the same names with a _4fold suffix, and the
        checkpoint is the FINAL epoch's; a --class-weight other than 0.3
        adds _cw<w>, and a --model-seed other than --seed adds _ms<n>, so a
        variant run never overwrites the standard one)

WHAT IS HELD OUT FROM WHAT
--------------------------
Three disjoint molecule sets, from the existing molecule-disjoint splitters: three folds train, one is the VALIDATION fold that early
stopping watches, and one is the test fold, which this script never touches.
Stopping on the test fold would be choosing the epoch count with the answer in
hand - the same mistake as fitting a shrinkage strength on the test fold - so
the test set is loaded only to prove it exists and is
disjoint, and is not scored here. Scoring it is score_test.py's job, once.

VALIDATION NUMBERS, AND WHY THESE ONES
--------------------------------------
Early stopping watches validation RMSE on EXACT rows only, one row per
comparison - the same headline convention as the baselines, so the number is
directly comparable with the baselines (rule 2). Censored rows cannot
contribute a meaningful squared error and are excluded from the metric even
though they do train the model through the hinge.

Two MCCs are reported, because the model has two ways to answer:
  mcc_regression   the three-class call implied by the predicted Delta p, via
                   app/metrics.py - comparable with every baseline
  mcc_class_head   the classification head's own argmax
If they disagree sharply the heads have learned different things, which is
worth knowing before trusting either.

THE SCHEDULE
------------
AdamW at 1e-3 with a one-cycle schedule over 60 epochs, batch 64 (decision
141: on constant fold 0 the improvement came during annealing and the curve
was flat by epoch 50). With a validation fold, early stopping on validation
RMSE with patience 30 still applies. `set_epoch` is called at the top of
every epoch so target dropout draws a fresh mask.

--no-val: TRAIN ON ALL FOUR NON-TEST FOLDS
------------------------------------------
Holding out a validation fold costs a quarter of the training data. Measured
on constant fold 0: shrunk_target_mean scores 1.023 on the test fold when fit
on four folds and 1.074 when fit on the model's three, so a 3-fold model is
handicapped against every baseline in the table. With the schedule fixed,
--no-val trains on the same four folds as the baselines, runs every epoch,
and keeps the FINAL epoch - nothing is chosen by looking at held-out data.
The validation columns of the curve are left empty.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.losses import (  # noqa: E402
    CLASS_WEIGHT, DeltaLoss, inverse_frequency_weights, run_tag,
)
from app.metrics import mcc_three_class, rmse  # noqa: E402
from app.model import SiameseDelta  # noqa: E402
from app.pair_dataset import (  # noqa: E402
    CLASS_NAMES, load_bioisostere_pairs, load_extra_pairs, collate_pairs,
    fit_sample, fold_datasets, pair_class_tag, parse_pair_classes, train_val_test_datasets,
)
from app.join import CENSOR_LABELS  # noqa: E402

EXACT = CENSOR_LABELS.index("exact")


def default_device():
    """CUDA if present, otherwise CPU - deliberately NOT Apple's MPS.

    Measured on this machine: MPS runs a training step in 126 ms against the
    CPU's 131-168 ms, a marginal gain, and it breaks the model's exact
    antisymmetry - max |f(A,B) + f(B,A)| comes out at 7.5e-9 on MPS against
    0.0 on CPU, from float rounding in the scatter/aggregate kernels. The
    error is numerically negligible but the guarantee is the point, so MPS is
    available through --device mps and is not the default.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def exact_mask(dataset):
    """Rows where neither side is a censored bound: the headline metric's rows."""
    return np.array([item["censor_a"] == "exact" and item["censor_b"] == "exact"
                     for item in dataset.items])


def evaluate(model, loader, device):
    """(predicted deltas, true deltas, predicted classes) over a whole loader."""
    model.eval()
    deltas, truth, classes = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(batch)
            deltas.append(output["delta"].cpu().numpy())
            truth.append(batch["delta"].cpu().numpy())
            classes.append(output["logits"].argmax(dim=1).cpu().numpy())
    return (np.concatenate(deltas), np.concatenate(truth), np.concatenate(classes))


def to_device(batch, device):
    """Move a batch onto the device.

    chemprop's BatchMolGraph.to() mutates IN PLACE and returns None, unlike
    every tensor method here, so it is called for its effect and not assigned.
    Assigning it silently replaces the graph with None and the first use fails
    somewhere far away.
    """
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    if device.type != "cpu":
        for key in ("graph_a", "graph_b"):
            moved[key].to(device)
    return moved


def class_counts(dataset):
    """{class name: count} over a dataset's items, by the recorded delta."""
    from app.pair_dataset import three_class
    counts = {name: 0 for name in CLASS_NAMES}
    for item in dataset.items:
        counts[CLASS_NAMES[three_class(item["delta"])]] += 1
    return counts


def keeps_checkpoint(has_val_fold, keep_final, val_rmse, best_rmse):
    """Whether this epoch's weights are written to the checkpoint file.

    No validation fold: every epoch overwrites it, so the file ends holding the
    final epoch. With a validation fold: only an epoch that improves validation
    RMSE, so the file ends holding the best one - unless --keep-final, which
    overwrites every epoch as well, so the validation fold is held out of
    training but never selects the model.
    """
    if not has_val_fold or keep_final:
        return True
    return val_rmse < best_rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp", default=str(REPO_ROOT / "data/pairs_bioisostere.csv"))
    parser.add_argument("--target-properties",
                        default=str(REPO_ROOT / "data/target_properties.csv"))
    parser.add_argument("--split", default="constant",
                        help="constant (default), target or scaffold")
    parser.add_argument("--fold", type=int, default=0, help="the TEST fold, never trained on")
    parser.add_argument("--val-fold", type=int, default=None,
                        help="default: the fold after --fold, wrapping round")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-val", action="store_true",
                        help="train on all four non-test folds, no early stopping, "
                             "keep the final epoch")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--keep-final", action="store_true",
                        help="keep the FINAL epoch even with a validation fold, instead of "
                             "the best validation RMSE, and never stop early; the validation "
                             "fold is then held out of training but not used to select the "
                             "model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fusion", default="difference_product",
                        help="difference_product (default) or full")
    parser.add_argument("--identity-fraction", type=float, default=0.05)
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop after this long regardless, for a first look")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None,
                        help="cpu, mps or cuda; default picks the fastest available")
    parser.add_argument("--class-weight", type=float, default=CLASS_WEIGHT,
                        help="weight of the classification head's loss against the "
                             f"regression loss's 1.0 (default {CLASS_WEIGHT}); 0 trains no head")
    parser.add_argument("--model-seed", type=int, default=None,
                        help="seed for weight initialisation, batch order and dropout; "
                             "defaults to --seed. --seed alone fixes the folds, so an "
                             "ensemble varies this and keeps --seed at 42")
    parser.add_argument("--pair-classes", default="bioisostere",
                        help="pair classes that TRAIN the model, comma-separated or 'all' "
                             "(bioisostere, same_headgroup, to_non_acid, deletion). The test "
                             "set is bioisostere only either way. Default: bioisostere only, "
                             "exactly as before. With a validation fold, the "
                             "extra rows are filtered against it too")
    parser.add_argument("--all-pairs", default=str(REPO_ROOT / "data/pairs.csv.gz"),
                        help="where the non-bioisostere classes are read from")
    parser.add_argument("--leakage-only", action="store_true",
                        help="build the split and print the extra-class leakage counts, "
                             "then stop without training")
    parser.add_argument("--fit-every", type=int, default=0,
                        help="every this many steps (and at each epoch end), RMSE on a fixed "
                             "sample of 5,000 bioisostere TRAINING comparisons, written to "
                             "results/fit_<split>_fold<k><tag>.csv; 0 (default) is off")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pair_classes = parse_pair_classes(args.pair_classes)

    model_seed = args.seed if args.model_seed is None else args.model_seed
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    device = torch.device(args.device) if args.device else default_device()
    print(f"device: {device}")

    pairs = load_bioisostere_pairs(args.inp, args.target_properties)
    columns = list(pd.read_csv(args.inp, nrows=0).columns)
    extra = load_extra_pairs(args.all_pairs, args.target_properties, pair_classes, columns)
    if args.leakage_only:
        from app.pair_dataset import _split_result, extra_training_rows
        result = _split_result(pairs, args.split, args.n_folds, args.seed)
        test_rows = pairs[result.kept() & (result.folds == args.fold)]
        print(f"  test fold {args.fold}: {len(test_rows):,} bioisostere rows")
        extra_training_rows(extra, test_rows)
        return
    if args.no_val:
        if args.val_fold is not None:
            raise SystemExit("--no-val and --val-fold contradict each other")
        train_set, test_set = fold_datasets(
            pairs, split=args.split, fold=args.fold, n_folds=args.n_folds,
            seed=args.seed, identity_fraction=args.identity_fraction, extra=extra)
        val_set = None
        print(f"  items: train {len(train_set):,} (four folds)  no validation  "
              f"test {len(test_set):,} (test is NOT scored here)")
    else:
        train_set, val_set, test_set = train_val_test_datasets(
            pairs, split=args.split, fold=args.fold, val_fold=args.val_fold,
            n_folds=args.n_folds, seed=args.seed,
            identity_fraction=args.identity_fraction, extra=extra)
        print(f"  items: train {len(train_set):,}  val {len(val_set):,}  "
              f"test {len(test_set):,} (test is NOT scored here)")
    # The validation fold actually used, resolving the default the dataset
    # helper applies (the fold after the test one), to record in the checkpoint.
    val_fold = None if args.no_val else (
        args.val_fold if args.val_fold is not None
        else (args.fold + 1) % args.n_folds)

    # Class weights from the TRAINING fold only, and printed, because an
    # unweighted head on a 74/13/12 split learns to answer "same" (rule 5).
    train_labels = torch.tensor([item["label_class"] for item in
                                 (train_set[i] for i in range(len(train_set)))])
    counts = class_counts(train_set)
    weights = inverse_frequency_weights(train_labels)
    total = sum(counts.values())
    print("  training class balance and inverse-frequency weights:")
    for index, name in enumerate(CLASS_NAMES):
        print(f"    {name:<7} {counts[name]:>7,} ({100 * counts[name] / total:4.1f}%)"
              f"  weight {weights[index]:.3f}")

    vocabs = train_set.vocabs
    n_pair_classes = len(vocabs["pair_class"]) + 1 if "pair_class" in vocabs else 0
    if pair_classes != ["bioisostere"]:
        print(f"  pair-class vocabulary: {vocabs['pair_class']}")
    model = SiameseDelta(len(vocabs["swap"]) + 1, len(vocabs["family"]) + 1,
                         len(vocabs["target"]) + 1, fusion=args.fusion,
                         n_pair_classes=n_pair_classes).to(device)
    print(f"  model: fusion={args.fusion}, "
          f"{model.parameter_counts()['total']:,} parameters")

    loss_fn = DeltaLoss(weights, class_weight=args.class_weight).to(device)
    print(f"  loss: regression 1.0 + classification {args.class_weight}; "
          f"model seed {model_seed}, fold seed {args.seed}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(model_seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pairs, generator=generator)
    val_loader = None if val_set is None else DataLoader(
        val_set, batch_size=256, shuffle=False, collate_fn=collate_pairs)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader))

    if val_set is not None:
        val_exact = exact_mask(val_set)
        print(f"  validation: {len(val_set):,} comparisons, "
              f"{int(val_exact.sum()):,} exact (the RMSE is on those)")

    suffix = (("_4fold" if args.no_val else "")
              + run_tag(args.class_weight, args.seed, model_seed)
              + pair_class_tag(pair_classes)
              # a non-standard schedule length is part of the name too
              + ("" if args.epochs == 60 else f"_e{args.epochs}")
              # a non-default validation fold too, so sweeps on different
              # validation folds never overwrite each other
              + ("" if args.no_val or args.val_fold is None
                 or args.val_fold == (args.fold + 1) % args.n_folds
                 else f"_v{args.val_fold}"))
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / f"results/train_{args.split}_fold{args.fold}{suffix}.csv")
    checkpoint_path = REPO_ROOT / f"models/{args.split}_fold{args.fold}{suffix}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    fit_set = fit_loader = None
    fit_rows = []
    if args.fit_every:
        fit_set = fit_sample(train_set, seed=args.seed)
        fit_loader = DataLoader(fit_set, batch_size=256, shuffle=False,
                                collate_fn=collate_pairs)
        fit_path = REPO_ROOT / f"results/fit_{args.split}_fold{args.fold}{suffix}.csv"

    def record_fit(step, epoch, window):
        """Fit on the bioisostere training sample, then back to train mode."""
        predicted, truth, _ = evaluate(model, fit_loader, device)
        model.train()
        fit_rows.append({"step": step, "epoch": step / len(train_loader),
                         "fit_rmse": rmse(truth, predicted),
                         "window_train_loss": window,
                         "minutes": (time.time() - started) / 60.0})
        pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
        print(f"    step {step:>7,}  bioisostere training-sample RMSE "
              f"{fit_rows[-1]['fit_rmse']:.4f}  window loss {window:.4f}", flush=True)

    history = []
    best_rmse, best_epoch, since_best = float("inf"), -1, 0
    started = time.time()
    print(f"\n{'epoch':>5} {'train loss':>10} {'reg':>7} {'cls':>7} "
          f"{'val rmse':>9} {'val mcc':>8} {'head mcc':>8} {'lr':>8} {'min':>6}")

    step, window_loss, window_batches = 0, 0.0, 0
    if fit_loader is not None:
        record_fit(0, 0, float("nan"))
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        totals = {"loss": 0.0, "regression": 0.0, "classification": 0.0,
                  "n_regression": 0.0, "n_class": 0.0, "batches": 0}
        for batch in train_loader:
            batch = to_device(batch, device)
            parts = loss_fn(model(batch), batch)
            optimiser.zero_grad(set_to_none=True)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            for key in ("loss", "regression", "classification", "n_regression",
                        "n_class"):
                totals[key] += float(parts[key])
            totals["batches"] += 1
            step += 1
            window_loss += float(parts["loss"])
            window_batches += 1
            if fit_loader is not None and (step % args.fit_every == 0
                                           or step == (epoch + 1) * len(train_loader)):
                record_fit(step, epoch, window_loss / window_batches)
                window_loss, window_batches = 0.0, 0

        if val_loader is None:
            val_rmse = val_mcc = head_mcc = float("nan")
        else:
            predicted, truth, head_class = evaluate(model, val_loader, device)
            val_rmse = rmse(truth[val_exact], predicted[val_exact])
            val_mcc = mcc_three_class(truth[val_exact], predicted[val_exact])
            head_mcc = mcc_three_class(
                truth[val_exact],
                # The head's class as a delta-like value, so the same binning applies.
                np.where(head_class[val_exact] == 0, -2.0,
                         np.where(head_class[val_exact] == 2, 2.0, 0.0)))
        minutes = (time.time() - started) / 60.0
        batches = max(totals["batches"], 1)
        history.append({
            "epoch": epoch, "train_loss": totals["loss"] / batches,
            "train_regression": totals["regression"] / batches,
            "train_classification": totals["classification"] / batches,
            "rows_with_regression_loss": totals["n_regression"],
            "rows_with_class_loss": totals["n_class"],
            "val_rmse": val_rmse, "val_mcc": val_mcc, "val_head_mcc": head_mcc,
            "lr": schedule.get_last_lr()[0], "minutes": minutes})

        improved = keeps_checkpoint(val_loader is not None, args.keep_final,
                                    val_rmse, best_rmse)
        if improved:
            best_rmse, best_epoch, since_best = val_rmse, epoch, 0
            torch.save({"model": model.state_dict(), "vocabs": vocabs,
                        "epoch": epoch, "val_rmse": val_rmse,
                        "fusion": args.fusion, "split": args.split,
                        "fold": args.fold, "seed": args.seed,
                        "train_folds": 4 if args.no_val else 3,
                        # which fold was held out as validation, so a later
                        # script can verify it instead of trusting the file
                        # name; None with --no-val
                        "val_fold": None if args.no_val else val_fold,
                        "keep_final": args.keep_final,
                        "class_weight": args.class_weight,
                        "model_seed": model_seed,
                        "pair_classes": pair_classes, "epochs": args.epochs,
                        "n_pair_classes": n_pair_classes}, checkpoint_path)
        else:
            since_best += 1

        print(f"{epoch:>5} {history[-1]['train_loss']:>10.4f} "
              f"{history[-1]['train_regression']:>7.4f} "
              f"{history[-1]['train_classification']:>7.4f} "
              f"{val_rmse:>9.4f} {val_mcc:>8.4f} {head_mcc:>8.4f} "
              f"{history[-1]['lr']:>8.2e} {minutes:>6.1f}"
              + ("  *best" if improved and val_loader is not None
                 and not args.keep_final else ""))

        pd.DataFrame(history).to_csv(out_path, index=False)
        if val_loader is not None and since_best >= args.patience:
            print(f"\nearly stop: no improvement for {args.patience} epochs")
            break
        if args.max_minutes and minutes >= args.max_minutes:
            print(f"\nstopped at --max-minutes {args.max_minutes}")
            break

    if val_loader is None:
        print(f"\nfinal epoch {best_epoch} kept (no validation fold), "
              f"checkpoint {checkpoint_path}")
    elif args.keep_final:
        print(f"\nfinal epoch {best_epoch} kept (--keep-final; the validation fold was "
              f"held out of training but did not select the model), validation RMSE "
              f"{best_rmse:.4f}, checkpoint {checkpoint_path}")
    else:
        print(f"\nbest validation RMSE {best_rmse:.4f} at epoch {best_epoch}, "
              f"checkpoint {checkpoint_path}")
    print(f"Summary: {len(history)} epochs written to {out_path}; the test fold "
          f"({len(test_set):,} comparisons) was not scored.")


if __name__ == "__main__":
    main()
