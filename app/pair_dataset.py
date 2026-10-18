"""PyTorch Dataset over the bioisostere pairs, for the Siamese model.

WHAT ONE ITEM IS
----------------
One comparison: molecule A, molecule B (the same molecule with a different
acidic headgroup), protein target T, and the label Delta p = pIC50(B) - pIC50(A).
Each item carries:

  graph_a, graph_b   chemprop MolGraphs (atoms, bonds, which atom bonds to which)
  swap_idx           the DIRECTED headgroup swap, e.g. acid -> tetrazole
  reverse_swap_idx   the same swap backwards, for app/model.py's mirrored pass
  family_idx         ChEMBL protein family, level 2 (transfers to unseen proteins:
                     +0.92 RMSE points beyond the swap class on molecule-disjoint
                     target folds; an earlier +2.43 was measured on folds that
                     shared molecules and overstated it)
  target_idx         the protein itself (+5.69 points on molecule-disjoint
                     constant folds when the protein has training data, against
                     an earlier +10.08 on shared folds; 0 when it does not)
  delta              Delta p, float
  label_class        0 = worse (Delta p < -1), 1 = same, 2 = better (Delta p > +1)
  censor_a, censor_b index into CENSOR_LABELS: exact / upper / lower / mixed

There is deliberately NO local-environment input: measured at +0.41 points on
the series split, it was dropped.

UNSEEN VALUES GET INDEX 0
-------------------------
Every vocabulary (swap, family, target) is built from the TRAINING rows only and
reserves index 0 for "not seen in training". On a target split every test
protein is new, so every test row gets target_idx 0. That is the honest input:
the model must fall back on the family, which is the whole point of having it.
Building the vocabulary over all rows would instead hand the model an untrained
embedding with a real-looking index.

THE TWO AUGMENTATIONS
---------------------
antisymmetry - for every (A, B, d) also yield (B, A, -d). pairs.csv ALREADY
    contains both directions of every comparison (checked: 57,184 of 57,184
    bioisostere rows have their mirror, with delta negated and censoring
    swapped). Adding mirrors on top would count each comparison four times. So
    the dataset always first collapses the input to one row per comparison
    (`one_row_per_comparison`, the existing scorer's helper) and then, only if
    antisymmetry is on, writes the mirror back. Off means one direction per
    comparison; on means both. Either way nothing is double-counted, whatever
    the input held.

identity - (A, A, 0): a molecule compared with itself must change by exactly
    zero. Added as a fixed fraction of the items (default 5%), drawn with the
    seed from molecules that are already in this dataset, so a training set
    never gains a molecule it did not have. They are mixed into ordinary
    batches by the shuffle rather than forming batches of their own.

TARGET DROPOUT
--------------
Index 0 ("unseen protein") never occurs in training on its own, so without
help its embedding would stay at its random initial value and the model would
read noise exactly where it most needs to degrade gracefully: on a new
protein. So each epoch a fraction of training items has target_idx blanked to
0. family_idx is NEVER blanked - family transfers to unseen proteins and
identity does not, so a blanked row still carries its family. The draw is
seeded by (seed, epoch) and redrawn by set_epoch, so a row is not blanked in
every epoch and a run is reproducible.

The rate is not a convention: `fold_datasets` measures it from the split as
the largest fraction of any fold's test comparisons whose protein is absent
from that fold's training rows. Measured on the real data, molecule-disjoint:
constant 4.8%, scaffold 10.9%, target 100% (by construction).

LEAKAGE (rule 1)
----------------
`fold_datasets` builds train and test from the existing splitters with
require_disjoint_molecules=True. The default constant split is NOT
molecule-disjoint on the real data - 2,201 molecules sit in two folds, because
a molecule with several cuttable bonds belongs to several series - so the
strict version is the default here. It keeps 45,806 of 57,184 rows (constant)
or 44,140 (target).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.utils import make_mol

from app.join import CENSOR_LABELS
from app.splits import (
    SEED, assert_molecules_disjoint, constant_split, directed_swap,
    one_row_per_comparison, scaffold_split, target_split,
)

UNKNOWN = 0                 # vocabulary index for a value not seen in training
CLASS_THRESHOLD = 1.0       # log units: one order of magnitude in IC50
WORSE, SAME, BETTER = 0, 1, 2
CLASS_NAMES = ("worse", "same", "better")

# Pair classes from pairs.csv that may TRAIN the model (--pair-classes, decision
# 163). The test set is always bioisostere only. not_a_swap is left out: 2,090
# rows that failed the swap check.
TRAINABLE_PAIR_CLASSES = ("bioisostere", "same_headgroup", "to_non_acid", "deletion")
DEFAULT_PAIR_CLASS = "bioisostere"   # for rows from a file with no pair_class column

REQUIRED_COLUMNS = ["parent_a_id", "smiles_a", "parent_b_id", "smiles_b",
                    "headgroup_a", "headgroup_b", "target_chembl_id",
                    "family_l2", "delta_p", "censored_a", "censored_b"]


def three_class(delta):
    """0 worse (< -1), 1 same, 2 better (> +1). Exactly +-1.00 counts as same.

    The boundary matters: pchembl values are rounded to two decimals, and 962
    bioisostere rows have |Delta p| of exactly 1.00.
    """
    if delta < -CLASS_THRESHOLD:
        return WORSE
    if delta > CLASS_THRESHOLD:
        return BETTER
    return SAME


def load_bioisostere_pairs(pairs_path, properties_path):
    """Read pairs.csv, keep bioisostere rows, attach family_l2. Prints counts."""
    pairs = pd.read_csv(pairs_path, low_memory=False)
    print(f"  pairs read: {len(pairs):,} rows from {pairs_path}")
    keep = pairs["pair_class"] == "bioisostere"
    pairs = pairs[keep].reset_index(drop=True)
    print(f"  pair_class == bioisostere: {len(pairs):,} rows kept, "
          f"{int((~keep).sum()):,} dropped")
    return attach_family(pairs, pd.read_csv(properties_path))


def parse_pair_classes(text):
    """'bioisostere,to_non_acid' or 'all' -> a sorted list, bioisostere always in it."""
    classes = list(TRAINABLE_PAIR_CLASSES) if text == "all" else \
        [c.strip() for c in text.split(",") if c.strip()]
    unknown = [c for c in classes if c not in TRAINABLE_PAIR_CLASSES]
    if unknown:
        raise ValueError(f"unknown pair classes {unknown}; use {TRAINABLE_PAIR_CLASSES} or 'all'")
    if "bioisostere" not in classes:
        raise ValueError("bioisostere must be among the training classes: it is the test class")
    return sorted(set(classes))


def pair_class_tag(classes):
    """File-name tag: '' for bioisostere only (every existing run), '_allpc' for
    all four trainable classes, else e.g. '_pc-bioisostere-deletion'."""
    if classes == ["bioisostere"]:
        return ""
    if set(classes) == set(TRAINABLE_PAIR_CLASSES):
        return "_allpc"
    return "_pc-" + "-".join(classes)


def load_extra_pairs(pairs_path, properties_path, pair_classes, columns):
    """The NON-bioisostere rows of the requested classes from pairs.csv.

    Only `columns` are read (the slim bioisostere file's columns), because the
    full file is 1.29M rows and this machine has 8 GB. Prints counts per class.
    """
    extra = [c for c in pair_classes if c != "bioisostere"]
    unknown = [c for c in extra if c not in TRAINABLE_PAIR_CLASSES]
    if unknown:
        raise ValueError(f"unknown pair classes {unknown}; use {TRAINABLE_PAIR_CLASSES}")
    if not extra:
        return None
    pairs = pd.read_csv(pairs_path, usecols=columns, low_memory=False)
    print(f"  pairs read: {len(pairs):,} rows from {pairs_path}")
    keep = pairs["pair_class"].isin(extra)
    for pair_class in extra:
        print(f"    {pair_class:<15} {int((pairs['pair_class'] == pair_class).sum()):>9,}")
    pairs = pairs[keep].reset_index(drop=True)
    print(f"  extra training classes {extra}: {len(pairs):,} rows kept, "
          f"{int((~keep).sum()):,} dropped (bioisostere comes from its own file)")
    return attach_family(pairs, pd.read_csv(properties_path))


def extra_training_rows(extra, test_rows):
    """The extra-class rows that may train a model tested on `test_rows`.

    The bioisostere folds are NOT rebuilt: rebuilding them over all classes
    would move bioisostere rows between folds and change the test set, and the
    test set has to stay the 16,116 comparisons every earlier number used.
    Instead every extra row is a candidate for training and is dropped if it
    could leak:
      shares a molecule with any test row (rule 1: the same molecule must
          never be on both sides), or
      has the same constant part as a test row: an acid -> amide pair cut at
          the headgroup of a test series is the test series itself, with one
          more molecule; the constant split exists to hold series out.
    Returns (rows kept, {reason: count}).
    """
    test_molecules = set(test_rows["parent_a_id"]) | set(test_rows["parent_b_id"])
    test_constants = set(test_rows["constant_smiles"])
    shares_molecule = (extra["parent_a_id"].isin(test_molecules)
                       | extra["parent_b_id"].isin(test_molecules))
    same_series = extra["constant_smiles"].isin(test_constants)
    keep = ~shares_molecule & ~same_series
    counts = {"extra rows": len(extra),
              "share a molecule with the test fold": int(shares_molecule.sum()),
              "same constant part as a test series (no shared molecule)":
                  int((same_series & ~shares_molecule).sum()),
              "kept for training": int(keep.sum())}
    for reason, n in counts.items():
        print(f"    {reason:<58} {n:>9,}")
    kept = extra[keep]
    left = (set(kept["parent_a_id"]) | set(kept["parent_b_id"])) & test_molecules
    if left:
        raise AssertionError(f"{len(left)} test molecules still in extra training rows")
    return kept, counts


def attach_family(pairs, properties):
    """Merge family_l2 by target; a target with no property row is 'unclassified'."""
    merged = pairs.merge(properties[["target_chembl_id", "family_l2"]],
                         on="target_chembl_id", how="left")
    if len(merged) != len(pairs):
        raise ValueError(f"family merge changed the row count: {len(pairs):,} -> "
                         f"{len(merged):,}; target_properties has duplicate targets")
    missing = int(merged["family_l2"].isna().sum())
    print(f"  rows with no family_l2: {missing:,} of {len(merged):,} "
          f"(set to 'unclassified')")
    merged["family_l2"] = merged["family_l2"].fillna("unclassified")
    return merged


def build_vocab(values):
    """{value: index}, sorted so the result is the same every run, 0 reserved."""
    vocab = {}
    for value in sorted(set(values), key=str):
        vocab[value] = len(vocab) + 1
    return vocab


def build_vocabs(train_rows):
    """Swap, family and target vocabularies from the TRAINING rows only.

    Both directions of each swap go in, because antisymmetry will produce the
    reverse one even if the collapsed input only held the forward one.
    """
    swaps = []
    for a, b in zip(train_rows["headgroup_a"], train_rows["headgroup_b"]):
        swaps.append(directed_swap(a, b))
        swaps.append(directed_swap(b, a))
        # Identity pairs need the same-headgroup key for both of their headgroups.
        swaps.append(directed_swap(a, a))
        swaps.append(directed_swap(b, b))
    vocabs = {"swap": build_vocab(swaps),
              "family": build_vocab(train_rows["family_l2"]),
              "target": build_vocab(train_rows["target_chembl_id"])}
    # Only when the rows carry more than one class: a bioisostere-only run keeps
    # exactly the vocabularies (and so the model) it always had.
    if "pair_class" in train_rows.columns and train_rows["pair_class"].nunique() > 1:
        vocabs["pair_class"] = build_vocab(train_rows["pair_class"])
    return vocabs


def _item(vocabs, id_a, smiles_a, hg_a, censor_a, id_b, smiles_b, hg_b, censor_b,
          target, family, delta, kind, pair_class=DEFAULT_PAIR_CLASS):
    """One item as a plain dict, before featurisation."""
    return {"pair_class_idx": vocabs.get("pair_class", {}).get(pair_class, UNKNOWN),
            "id_a": id_a, "smiles_a": smiles_a, "censor_a": censor_a,
            "id_b": id_b, "smiles_b": smiles_b, "censor_b": censor_b,
            "swap_idx": vocabs["swap"].get(directed_swap(hg_a, hg_b), UNKNOWN),
            # The SAME swap written backwards. app/model.py evaluates its trunk
            # once each way and subtracts, and the mirrored pass needs the
            # reverse swap's own vocabulary entry rather than an assumption
            # that it is the negation of this one. build_vocabs puts both
            # directions in, so this misses only when the forward one does.
            "reverse_swap_idx": vocabs["swap"].get(directed_swap(hg_b, hg_a),
                                                   UNKNOWN),
            "family_idx": vocabs["family"].get(family, UNKNOWN),
            "target_idx": vocabs["target"].get(target, UNKNOWN),
            "target": target, "delta": delta, "kind": kind}


class PairDataset(Dataset):
    """Bioisostere comparisons as Siamese-model items. See the module docstring.

    rows: pair rows (one or both directions, it does not matter) with the
          REQUIRED_COLUMNS. Pass only one fold's rows.
    vocabs: from build_vocabs on the training rows - the same dict for the
            train and the test dataset of a fold.
    """

    def __init__(self, rows, vocabs, antisymmetry=True, identity_fraction=0.05,
                 target_dropout=0.0, seed=SEED, name="dataset"):
        missing = [column for column in REQUIRED_COLUMNS if column not in rows.columns]
        if missing:
            raise ValueError(f"PairDataset needs columns {missing}")
        self.name = name
        self.vocabs = vocabs
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        self._graphs = {}

        forward = rows[one_row_per_comparison(rows)].reset_index(drop=True)
        print(f"  {name}: {len(rows):,} rows in, {len(forward):,} after collapsing "
              f"mirrors to one row per comparison")

        has_class = "pair_class" in forward.columns
        items = []
        for row in forward.itertuples(index=False):
            delta = float(row.delta_p)
            pair_class = row.pair_class if has_class else DEFAULT_PAIR_CLASS
            items.append(_item(vocabs, row.parent_a_id, row.smiles_a, row.headgroup_a,
                               row.censored_a, row.parent_b_id, row.smiles_b,
                               row.headgroup_b, row.censored_b, row.target_chembl_id,
                               row.family_l2, delta, "forward", pair_class))
            if antisymmetry:
                # Written as 0.0 - delta so a delta of 0.0 mirrors to 0.0, not -0.0.
                items.append(_item(vocabs, row.parent_b_id, row.smiles_b,
                                   row.headgroup_b, row.censored_b, row.parent_a_id,
                                   row.smiles_a, row.headgroup_a, row.censored_a,
                                   row.target_chembl_id, row.family_l2,
                                   0.0 - delta, "mirror", pair_class))
        n_real = len(items)
        print(f"  {name}: antisymmetry {'on' if antisymmetry else 'off'} -> "
              f"{n_real:,} items")

        n_identity = int(round(identity_fraction * n_real))
        if n_identity:
            items.extend(self._identity_items(forward, n_identity, seed))
        print(f"  {name}: identity pairs {n_identity:,} "
              f"({identity_fraction:.0%} of {n_real:,}) -> {len(items):,} items")
        self.items = items
        self.seed = seed
        self.target_dropout = target_dropout
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Redraw which items have their target blanked, from (seed, epoch).

        Call at the start of every epoch. With DataLoader worker processes,
        call it before the loader's iterator is created, and do not use
        persistent_workers, or the workers keep the old draw.
        """
        rng = np.random.default_rng([self.seed, epoch])
        self.target_dropped = rng.random(len(self.items)) < self.target_dropout
        if self.target_dropout:
            print(f"  {self.name}: epoch {epoch}, target blanked on "
                  f"{int(self.target_dropped.sum()):,} of {len(self.items):,} items "
                  f"(rate {self.target_dropout:.3f}); family kept on all")

    def _identity_items(self, forward, n_identity, seed):
        """(M, M, 0) items, M and its target drawn from this dataset's own rows.

        Each draw picks a row and then a side of it, so a molecule's chance of
        being drawn follows how often it appears, and the target is one it was
        really measured on. Censoring is 'exact' on both sides: the delta of a
        molecule against itself is exactly zero, not a bound.
        """
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, len(forward), size=n_identity)
        sides = rng.integers(0, 2, size=n_identity)
        items = []
        for row_number, side in zip(rows, sides):
            row = forward.iloc[int(row_number)]
            suffix = "a" if side == 0 else "b"
            molecule = row[f"parent_{suffix}_id"]
            smiles = row[f"smiles_{suffix}"]
            headgroup = row[f"headgroup_{suffix}"]
            items.append(_item(self.vocabs, molecule, smiles, headgroup, "exact",
                               molecule, smiles, headgroup, "exact",
                               row["target_chembl_id"], row["family_l2"], 0.0,
                               "identity", row.get("pair_class", DEFAULT_PAIR_CLASS)))
        return items

    def molecule_ids(self):
        """Every molecule id that any item of this dataset will show the model."""
        ids = set()
        for item in self.items:
            ids.add(item["id_a"])
            ids.add(item["id_b"])
        return ids

    def graph(self, molecule_id, smiles):
        """The molecule's MolGraph, featurised once and cached by id."""
        if molecule_id not in self._graphs:
            self._graphs[molecule_id] = self.featurizer(make_mol(smiles))
        return self._graphs[molecule_id]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {"graph_a": self.graph(item["id_a"], item["smiles_a"]),
                "graph_b": self.graph(item["id_b"], item["smiles_b"]),
                "swap_idx": item["swap_idx"],
                "reverse_swap_idx": item["reverse_swap_idx"],
                "family_idx": item["family_idx"],
                "pair_class_idx": item["pair_class_idx"],
                "target_idx": UNKNOWN if self.target_dropped[index]
                else item["target_idx"],
                "target_dropped": bool(self.target_dropped[index]),
                "delta": item["delta"],
                "label_class": three_class(item["delta"]),
                "censor_a": CENSOR_LABELS.index(item["censor_a"]),
                "censor_b": CENSOR_LABELS.index(item["censor_b"]),
                "id_a": item["id_a"], "id_b": item["id_b"], "kind": item["kind"]}


def fit_sample(dataset, n=5000, seed=SEED):
    """A copy of a TRAINING dataset cut down to n bioisostere comparisons.

    For watching how well the model fits the class we test on while it trains
    on several: forward items only (one per comparison), both
    sides exact, no target blanking. The draw is over the comparisons sorted by
    (molecule A, molecule B, target) with a fixed seed, so a bioisostere-only
    run and an all-class run on the same fold get the SAME comparisons and their
    numbers can be set side by side. The molecule-graph cache is shared, not
    copied.
    """
    import copy
    bioisostere = dataset.vocabs.get("pair_class", {}).get("bioisostere", UNKNOWN)
    picks = [i for i, item in enumerate(dataset.items)
             if item["kind"] == "forward" and item["pair_class_idx"] == bioisostere
             and item["censor_a"] == "exact" and item["censor_b"] == "exact"]
    picks.sort(key=lambda i: (str(dataset.items[i]["id_a"]), str(dataset.items[i]["id_b"]),
                              str(dataset.items[i]["target"])))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(picks), size=min(n, len(picks)), replace=False))
    sample = copy.copy(dataset)
    sample.items = [dataset.items[picks[j]] for j in chosen]
    sample.name = "fit sample"
    sample.target_dropout = 0.0
    sample.target_dropped = np.zeros(len(sample.items), dtype=bool)
    print(f"  fit sample: {len(sample.items):,} of {len(picks):,} bioisostere exact "
          f"training comparisons")
    return sample


def collate_pairs(batch):
    """A list of items -> one batch: two BatchMolGraphs plus stacked tensors.

    The molecule ids and item kinds pass through as lists, so a training loop
    (or a test) can check exactly which molecules a batch contained.
    """
    out = {"graph_a": BatchMolGraph([item["graph_a"] for item in batch]),
           "graph_b": BatchMolGraph([item["graph_b"] for item in batch])}
    for key in ("swap_idx", "reverse_swap_idx", "family_idx", "target_idx",
                "pair_class_idx", "label_class", "censor_a", "censor_b"):
        out[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    out["delta"] = torch.tensor([item["delta"] for item in batch], dtype=torch.float32)
    out["target_dropped"] = torch.tensor([item["target_dropped"] for item in batch],
                                         dtype=torch.bool)
    for key in ("id_a", "id_b", "kind"):
        out[key] = [item[key] for item in batch]
    return out


SPLITS = {"constant": constant_split, "target": target_split}


def unknown_target_fractions(pairs, result):
    """{fold: fraction of its test comparisons whose protein never trains}.

    Counted over one row per comparison, the way scoring counts them, so a
    comparison measured in both directions is not counted twice.
    """
    kept = result.kept()
    one = one_row_per_comparison(pairs)
    fractions = {}
    for fold in sorted(set(result.folds[kept])):
        trained = set(pairs.loc[kept & (result.folds != fold), "target_chembl_id"])
        tested = pairs.loc[kept & (result.folds == fold) & one, "target_chembl_id"]
        unknown = 0
        for target in tested:
            if target not in trained:
                unknown += 1
        fractions[int(fold)] = unknown / len(tested) if len(tested) else 0.0
    return fractions


def _split_result(pairs, split, n_folds, seed):
    """The fold assignment, always molecule-disjoint, asserted not assumed."""
    if split == "scaffold":
        result = scaffold_split(pairs, n_folds=n_folds, seed=seed)
    elif split in SPLITS:
        result = SPLITS[split](pairs, n_folds=n_folds, seed=seed,
                               require_disjoint_molecules=True)
    else:
        raise ValueError(f"unknown split {split!r}; use constant, target or scaffold")
    assert_molecules_disjoint(pairs, result)
    print(result.summary())
    return result


def _measured_dropout(pairs, result, target_dropout):
    if target_dropout != "measured":
        return target_dropout
    fractions = unknown_target_fractions(pairs, result)
    rate = max(fractions.values())
    print(f"  unknown-target fraction of test comparisons per fold: "
          + ", ".join(f"{f}: {v:.3f}" for f, v in fractions.items())
          + f" -> target dropout {rate:.3f}")
    return rate


def _assert_disjoint(named_datasets):
    """No molecule may appear in two of the given datasets."""
    names = list(named_datasets)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = named_datasets[first].molecule_ids() \
                & named_datasets[second].molecule_ids()
            if shared:
                raise AssertionError(
                    f"{len(shared)} molecule(s) in both {first} and {second}, "
                    f"e.g. {sorted(shared)[:3]}")


def train_val_test_datasets(pairs, split="constant", fold=0, val_fold=None,
                            n_folds=5, seed=SEED, antisymmetry=True,
                            identity_fraction=0.05, target_dropout="measured",
                            extra=None):
    """(train, val, test): a validation fold taken from the TRAINING folds.

    Early stopping needs a held-out set, and it must never be the test fold -
    stopping on the test fold is choosing the number of epochs with the answer
    in hand, which is the same mistake as fitting a shrinkage strength on the
    test fold. So one of the remaining folds becomes the
    validation set and training uses the other three.

    val_fold defaults to the fold after the test one, wrapping round. All three
    molecule sets are checked disjoint, and the vocabularies are built from the
    TRAINING rows only - not train plus validation - so validation measures the
    same unseen-value handling that the test fold will.
    """
    result = _split_result(pairs, split, n_folds, seed)
    folds = sorted(set(result.folds[result.kept()]))
    if fold not in folds:
        raise ValueError(f"fold {fold} is not one of {folds}")
    if val_fold is None:
        val_fold = folds[(folds.index(fold) + 1) % len(folds)]
    if val_fold == fold:
        raise ValueError(f"val_fold must differ from the test fold {fold}")
    if val_fold not in folds:
        raise ValueError(f"val_fold {val_fold} is not one of {folds}")

    kept = result.kept()
    train_rows = pairs[kept & (result.folds != fold) & (result.folds != val_fold)]
    val_rows = pairs[kept & (result.folds == val_fold)]
    test_rows = pairs[kept & (result.folds == fold)]
    print(f"  test fold {fold}, validation fold {val_fold}: "
          f"{len(train_rows):,} train rows, {len(val_rows):,} val rows, "
          f"{len(test_rows):,} test rows")
    if extra is not None:
        # Filtered against BOTH held-out folds: a molecule or series of the
        # validation fold in training would leak into the epoch choice just as
        # one of the test fold would leak into the score.
        print(f"  extra training rows against validation fold {val_fold} and "
              f"test fold {fold}:")
        kept_extra, _ = extra_training_rows(extra, pd.concat([val_rows, test_rows]))
        train_rows = pd.concat([train_rows, kept_extra], ignore_index=True)
        print(f"  {len(train_rows):,} train rows with the extra classes")

    target_dropout = _measured_dropout(pairs, result, target_dropout)
    vocabs = build_vocabs(train_rows)
    datasets = {
        "train": PairDataset(train_rows, vocabs, antisymmetry=antisymmetry,
                             identity_fraction=identity_fraction,
                             target_dropout=target_dropout, seed=seed, name="train"),
        "val": PairDataset(val_rows, vocabs, antisymmetry=False,
                           identity_fraction=0.0, seed=seed, name="val"),
        "test": PairDataset(test_rows, vocabs, antisymmetry=False,
                            identity_fraction=0.0, seed=seed, name="test")}
    _assert_disjoint(datasets)
    return datasets["train"], datasets["val"], datasets["test"]


def fold_datasets(pairs, split="constant", fold=0, n_folds=5, seed=SEED,
                  antisymmetry=True, identity_fraction=0.05,
                  target_dropout="measured", extra=None):
    """(train, test) PairDatasets for one fold, from the existing splitters.

    target_dropout="measured" sets the rate to the largest unknown-target
    fraction over this split's folds (see the module docstring); pass a number
    to override it.

    extra: rows of other pair classes (load_extra_pairs). They join TRAINING
    only, after extra_training_rows drops every row that could leak; the folds,
    the test rows and the measured target-dropout rate stay the bioisostere
    ones.

    The split is always molecule-disjoint (scaffold is by construction; the
    others are asked for it), and that is asserted, not assumed. The test set
    gets no augmentation: one row per comparison is how scoring counts them, and identity pairs carry no information to score.
    """
    result = _split_result(pairs, split, n_folds, seed)
    kept = result.kept()
    train_rows = pairs[kept & (result.folds != fold)]
    test_rows = pairs[kept & (result.folds == fold)]
    print(f"  fold {fold}: {len(train_rows):,} train rows, {len(test_rows):,} test rows")
    if extra is not None:
        print(f"  extra training rows against test fold {fold}:")
        kept_extra, _ = extra_training_rows(extra, test_rows)
        train_rows = pd.concat([train_rows, kept_extra], ignore_index=True)
        print(f"  fold {fold}: {len(train_rows):,} train rows with the extra classes")

    target_dropout = _measured_dropout(pairs, result, target_dropout)
    vocabs = build_vocabs(train_rows)
    train = PairDataset(train_rows, vocabs, antisymmetry=antisymmetry,
                        identity_fraction=identity_fraction,
                        target_dropout=target_dropout, seed=seed, name="train")
    test = PairDataset(test_rows, vocabs, antisymmetry=False, identity_fraction=0.0,
                       seed=seed, name="test")
    _assert_disjoint({"train": train, "test": test})
    return train, test
