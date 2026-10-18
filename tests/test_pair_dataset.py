"""Tests for app/pair_dataset.py on a synthetic pair table.

Three things the loader must get right, because a mistake in any of them is
silent - training still runs and the numbers still come out:

  1. augmentation keeps the data intact: a mirror is the same comparison
     written backwards (graphs swapped, delta negated, censoring swapped,
     class flipped), and nothing is counted twice even though the input
     already holds both directions;
  2. identity pairs are exactly (M, M, 0);
  3. no molecule from a test fold ever reaches a training batch.

The table mimics pairs.csv: both directions of every comparison, four targets,
real SMILES (chemprop parses them). Some molecule pairs are deliberately listed
under TWO constants, which is what happens on the real data when a molecule has
two cuttable bonds - it is the reason the loose constant split shares molecules
between folds.
"""

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from app.join import CENSOR_LABELS
from app.pair_dataset import (
    BETTER, SAME, UNKNOWN, WORSE, PairDataset, attach_family, build_vocabs,
    collate_pairs, fold_datasets, three_class,
)
from app.splits import constant_split, one_row_per_comparison

# Written so the first and last atoms are ring carbons: the headgroup bonds to
# the first atom (acid, acylsulfonamide) or the last (tetrazole), and a bond to
# an aromatic nitrogen would be a valence error RDKit rightly refuses.
RINGS = ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccsc1", "c1nccnc1",
         "c1ccc(F)cc1", "c1ccc(Cl)cc1", "c1ccc(C)cc1"]
HEADGROUPS = [("carboxylic_acid", "OC(=O){}"),
              ("tetrazole", "{}-c1nnn[nH]1"),
              ("acylsulfonamide", "CS(=O)(=O)NC(=O){}")]
CENSORS = ["exact", "upper", "lower", "mixed"]
FAMILIES = {"T0": "Protease", "T1": "Kinase", "T2": "Protease", "T3": "Hydrolase"}


def targets_of(series):
    """Mostly one protein per series, so a target split has something to keep;
    the last series is on two, so the strict split has something to drop."""
    if series == len(RINGS) - 1:
        return (f"T{series % 4}", "T0")
    return (f"T{series % 4}",)


def build_pairs():
    """Both directions of every within-series pair, on each series' targets."""
    rows = []
    counter = 0
    for s, ring in enumerate(RINGS):
        molecules = [(f"S{s}_M{h}", name, template.format(ring))
                     for h, (name, template) in enumerate(HEADGROUPS)]
        constants = [f"[*:1]{ring}"]
        if s % 2 == 0:
            # The same molecules under a second constant: a bridge between series.
            constants.append(f"[*:1]{ring}_alt")
        for constant in constants:
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    for target in targets_of(s):
                        counter += 1
                        # Covers both class boundaries, exact +-1 and 0.
                        delta = [-1.5, -1.0, -0.3, 0.0, 0.4, 1.0, 1.01, 2.2][counter % 8]
                        id_a, hg_a, smi_a = molecules[i]
                        id_b, hg_b, smi_b = molecules[j]
                        c_a, c_b = CENSORS[counter % 4], CENSORS[(counter + 1) % 4]
                        common = {"pair_class": "bioisostere", "constant_smiles": constant,
                                  "target_chembl_id": target, "standard_type": "IC50",
                                  "year": 2015}
                        rows.append({**common,
                                     "parent_a_id": id_a, "smiles_a": smi_a, "headgroup_a": hg_a,
                                     "parent_b_id": id_b, "smiles_b": smi_b, "headgroup_b": hg_b,
                                     "delta_p": delta, "censored_a": c_a, "censored_b": c_b})
                        rows.append({**common,
                                     "parent_a_id": id_b, "smiles_a": smi_b, "headgroup_a": hg_b,
                                     "parent_b_id": id_a, "smiles_b": smi_a, "headgroup_b": hg_a,
                                     "delta_p": 0.0 - delta, "censored_a": c_b,
                                     "censored_b": c_a})
    pairs = pd.DataFrame(rows)
    properties = pd.DataFrame({"target_chembl_id": list(FAMILIES),
                               "family_l2": list(FAMILIES.values())})
    return attach_family(pairs, properties)


@pytest.fixture
def pairs():
    return build_pairs()


def real_items(dataset):
    return [item for item in dataset.items if item["kind"] != "identity"]


# --- the class label ---------------------------------------------------------

def test_three_class_boundaries():
    assert three_class(-1.01) == WORSE
    assert three_class(-1.0) == SAME          # exactly -1 is not "below -1"
    assert three_class(0.0) == SAME
    assert three_class(1.0) == SAME
    assert three_class(1.01) == BETTER


# --- augmentation integrity ----------------------------------------------------

def test_mirrors_in_the_input_are_not_double_counted(pairs):
    """pairs.csv already holds both directions; antisymmetry must not add more."""
    vocabs = build_vocabs(pairs)
    n_comparisons = int(one_row_per_comparison(pairs).sum())
    assert n_comparisons * 2 == len(pairs)

    on = PairDataset(pairs, vocabs, antisymmetry=True, identity_fraction=0.0)
    off = PairDataset(pairs, vocabs, antisymmetry=False, identity_fraction=0.0)
    assert len(off) == n_comparisons
    assert len(on) == 2 * n_comparisons == len(pairs)

    # And the result does not depend on whether the input held the mirrors.
    one_way = pairs[one_row_per_comparison(pairs)]
    assert len(PairDataset(one_way, vocabs, identity_fraction=0.0)) == len(on)


def test_antisymmetry_reproduces_the_file_exactly(pairs):
    """Every generated item is a row of the file, and every row is generated once."""
    vocabs = build_vocabs(pairs)
    dataset = PairDataset(pairs, vocabs, antisymmetry=True, identity_fraction=0.0)
    generated = sorted((item["id_a"], item["id_b"], item["target"], item["delta"],
                        item["censor_a"], item["censor_b"]) for item in dataset.items)
    in_file = sorted(zip(pairs["parent_a_id"], pairs["parent_b_id"],
                         pairs["target_chembl_id"], pairs["delta_p"].astype(float),
                         pairs["censored_a"], pairs["censored_b"]))
    assert generated == in_file


def test_each_mirror_is_its_forward_item_reversed(pairs):
    vocabs = build_vocabs(pairs)
    dataset = PairDataset(pairs, vocabs, antisymmetry=True, identity_fraction=0.0)
    for k in range(0, len(dataset), 2):
        forward, mirror = dataset[k], dataset[k + 1]
        assert dataset.items[k]["kind"] == "forward"
        assert dataset.items[k + 1]["kind"] == "mirror"
        assert (mirror["id_a"], mirror["id_b"]) == (forward["id_b"], forward["id_a"])
        assert mirror["graph_a"] is forward["graph_b"]
        assert mirror["graph_b"] is forward["graph_a"]
        assert mirror["delta"] == -forward["delta"]
        assert (mirror["censor_a"], mirror["censor_b"]) == (forward["censor_b"],
                                                            forward["censor_a"])
        assert mirror["label_class"] == 2 - forward["label_class"]
        # Same protein, same family; the swap is the reverse direction.
        assert mirror["target_idx"] == forward["target_idx"]
        assert mirror["family_idx"] == forward["family_idx"]
        assert mirror["swap_idx"] != forward["swap_idx"]


def test_a_zero_delta_mirrors_to_positive_zero(pairs):
    vocabs = build_vocabs(pairs)
    dataset = PairDataset(pairs, vocabs, identity_fraction=0.0)
    zeros = [item["delta"] for item in dataset.items if item["delta"] == 0.0]
    assert zeros and all(str(value) == "0.0" for value in zeros)


# --- identity pairs -------------------------------------------------------------

def test_identity_pairs_carry_exactly_zero(pairs):
    vocabs = build_vocabs(pairs)
    dataset = PairDataset(pairs, vocabs, identity_fraction=0.05)
    identity = [k for k, item in enumerate(dataset.items) if item["kind"] == "identity"]
    assert len(identity) == round(0.05 * len(real_items(dataset)))
    for k in identity:
        item = dataset[k]
        assert item["delta"] == 0.0 and str(item["delta"]) == "0.0"
        assert item["label_class"] == SAME
        assert item["id_a"] == item["id_b"]
        assert item["graph_a"] is item["graph_b"]
        assert item["censor_a"] == item["censor_b"] == CENSOR_LABELS.index("exact")
        assert item["swap_idx"] != UNKNOWN


def test_identity_pairs_add_no_new_molecules_and_are_seeded(pairs):
    vocabs = build_vocabs(pairs)
    with_identity = PairDataset(pairs, vocabs, identity_fraction=0.2, seed=42)
    without = PairDataset(pairs, vocabs, identity_fraction=0.0)
    assert with_identity.molecule_ids() == without.molecule_ids()

    again = PairDataset(pairs, vocabs, identity_fraction=0.2, seed=42)
    assert [i["id_a"] for i in again.items] == [i["id_a"] for i in with_identity.items]


def test_identity_target_is_one_the_molecule_was_measured_on(pairs):
    measured = set(zip(pairs["parent_a_id"], pairs["target_chembl_id"]))
    dataset = PairDataset(pairs, build_vocabs(pairs), identity_fraction=0.5)
    for item in dataset.items:
        if item["kind"] == "identity":
            assert (item["id_a"], item["target"]) in measured


# --- leakage ----------------------------------------------------------------------

def test_the_loose_constant_split_would_leak_on_this_table(pairs):
    """Without this, the strict test below could pass on a table that never
    tested anything."""
    result = constant_split(pairs, n_folds=3, seed=42)
    assert result.stats["molecules shared between folds"] > 0


@pytest.mark.parametrize("split", ["constant", "target", "scaffold"])
def test_no_test_molecule_reaches_a_training_batch(pairs, split):
    for fold in range(3):
        train, test = fold_datasets(pairs, split=split, fold=fold, n_folds=3,
                                    identity_fraction=0.2)
        test_molecules = test.molecule_ids()
        loader = DataLoader(train, batch_size=4, shuffle=True, collate_fn=collate_pairs,
                            generator=torch.Generator().manual_seed(42))
        seen = 0
        for batch in loader:
            assert not (set(batch["id_a"]) | set(batch["id_b"])) & test_molecules
            seen += len(batch["id_a"])
        assert seen == len(train)


def test_test_fold_is_unaugmented_and_unseen_targets_are_unknown(pairs):
    train, test = fold_datasets(pairs, split="target", fold=0, n_folds=3)
    assert all(item["kind"] == "forward" for item in test.items)
    # On a target split no test protein was in training.
    assert all(item["target_idx"] == UNKNOWN for item in test.items)
    assert all(item["target_idx"] != UNKNOWN for item in train.items)


# --- batching -----------------------------------------------------------------------

def test_collate_shapes(pairs):
    dataset = PairDataset(pairs, build_vocabs(pairs))
    batch = collate_pairs([dataset[k] for k in range(5)])
    assert batch["graph_a"].batch.max().item() == 4
    assert batch["graph_b"].batch.max().item() == 4
    for key in ("swap_idx", "family_idx", "target_idx", "label_class", "censor_a",
                "censor_b", "delta"):
        assert batch[key].shape == (5,)
    assert batch["delta"].dtype == torch.float32


# --- target dropout ---------------------------------------------------------------

def test_dropout_blanks_target_but_never_family(pairs):
    dataset = PairDataset(pairs, build_vocabs(pairs), target_dropout=0.5)
    dropped = 0
    for k in range(len(dataset)):
        item = dataset[k]
        assert item["family_idx"] != UNKNOWN
        if item["target_dropped"]:
            dropped += 1
            assert item["target_idx"] == UNKNOWN
        else:
            assert item["target_idx"] == dataset.items[k]["target_idx"] != UNKNOWN
    assert 0.3 * len(dataset) < dropped < 0.7 * len(dataset)


def test_dropout_zero_blanks_nothing_and_leaves_labels_alone(pairs):
    vocabs = build_vocabs(pairs)
    plain = PairDataset(pairs, vocabs, target_dropout=0.0)
    dropped = PairDataset(pairs, vocabs, target_dropout=0.5)
    assert not plain.target_dropped.any()
    for k in range(len(plain)):
        assert plain[k]["delta"] == dropped[k]["delta"]
        assert plain[k]["label_class"] == dropped[k]["label_class"]


def test_dropout_is_redrawn_per_epoch_and_seeded(pairs):
    vocabs = build_vocabs(pairs)
    dataset = PairDataset(pairs, vocabs, target_dropout=0.5, seed=42)
    epoch0 = dataset.target_dropped.copy()
    dataset.set_epoch(1)
    assert (dataset.target_dropped != epoch0).any()
    dataset.set_epoch(0)
    assert (dataset.target_dropped == epoch0).all()
    again = PairDataset(pairs, vocabs, target_dropout=0.5, seed=42)
    assert (again.target_dropped == epoch0).all()


def test_measured_dropout_matches_the_split(pairs):
    """Target split: every test protein is new, so the measured rate is 1.
    The test set is never blanked - its unknowns are real, not simulated."""
    train, test = fold_datasets(pairs, split="target", fold=0, n_folds=3)
    assert train.target_dropout == 1.0
    assert not test.target_dropped.any()

    train, _ = fold_datasets(pairs, split="constant", fold=0, n_folds=3)
    assert 0.0 <= train.target_dropout < 1.0


def test_unknown_target_fractions_counts_comparisons_not_rows(pairs):
    from app.pair_dataset import unknown_target_fractions
    from app.splits import target_split
    result = target_split(pairs, n_folds=3, seed=42, require_disjoint_molecules=True)
    fractions = unknown_target_fractions(pairs, result)
    assert fractions and all(value == 1.0 for value in fractions.values())


# --- the three-way split ------------------------------------------------------------

def test_validation_fold_is_disjoint_from_train_and_test(pairs):
    """Early stopping must not see a test molecule, and must not train on one."""
    from app.pair_dataset import train_val_test_datasets
    train, val, test = train_val_test_datasets(pairs, split="constant", fold=0,
                                               n_folds=3, identity_fraction=0.2)
    assert not train.molecule_ids() & val.molecule_ids()
    assert not train.molecule_ids() & test.molecule_ids()
    assert not val.molecule_ids() & test.molecule_ids()
    assert len(train) and len(val) and len(test)


def test_validation_and_test_are_unaugmented(pairs):
    from app.pair_dataset import train_val_test_datasets
    train, val, test = train_val_test_datasets(pairs, split="constant", fold=0,
                                               n_folds=3, identity_fraction=0.2)
    for dataset in (val, test):
        assert all(item["kind"] == "forward" for item in dataset.items)
        assert not dataset.target_dropped.any()
    assert any(item["kind"] == "identity" for item in train.items)


def test_vocabularies_come_from_training_rows_only(pairs):
    """A protein that appears only in validation must read as unseen, or the
    validation score would flatter the model."""
    from app.pair_dataset import train_val_test_datasets
    train, val, _test = train_val_test_datasets(pairs, split="target", fold=0,
                                                n_folds=3)
    trained_targets = {item["target"] for item in train.items}
    for item in val.items:
        if item["target"] not in trained_targets:
            assert item["target_idx"] == UNKNOWN


def test_the_test_fold_can_never_be_the_validation_fold(pairs):
    from app.pair_dataset import train_val_test_datasets
    with pytest.raises(ValueError, match="must differ"):
        train_val_test_datasets(pairs, split="constant", fold=1, val_fold=1,
                                n_folds=3)


def test_default_validation_fold_wraps_round(pairs):
    """Test fold 2 of 3 should take fold 0 for validation, not fold 3."""
    from app.pair_dataset import train_val_test_datasets
    train, val, test = train_val_test_datasets(pairs, split="constant", fold=2,
                                               n_folds=3)
    assert len(val) and not val.molecule_ids() & test.molecule_ids()
