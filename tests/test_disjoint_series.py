"""Tests for require_disjoint_series on both held-out-chemistry splits.

What the option exists to stop: a chemical series appearing in
training under one swap and in test under another. If training shows that
series S tolerates acid -> acylsulfonamide, the model has learned something
real about S, and then meets S again at test time as acid -> tetrazole. The
fold is still called "hold out tetrazole", but it is no longer a test on
unfamiliar chemistry.

Every test here builds that exact case - one series, two swaps - and checks
that the loose split keeps it and the strict split does not. The loose
assertions matter as much as the strict ones: they are what makes the test
fail if someone makes strictness the default by accident.
"""

import pandas as pd
import pytest

from app.splits import (
    DROPPED,
    assert_headgroup_never_seen,
    assert_mask_folds,
    assert_mirrors_together,
    check_split,
    fold_overlap,
    headgroup_split,
    transformation_folds,
    transformation_split,
)

ACID = "OC(=O)c1ccc(C)cc1"
TETRAZOLE = "Cc1ccc(-c2nnn[nH]2)cc1"
ACYLSULF = "Cc1ccc(C(=O)NS(C)(=O)=O)cc1"
SULFONAMIDE = "Cc1ccc(S(N)(=O)=O)cc1"
HYDROXAMIC = "Cc1ccc(C(=O)NO)cc1"


def pair(a, smiles_a, hg_a, b, smiles_b, hg_b, constant):
    return {"parent_a_id": a, "smiles_a": smiles_a, "headgroup_a": hg_a,
            "parent_b_id": b, "smiles_b": smiles_b, "headgroup_b": hg_b,
            "constant_smiles": constant}


# Series "SHARED" is measured with three different swaps, so under a loose
# split it lands in whichever fold each swap went to. Series "T_ONLY" and
# "A_ONLY" each carry one swap and are never torn.
@pytest.fixture
def pairs():
    rows = []
    for i in range(4):
        rows.append(pair(f"a{i}", ACID, "carboxylic_acid", f"t{i}", TETRAZOLE,
                         "tetrazole", "SHARED"))
        rows.append(pair(f"a{i}", ACID, "carboxylic_acid", f"s{i}", ACYLSULF,
                         "acylsulfonamide", "SHARED"))
        rows.append(pair(f"a{i}", ACID, "carboxylic_acid", f"h{i}", HYDROXAMIC,
                         "hydroxamic_acid", "SHARED"))
    for i in range(4, 8):
        rows.append(pair(f"a{i}", ACID, "carboxylic_acid", f"t{i}", TETRAZOLE,
                         "tetrazole", "T_ONLY"))
        rows.append(pair(f"a{i}", ACID, "carboxylic_acid", f"s{i}", ACYLSULF,
                         "acylsulfonamide", "A_ONLY"))
    return pd.DataFrame(rows)


# --- the transformation split ------------------------------------------------

def test_loose_transformation_split_tears_a_series_across_folds(pairs):
    """The case the option addresses: SHARED is in every fold at once."""
    result = transformation_split(pairs, n_folds=3, seed=42)
    folds = set(result.folds[pairs["constant_smiles"] == "SHARED"])
    assert len(folds) > 1, "fixture no longer reproduces the leak it tests"
    assert result.stats["series shared between folds"] >= 1
    assert not result.series_disjoint


def test_strict_transformation_split_gives_each_series_one_fold(pairs):
    result = transformation_split(pairs, n_folds=3, seed=42,
                                  require_disjoint_series=True)
    kept = result.kept()
    placed = pd.DataFrame({"series": pairs.loc[kept, "constant_smiles"],
                           "fold": result.folds[kept]})
    for series, group in placed.groupby("series"):
        assert group["fold"].nunique() == 1, f"{series} spans {set(group['fold'])}"
    assert result.stats["series shared between folds"] == 0
    assert result.series_disjoint
    check_split(pairs, result)   # includes assert_series_disjoint


def test_strict_transformation_split_costs_pairs_and_says_so(pairs):
    loose = transformation_split(pairs, n_folds=3, seed=42)
    strict = transformation_split(pairs, n_folds=3, seed=42,
                                  require_disjoint_series=True)
    dropped = strict.stats["pairs dropped to keep series disjoint"]
    assert dropped > 0
    assert int(strict.kept().sum()) == int(loose.kept().sum()) - dropped


def test_strict_transformation_split_keeps_the_biggest_fold_of_a_torn_series(pairs):
    """SHARED has 4 pairs in each of its three swaps, so it keeps one fold's
    worth and loses the other two - not all of them."""
    strict = transformation_split(pairs, n_folds=3, seed=42,
                                  require_disjoint_series=True)
    shared = strict.folds[pairs["constant_smiles"] == "SHARED"]
    assert int((shared != DROPPED).sum()) == 4


def test_a_split_that_never_promised_series_disjointness_is_not_checked(pairs):
    """assert_series_disjoint must read the flag, not the data, or every loose
    split would fail the shared check_split path."""
    check_split(pairs, transformation_split(pairs, n_folds=3, seed=42))


def test_strict_transformation_split_needs_the_constant_column(pairs):
    with pytest.raises(ValueError, match="constant_smiles"):
        transformation_split(pairs.drop(columns=["constant_smiles"]), n_folds=3,
                             require_disjoint_series=True)


# --- the transformation split as per-fold masks ------------------------------
#
# The reason this version exists: the headline result of the
# project is "performance drops by X when the series is held out too", and that
# number only means one thing if loose and strict score the SAME test pairs.

def test_the_test_folds_are_identical_loose_and_strict(pairs):
    loose = transformation_folds(pairs, n_folds=3, seed=42)
    strict = transformation_folds(pairs, n_folds=3, seed=42,
                                  require_disjoint_series=True)
    assert [f.fold for f in loose.folds] == [f.fold for f in strict.folds]
    for a, b in zip(loose.folds, strict.folds):
        assert a.test.equals(b.test)


def test_loose_masks_reproduce_the_plain_split(pairs):
    """Loose must be exactly "train on everything else", or this refactor has
    silently changed every number already in the results table."""
    plain = transformation_split(pairs, n_folds=3, seed=42)
    masks = transformation_folds(pairs, n_folds=3, seed=42)
    kept = plain.kept()
    for fold in masks.folds:
        assert fold.test.equals(kept & (plain.folds == fold.fold))
        assert fold.train.equals(kept & (plain.folds != fold.fold))


def test_strict_masks_only_shrink_training(pairs):
    loose = transformation_folds(pairs, n_folds=3, seed=42)
    strict = transformation_folds(pairs, n_folds=3, seed=42,
                                  require_disjoint_series=True)
    shrunk = 0
    for a, b in zip(loose.folds, strict.folds):
        assert (b.train & ~a.train).sum() == 0          # no row gained
        if b.train.sum() < a.train.sum():
            shrunk += 1
        assert b.train_dropped_for_series == int((a.train & ~b.train).sum())
    assert shrunk, "fixture no longer reproduces the leak it tests"


def test_strict_masks_leave_no_test_series_in_training(pairs):
    strict = transformation_folds(pairs, n_folds=3, seed=42,
                                  require_disjoint_series=True)
    assert strict.series_disjoint
    assert_mask_folds(pairs, strict)
    for fold in strict.folds:
        trained = set(pairs.loc[fold.train, "constant_smiles"])
        tested = set(pairs.loc[fold.test, "constant_smiles"])
        assert not (trained & tested)


def test_mask_folds_partition_every_row_in_both_modes(pairs):
    for strict in (False, True):
        result = transformation_folds(pairs, n_folds=3, seed=42,
                                      require_disjoint_series=strict)
        for fold in result.folds:
            covered = (fold.train.astype(int) + fold.test.astype(int)
                       + fold.excluded.astype(int))
            assert (covered == 1).all()
        assert_mask_folds(pairs, result)


def test_the_home_fold_version_is_still_available_and_differs(pairs):
    """Both are kept on purpose. The partition version drops test pairs; the
    mask version does not, and that difference is the whole point."""
    partition = transformation_split(pairs, n_folds=3, seed=42,
                                     require_disjoint_series=True)
    masks = transformation_folds(pairs, n_folds=3, seed=42,
                                 require_disjoint_series=True)
    partition_test = {f: int((partition.folds == f).sum()) for f in range(3)}
    mask_test = {f.fold: int(f.test.sum()) for f in masks.folds}
    assert partition_test != mask_test
    assert sum(mask_test.values()) > sum(partition_test.values())


def test_mask_fold_lookup_works_by_number_and_name(pairs):
    result = transformation_folds(pairs, n_folds=3, seed=42)
    assert result.fold(0).fold == 0
    assert result.fold(result.folds[0].name) is result.folds[0]
    with pytest.raises(KeyError):
        result.fold("not a fold")


# --- leave one headgroup out -------------------------------------------------

def test_loose_headgroup_fold_trains_on_the_test_series(pairs):
    fold = headgroup_split(pairs, headgroups=["tetrazole"]).fold("tetrazole")
    trained_series = set(pairs.loc[fold.train, "constant_smiles"])
    tested_series = set(pairs.loc[fold.test, "constant_smiles"])
    assert "SHARED" in trained_series & tested_series


def test_strict_headgroup_fold_trains_on_no_test_series(pairs):
    result = headgroup_split(pairs, headgroups=["tetrazole"],
                             require_disjoint_series=True)
    fold = result.fold("tetrazole")
    trained_series = set(pairs.loc[fold.train, "constant_smiles"])
    tested_series = set(pairs.loc[fold.test, "constant_smiles"])
    assert not (trained_series & tested_series)
    assert fold.train_dropped_for_series > 0
    assert result.series_disjoint
    assert_headgroup_never_seen(pairs, result)


def test_strict_headgroup_fold_keeps_series_that_never_meet_the_test_set(pairs):
    """A_ONLY carries only acid -> acylsulfonamide and no tetrazole pair, so
    strictness must not touch it - otherwise the option is just deleting data."""
    fold = headgroup_split(pairs, headgroups=["tetrazole"],
                           require_disjoint_series=True).fold("tetrazole")
    assert set(pairs.loc[fold.train, "constant_smiles"]) == {"A_ONLY"}


def test_dropped_training_pairs_become_excluded_not_lost(pairs):
    """train/test/excluded must still cover every row exactly once, or the
    fold sizes stop adding up to the dataset."""
    for strict in (False, True):
        fold = headgroup_split(pairs, headgroups=["tetrazole"],
                               require_disjoint_series=strict).fold("tetrazole")
        covered = (fold.train.astype(int) + fold.test.astype(int)
                   + fold.excluded.astype(int))
        assert (covered == 1).all()


def test_strict_headgroup_fold_only_ever_shrinks_training(pairs):
    loose = headgroup_split(pairs, headgroups=["tetrazole"]).fold("tetrazole")
    strict = headgroup_split(pairs, headgroups=["tetrazole"],
                             require_disjoint_series=True).fold("tetrazole")
    assert strict.train.sum() < loose.train.sum()
    assert (strict.train & ~loose.train).sum() == 0   # no row gained
    assert strict.test.equals(loose.test)             # the test set is untouched


def test_the_assertion_catches_a_hand_broken_strict_fold(pairs):
    """The checker must actually fire, not pass because nothing reaches it."""
    result = headgroup_split(pairs, headgroups=["tetrazole"],
                             require_disjoint_series=True)
    fold = result.fold("tetrazole")
    first_test_row = pairs.index[fold.test][0]
    broken = fold.train.copy()
    broken.loc[first_test_row] = False
    # Put a pair from a tested series back into training, keeping the partition.
    familiar = pairs.index[(pairs["constant_smiles"] == "SHARED")
                           & ~fold.test & ~fold.train][0]
    broken.loc[familiar] = True
    fold.train = broken
    fold.excluded = ~fold.test & ~fold.train
    with pytest.raises(AssertionError, match="series are in both train and test"):
        assert_headgroup_never_seen(pairs, result)


# --- the overlap report ------------------------------------------------------

def test_fold_overlap_reports_the_loose_leak_and_the_strict_zero(pairs):
    loose = headgroup_split(pairs, headgroups=["tetrazole"]).fold("tetrazole")
    strict = headgroup_split(pairs, headgroups=["tetrazole"],
                             require_disjoint_series=True).fold("tetrazole")
    before = fold_overlap(pairs, loose.train, loose.test)
    after = fold_overlap(pairs, strict.train, strict.test)
    assert before["series in train"] > 0
    assert after["series in train"] == 0
    assert before["test pairs"] == after["test pairs"]


def test_fold_overlap_on_an_empty_training_set_does_not_divide_by_zero(pairs):
    empty = pd.Series(False, index=pairs.index)
    report = fold_overlap(pairs, empty, pd.Series(True, index=pairs.index))
    assert report["series in train"] == 0.0 and report["molecule in train"] == 0.0


# --- mirrored rows (both directions of one comparison) -----------------------
#
# mmpdb --symmetric stores A -> B and B -> A, and on the real joined data 100%
# of rows have their mirror present. Training on one and testing
# on the other is leakage: the label is just the negation.

def test_every_splitter_keeps_both_directions_in_one_fold(pairs):
    """The mirrors are added explicitly here, since the fixture is one-way."""
    mirrored = pairs.rename(columns={
        "parent_a_id": "parent_b_id", "parent_b_id": "parent_a_id",
        "smiles_a": "smiles_b", "smiles_b": "smiles_a",
        "headgroup_a": "headgroup_b", "headgroup_b": "headgroup_a"})
    both = pd.concat([pairs, mirrored], ignore_index=True)

    check_split(both, transformation_split(both, n_folds=3, seed=42))
    assert_mask_folds(both, transformation_folds(both, n_folds=3, seed=42))
    assert_mask_folds(both, transformation_folds(both, n_folds=3, seed=42,
                                                 require_disjoint_series=True))
    assert_headgroup_never_seen(both, headgroup_split(both, headgroups=["tetrazole"]))


def test_the_mirror_assertion_catches_a_split_that_separates_them():
    """Row 0 and row 1 are the same comparison written both ways. Putting one in
    training and the other in test must raise."""
    forward = pair("M1", ACID, "carboxylic_acid", "M2", TETRAZOLE, "tetrazole", "S")
    backward = pair("M2", TETRAZOLE, "tetrazole", "M1", ACID, "carboxylic_acid", "S")
    table = pd.DataFrame([forward, backward])
    train = pd.Series([True, False])
    test = pd.Series([False, True])
    with pytest.raises(AssertionError, match="mirror"):
        assert_mirrors_together(table, train, test)
    # The same two rows in one fold are fine.
    assert assert_mirrors_together(table, pd.Series([False, False]),
                                   pd.Series([True, True]))
