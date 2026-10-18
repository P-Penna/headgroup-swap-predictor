"""Tests for the leave-one-headgroup-out split in app/splits.py.

The property that matters: in the fold for headgroup H, no training molecule
contains H - not as a label and not hidden in a molecule with two headgroups.
"""

import pandas as pd
import pytest

from app.splits import assert_headgroup_never_seen, headgroup_split

ACID = "OC(=O)c1ccc(C)cc1"
TETRAZOLE = "Cc1ccc(-c2nnn[nH]2)cc1"
ACYLSULF = "Cc1ccc(C(=O)NS(C)(=O)=O)cc1"
SULFONAMIDE = "Cc1ccc(S(N)(=O)=O)cc1"
# Two headgroups on one molecule: acid at one end, tetrazole at the other.
ACID_PLUS_TET = "OC(=O)Cc1ccc(-c2nnn[nH]2)cc1"
SULF_PLUS_TET = "NS(=O)(=O)Cc1ccc(-c2nnn[nH]2)cc1"


def pair(a, smiles_a, hg_a, b, smiles_b, hg_b):
    return {"parent_a_id": a, "smiles_a": smiles_a, "headgroup_a": hg_a,
            "parent_b_id": b, "smiles_b": smiles_b, "headgroup_b": hg_b}


@pytest.fixture
def pairs():
    return pd.DataFrame([
        pair("M1", ACID, "carboxylic_acid", "M2", TETRAZOLE, "tetrazole"),       # 0 tet
        pair("M1", ACID, "carboxylic_acid", "M3", ACYLSULF, "acylsulfonamide"),  # 1 acyl
        pair("M3", ACYLSULF, "acylsulfonamide", "M2", TETRAZOLE, "tetrazole"),   # 2 both
        pair("M1", ACID, "carboxylic_acid", "M4", SULFONAMIDE,
             "primary_sulfonamide"),                                             # 3 sulf
        # acid -> sulfonamide with a tetrazole sitting unchanged on both ends
        pair("M5", ACID_PLUS_TET, "carboxylic_acid", "M6", SULF_PLUS_TET,
             "primary_sulfonamide"),                                             # 4 hidden
    ])


def fold_for(result, headgroup):
    return next(fold for fold in result.folds if fold.headgroup == headgroup)


def test_test_set_is_every_pair_involving_the_headgroup(pairs):
    result = headgroup_split(pairs, headgroups=["tetrazole"])
    assert list(fold_for(result, "tetrazole").test) == [True, False, True, False, False]


def test_a_pair_between_two_held_out_groups_is_test_in_both_folds(pairs):
    result = headgroup_split(pairs, headgroups=["tetrazole", "acylsulfonamide"])
    assert fold_for(result, "tetrazole").test.iloc[2]
    assert fold_for(result, "acylsulfonamide").test.iloc[2]


def test_a_hidden_copy_keeps_a_pair_out_of_training(pairs):
    """Row 4 is labelled acid -> sulfonamide, but both molecules carry a
    tetrazole. Training on it would show the model a tetrazole, so in the
    tetrazole fold it is excluded - neither train nor test."""
    fold = fold_for(headgroup_split(pairs, headgroups=["tetrazole"]), "tetrazole")
    assert not fold.train.iloc[4] and not fold.test.iloc[4]
    assert fold.excluded.iloc[4]
    assert list(fold.train) == [False, True, False, True, False]


def test_the_hidden_pair_still_trains_other_folds(pairs):
    fold = fold_for(headgroup_split(pairs, headgroups=["acylsulfonamide"]),
                    "acylsulfonamide")
    assert fold.train.iloc[4]


def test_the_checker_passes_a_correct_split(pairs):
    result = headgroup_split(pairs, headgroups=["tetrazole", "acylsulfonamide",
                                                "primary_sulfonamide"])
    assert assert_headgroup_never_seen(pairs, result)


def test_the_checker_catches_a_hidden_copy_in_training(pairs):
    """Label-only exclusion would put row 4 in training; the check must fire."""
    result = headgroup_split(pairs, headgroups=["tetrazole"])
    fold = result.folds[0]
    fold.train.iloc[4] = True
    fold.excluded.iloc[4] = False
    with pytest.raises(AssertionError, match="present in training molecule"):
        assert_headgroup_never_seen(pairs, result)


def test_acid_can_never_be_held_out(pairs):
    with pytest.raises(ValueError, match="never be held out"):
        headgroup_split(pairs, headgroups=["carboxylic_acid"])


def test_headgroups_without_a_fold_are_listed_with_molecule_counts(pairs):
    result = headgroup_split(pairs, headgroups=["tetrazole"])
    assert result.not_held_out == {"primary_sulfonamide": 2, "acylsulfonamide": 1}
