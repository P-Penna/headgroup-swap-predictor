"""Tests for app/swap_check.py: telling a real headgroup swap from a mislabel."""

from app.swap_check import GENUINE, NEITHER, NOT_ADDED, NOT_REMOVED, UNREADABLE, check_swap


def test_plain_acid_to_tetrazole_is_genuine():
    assert check_swap("OC(=O)c1ccccc1", "c1ccccc1-c1nnn[nH]1",
                      "carboxylic_acid", "tetrazole") == GENUINE


def test_diacid_with_one_acid_swapped_is_genuine():
    """B still has an acid, so a presence check would call this a mislabel.
    Counting sees 2 acids -> 1 acid, and 0 tetrazoles -> 1."""
    assert check_swap("OC(=O)c1ccc(C(=O)O)cc1", "OC(=O)c1ccc(-c2nnn[nH]2)cc1",
                      "carboxylic_acid", "tetrazole") == GENUINE


def test_unchanged_phosphonic_acid_is_not_a_swap():
    """The real-data pattern: both carry a phosphonic acid; what changed is an
    acid becoming a triazole. Labelled acid <-> phosphonic by a big cut."""
    a = "OC(=O)COCC(CO)(P(=O)(O)O)Oc1ccccc1"
    b = "c1ncn(n1)CC(CO)(P(=O)(O)O)Oc1ccccc1"
    assert check_swap(a, b, "carboxylic_acid", "phosphonic_acid") == NOT_ADDED


def test_group_kept_on_b_is_not_removed():
    """A: acid + hydroxamic. B: acid + hydroxamic + an extra hydroxamic.
    Labelled acid -> hydroxamic, but the acid was never taken away."""
    a = "OC(=O)CCc1ccc(C(=O)NO)cc1"
    b = "OC(=O)CCc1ccc(C(=O)NO)cc1C(=O)NO"
    assert check_swap(a, b, "carboxylic_acid", "hydroxamic_acid") == NOT_REMOVED


def test_nothing_changed_in_either_group():
    a = "OC(=O)Cc1ccc(-c2nnn[nH]2)cc1"
    b = "OC(=O)CCc1ccc(-c2nnn[nH]2)cc1"
    assert check_swap(a, b, "carboxylic_acid", "tetrazole") == NEITHER


def test_unreadable_smiles():
    assert check_swap("not a smiles", "CC", "carboxylic_acid", "tetrazole") == UNREADABLE
