"""Tests for app/splits.py, on a synthetic pair table.

No real data is needed: every property that makes a split trustworthy - no
group split across folds, no molecule in two folds, no pair spanning folds,
time running one way - is a property of the splitter, not of the dataset.

The synthetic table is built to the schema the real matched-pair table has
(the pairing step, not shipped here), plus the target and year
columns that step 1.5 adds when pairs are joined to the activity table.

The molecules are real, because the scaffold splitter calls RDKit and a fake
SMILES would simply fail to parse.
"""

import pandas as pd
import pytest

from app.splits import (
    DROPPED,
    PRIMARY_SERIES_SPLIT,
    SPLITTERS,
    assert_molecules_disjoint,
    assign_groups_to_folds,
    check_split,
    compare_series_splits,
    constant_split,
    molecule_table,
    murcko_scaffold,
    require_columns,
    scaffold_split,
    swap_class,
    target_split,
    time_split,
    transformation_split,
)

# Three chemical series. Within a series the molecules share a scaffold and
# differ only in the acidic headgroup, which is what a headgroup swap is.
BENZOIC = "c1ccccc1"
NAPHTHOIC = "c1ccc2ccccc2c1"
PYRIDINE = "c1ccncc1"

SERIES = {
    "S1": (BENZOIC, ["OC(=O)c1ccccc1", "c1ccccc1-c1nnn[nH]1", "CS(=O)(=O)NC(=O)c1ccccc1"]),
    "S2": (NAPHTHOIC, ["OC(=O)c1ccc2ccccc2c1", "c1ccc2ccccc2c1-c1nnn[nH]1"]),
    "S3": (PYRIDINE, ["OC(=O)c1ccncc1", "c1ccncc1-c1nnn[nH]1", "NS(=O)(=O)c1ccncc1"]),
}

ACID = "[*:1]C(=O)O"
TETRAZOLE = "[*:1]c1nnn[nH]1"
ACYLSULFONAMIDE = "[*:1]C(=O)NS(C)(=O)=O"
SULFONAMIDE = "[*:1]S(N)(=O)=O"
FRAGMENT_OF = {0: ACID, 1: TETRAZOLE, 2: ACYLSULFONAMIDE}
# The headgroup label each fragment carries, so the two columns never disagree.
HEADGROUP_OF = {ACID: "carboxylic_acid", TETRAZOLE: "tetrazole",
                ACYLSULFONAMIDE: "acylsulfonamide", SULFONAMIDE: "primary_sulfonamide"}


def build_pairs(symmetric=True, targets=("T1", "T2"), years=(2010, 2020)):
    """Every within-series pair, in both directions, across targets and years."""
    rows = []
    for series, (_scaffold, molecules) in SERIES.items():
        ids = [f"{series}_M{i}" for i in range(len(molecules))]
        for i in range(len(molecules)):
            for j in range(len(molecules)):
                if i == j or (not symmetric and j < i):
                    continue
                for target in targets:
                    for year in years:
                        frag_a = FRAGMENT_OF.get(i, SULFONAMIDE)
                        frag_b = FRAGMENT_OF.get(j, SULFONAMIDE)
                        rows.append({
                            "parent_a_id": ids[i], "smiles_a": molecules[i],
                            "parent_b_id": ids[j], "smiles_b": molecules[j],
                            "headgroup_a": HEADGROUP_OF[frag_a],
                            "headgroup_b": HEADGROUP_OF[frag_b],
                            "sub_form_a": "", "sub_form_b": "",
                            "pair_class": "bioisostere",
                            "n_heavies_a": 4, "n_heavies_b": 6,
                            "constant_smiles": f"[*:1]{series}",
                            "transformation": f"{frag_a}>>{frag_b}",
                            "env_radius2_smarts": "[#0;X1:1]-[c]",
                            "env_radius2_pseudosmiles": "[*:1]-c",
                            "target_chembl_id": target,
                            "year": year,
                        })
    return pd.DataFrame(rows)


@pytest.fixture
def pairs():
    return build_pairs()


# --- the column contract ----------------------------------------------------

def test_each_splitter_validates_its_own_columns(pairs):
    for kind in SPLITTERS:
        require_columns(pairs, kind)          # the full table satisfies all four


def test_target_split_explains_the_missing_column(pairs):
    """The raw matched pairs have no target, so the error must say where it comes from."""
    without = pairs.drop(columns=["target_chembl_id"])
    with pytest.raises(ValueError, match="pairing step"):
        require_columns(without, "target")


def test_time_split_explains_the_missing_column(pairs):
    with pytest.raises(ValueError, match="pairing step"):
        require_columns(pairs.drop(columns=["year"]), "time")


def test_series_and_transformation_splits_work_on_the_mmps_schema_alone(pairs):
    """These must not need the 1.5 activity join."""
    mmps_only = pairs.drop(columns=["target_chembl_id", "year"])
    for kind in ("constant", "scaffold", "transformation"):
        require_columns(mmps_only, kind)
        assert SPLITTERS[kind](mmps_only, n_folds=2).kept().any()


# --- constant split: the primary series split ------------------

def test_constant_is_the_primary_series_split():
    assert PRIMARY_SERIES_SPLIT == "constant"
    assert "constant" in SPLITTERS and "scaffold" in SPLITTERS


def test_constant_split_drops_no_pairs(pairs):
    """Both molecules share the constant by construction, so nothing is lost
    for the two ends disagreeing - the scaffold split's main weakness."""
    result = constant_split(pairs, n_folds=2)
    assert int(result.kept().sum()) == len(pairs)
    assert int((~result.kept()).sum()) == 0
    check_split(pairs, result)


def test_constant_split_keeps_ring_headgroup_swaps(pairs):
    """Acid -> tetrazole changes the Murcko scaffold but NOT the constant, so
    the primary split keeps it where the secondary one drops it."""
    acid_to_tetrazole = pairs["transformation"] == f"{ACID}>>{TETRAZOLE}"
    assert acid_to_tetrazole.any()
    primary = constant_split(pairs, n_folds=2)
    secondary = scaffold_split(pairs, n_folds=2)
    assert primary.kept()[acid_to_tetrazole].all()
    assert not secondary.kept()[acid_to_tetrazole].any()


def test_constant_split_never_splits_a_series(pairs):
    result = constant_split(pairs, n_folds=2)
    frame = pd.DataFrame({"c": pairs["constant_smiles"], "f": result.folds})
    frame = frame[frame["f"] != DROPPED]
    assert (frame.groupby("c")["f"].nunique() == 1).all()


def test_constant_split_can_be_made_molecule_disjoint(pairs):
    strict = constant_split(pairs, n_folds=2, require_disjoint_molecules=True)
    assert strict.molecules_disjoint is True
    assert strict.stats["molecules shared between folds"] == 0
    check_split(pairs, strict)


def test_constant_split_drops_rows_with_no_constant(pairs):
    broken = pairs.copy()
    broken.loc[broken.index[:3], "constant_smiles"] = None
    result = constant_split(broken, n_folds=2)
    assert result.stats["pairs with no constant"] == 3
    assert (result.folds.iloc[:3] == DROPPED).all()


def test_compare_series_splits_exposes_the_scaffold_bias(pairs):
    """The disagreement must be reported, and dominated by ring-headgroup swaps."""
    primary, secondary, report = compare_series_splits(pairs, n_folds=2)
    assert primary.kind == "constant" and secondary.kind == "scaffold"
    assert report["constant kept"] == len(pairs)
    assert report["scaffold kept"] < report["constant kept"]
    assert report["kept by constant but dropped by scaffold"] > 0
    assert 0 < report["share of constant-kept pairs lost by scaffold"] <= 1
    lost = report["transformations scaffold loses most"]
    assert lost, "expected some transformations to be listed"
    # every transformation the scaffold split loses here involves the tetrazole ring
    assert all("nnn" in transformation for transformation in lost)


def test_unknown_kind_is_rejected(pairs):
    with pytest.raises(ValueError, match="unknown split kind"):
        require_columns(pairs, "random")


# --- the molecule table -----------------------------------------------------

def test_molecule_table_collects_both_sides(pairs):
    molecules = molecule_table(pairs)
    expected = sum(len(m) for _s, m in SERIES.values())
    assert len(molecules) == expected


def test_inconsistent_smiles_for_one_id_is_an_error(pairs):
    broken = pairs.copy()
    broken.loc[broken.index[0], "smiles_a"] = "CCCCCCO"
    with pytest.raises(ValueError, match="more than one"):
        molecule_table(broken)


# --- group-to-fold assignment ----------------------------------------------

def test_assignment_balances_folds():
    sizes = pd.Series({"a": 100, "b": 90, "c": 10, "d": 5, "e": 5})
    folds = assign_groups_to_folds(sizes, n_folds=2, seed=42)
    load = [0, 0]
    for group, fold in folds.items():
        load[fold] += sizes[group]
    assert abs(load[0] - load[1]) <= 20       # 105 vs 105 in the ideal case


def test_assignment_is_deterministic_for_a_seed():
    sizes = pd.Series({name: 1 for name in "abcdefghij"})
    first = assign_groups_to_folds(sizes, 3, seed=42)
    assert first == assign_groups_to_folds(sizes, 3, seed=42)
    assert first != assign_groups_to_folds(sizes, 3, seed=7)


def test_one_fold_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        assign_groups_to_folds(pd.Series({"a": 1}), n_folds=1)


# --- scaffold split ---------------------------------------------------------

def test_murcko_scaffold_strips_the_headgroup():
    """Two analogues differing only in a non-ring headgroup share a scaffold."""
    assert murcko_scaffold("OC(=O)c1ccccc1") == murcko_scaffold("CS(=O)(=O)NC(=O)c1ccccc1")


def test_murcko_scaffold_changes_when_the_headgroup_is_a_ring():
    """A tetrazole IS a ring, so swapping it changes the scaffold. This is why
    the scaffold split drops ring-headgroup swaps."""
    assert murcko_scaffold("OC(=O)c1ccccc1") != murcko_scaffold("c1ccccc1-c1nnn[nH]1")


def test_murcko_scaffold_handles_unparseable_smiles():
    assert murcko_scaffold("not a molecule") is None
    assert murcko_scaffold(None) is None


def test_scaffold_split_keeps_no_molecule_in_two_folds(pairs):
    result = scaffold_split(pairs, n_folds=2)
    check_split(pairs, result)                 # raises if a molecule leaks


def test_scaffold_split_drops_ring_headgroup_pairs(pairs):
    """Acid -> tetrazole changes the scaffold, so those pairs cannot be placed."""
    result = scaffold_split(pairs, n_folds=2)
    assert result.stats["pairs whose two molecules have different scaffolds"] > 0
    dropped = pairs[result.folds == DROPPED]
    assert (dropped["transformation"].str.contains("nnn")).any()


def test_scaffold_split_keeps_same_scaffold_pairs(pairs):
    """Acid -> acylsulfonamide keeps the scaffold, so it must survive."""
    result = scaffold_split(pairs, n_folds=2)
    kept = pairs[result.kept()]
    assert len(kept) > 0
    assert (kept["transformation"] == f"{ACID}>>{ACYLSULFONAMIDE}").any()


def test_scaffold_split_reports_unparseable_molecules(pairs):
    """One molecule's SMILES is broken everywhere it appears, on both sides of
    the pair - otherwise molecule_table rightly complains about two structures
    for one id."""
    broken = pairs.copy()
    broken.loc[broken["parent_a_id"] == "S1_M0", "smiles_a"] = "not a molecule"
    broken.loc[broken["parent_b_id"] == "S1_M0", "smiles_b"] = "not a molecule"
    result = scaffold_split(broken, n_folds=2)
    assert result.stats["molecules whose SMILES would not parse"] == 1


# --- target split -----------------------------------------------------------

def test_target_split_never_splits_a_target(pairs):
    result = target_split(pairs, n_folds=2)
    check_split(pairs, result)
    per_target = pd.DataFrame({"t": pairs["target_chembl_id"], "f": result.folds})
    per_target = per_target[per_target["f"] != DROPPED]
    assert (per_target.groupby("t")["f"].nunique() == 1).all()


def test_target_split_shares_molecules_by_design(pairs):
    """Documented behaviour: the same compound may appear on both sides."""
    result = target_split(pairs, n_folds=2)
    assert result.stats["molecules shared between folds"] > 0


def test_target_split_can_be_made_molecule_disjoint(pairs):
    strict = target_split(pairs, n_folds=2, require_disjoint_molecules=True)
    assert strict.stats["molecules shared between folds"] == 0
    assert strict.stats["pairs dropped to keep molecules disjoint"] > 0
    loose = target_split(pairs, n_folds=2)
    assert int(strict.kept().sum()) < int(loose.kept().sum())


# --- transformation split ---------------------------------------------------

def swap_keys(pairs):
    return pd.Series([swap_class(a, b) for a, b in
                      zip(pairs["headgroup_a"], pairs["headgroup_b"])], index=pairs.index)


def test_swap_class_merges_both_directions():
    assert (swap_class("carboxylic_acid", "tetrazole")
            == swap_class("tetrazole", "carboxylic_acid")
            == ("carboxylic_acid", "tetrazole"))


def test_swap_class_names_a_missing_headgroup():
    """deletion / to_non_acid pairs have a blank side; it gets a name."""
    assert swap_class("carboxylic_acid", "") == ("carboxylic_acid", "none")
    assert swap_class(float("nan"), "tetrazole") == ("none", "tetrazole")
    assert swap_class("", None) is None


def test_transformation_split_ignores_the_fragment_string():
    """The leak this key fixes. mmpdb writes the same acid -> tetrazole swap
    as different strings depending on where the cut fell. Keyed on the string
    they could land in different folds - the 'unseen' swap would have been
    trained on. Keyed on the headgroup pair they cannot."""
    short_cut = "[*:1]C(=O)O>>[*:1]c1nnn[nH]1"
    long_cut = "[*:1]c1ccc(C(=O)O)cc1>>[*:1]c1ccc(-c2nnn[nH]2)cc1"
    rows = []
    for i in range(6):
        rows.append({"parent_a_id": f"A{i}", "parent_b_id": f"B{i}",
                     "headgroup_a": "carboxylic_acid", "headgroup_b": "tetrazole",
                     "transformation": short_cut if i % 2 else long_cut})
    for i in range(6):
        rows.append({"parent_a_id": f"C{i}", "parent_b_id": f"D{i}",
                     "headgroup_a": "carboxylic_acid", "headgroup_b": "phosphonic_acid",
                     "transformation": "[*:1]C(=O)O>>[*:1]P(=O)(O)O"})
    pairs = pd.DataFrame(rows)
    # 3 folds, because with 3 the old string key DOES put the two strings in
    # different folds (checked), so this test fails on the bug it guards.
    result = transformation_split(pairs, n_folds=3)
    tetrazole_folds = set(result.folds[pairs["headgroup_b"] == "tetrazole"])
    assert len(tetrazole_folds) == 1
    assert result.stats["distinct swap classes (headgroup pairs, directions merged)"] == 2


def test_transformation_split_puts_both_directions_in_one_fold(pairs):
    """--symmetric stores A>>B and B>>A; if they split, the test set contains
    the training swap written backwards."""
    result = transformation_split(pairs, n_folds=2)
    check_split(pairs, result)
    frame = pd.DataFrame({"key": swap_keys(pairs), "fold": result.folds})
    frame = frame[frame["fold"] != DROPPED]
    assert (frame.groupby("key")["fold"].nunique() == 1).all()


def test_transformation_split_holds_out_whole_swap_classes(pairs):
    result = transformation_split(pairs, n_folds=2)
    kept = result.folds != DROPPED
    keys = swap_keys(pairs)[kept]
    folds = result.folds[kept]
    train_keys = set(keys[folds == 0])
    test_keys = set(keys[folds == 1])
    assert train_keys and test_keys
    assert not (train_keys & test_keys)


def test_transformation_split_shares_molecules_by_design(pairs):
    """One acid takes part in several transformations, so holding out a
    transformation cannot also hold out its molecules. This is documented
    behaviour, not a leak the checker should reject."""
    result = transformation_split(pairs, n_folds=2)
    assert result.stats["molecules shared between folds"] > 0
    assert result.molecules_disjoint is False
    assert_molecules_disjoint(pairs, result)        # must be a no-op, not a failure


def test_transformation_split_can_be_made_molecule_disjoint(pairs):
    strict = transformation_split(pairs, n_folds=2, require_disjoint_molecules=True)
    assert strict.molecules_disjoint is True
    assert strict.stats["molecules shared between folds"] == 0
    check_split(pairs, strict)
    loose = transformation_split(pairs, n_folds=2)
    assert int(strict.kept().sum()) < int(loose.kept().sum())


def test_only_the_scaffold_split_promises_disjoint_molecules(pairs):
    assert scaffold_split(pairs, n_folds=2).molecules_disjoint is True
    for kind in ("target", "transformation"):
        assert SPLITTERS[kind](pairs, n_folds=2).molecules_disjoint is False
    assert time_split(pairs, cutoff_year=2015).molecules_disjoint is False


def test_transformation_split_drops_pairs_with_no_headgroup(pairs):
    broken = pairs.copy()
    broken.loc[broken.index[0], ["headgroup_a", "headgroup_b"]] = ""
    result = transformation_split(broken, n_folds=2)
    assert result.stats["pairs with no headgroup on either side (dropped)"] == 1
    assert result.folds.iloc[0] == DROPPED


# --- time split -------------------------------------------------------------

def test_time_split_runs_one_way(pairs):
    result = time_split(pairs, cutoff_year=2015)
    check_split(pairs, result)
    assert result.n_folds == 2
    assert pairs.loc[result.folds == 0, "year"].max() <= 2015
    assert pairs.loc[result.folds == 1, "year"].min() > 2015


def test_time_split_drops_missing_years(pairs):
    with_gaps = pairs.copy()
    with_gaps.loc[with_gaps.index[:4], "year"] = None
    result = time_split(with_gaps, cutoff_year=2015)
    assert result.stats["pairs with no year (dropped)"] == 4
    assert (result.folds.iloc[:4] == DROPPED).all()


def test_time_split_catches_an_out_of_order_fold(pairs):
    """The order check must actually fire, not just pass vacuously."""
    result = time_split(pairs, cutoff_year=2015)
    result.folds.iloc[:] = 1                   # claim everything is test
    with pytest.raises(AssertionError, match="at or before the cutoff"):
        check_split(pairs, result)


# --- the checks themselves must be able to fail -----------------------------

def test_leakage_check_catches_a_split_group(pairs):
    result = transformation_split(pairs, n_folds=2)
    moved = result.folds.copy()
    first_fold = moved[moved != DROPPED].iloc[0]
    # Move a single pair to another fold, splitting its swap class.
    target_row = moved[moved == first_fold].index[0]
    moved.loc[target_row] = 1 - first_fold
    result.folds = moved
    with pytest.raises(AssertionError, match="span more than one fold"):
        check_split(pairs, result)


def test_leakage_check_catches_a_molecule_in_two_folds():
    """A random split over pairs is exactly what rule 1 forbids; the check
    must reject it."""
    pairs = build_pairs()
    result = scaffold_split(pairs, n_folds=2)    # promises molecules_disjoint
    # Replace the honest split with a random pair-level one.
    rng = pd.Series(range(len(pairs))).mod(2)
    result.folds = pd.Series(rng.values, index=pairs.index)
    result.groups = pd.Series([None] * len(pairs), index=pairs.index)
    with pytest.raises(AssertionError, match="appear in fold"):
        check_split(pairs, result)


# --- shared behaviour across all four --------------------------------------

@pytest.mark.parametrize("kind", sorted(SPLITTERS))
def test_every_splitter_is_deterministic(pairs, kind):
    kwargs = {"cutoff_year": 2015} if kind == "time" else {"n_folds": 2}
    first = SPLITTERS[kind](pairs, **kwargs)
    second = SPLITTERS[kind](pairs, **kwargs)
    assert first.folds.tolist() == second.folds.tolist()


@pytest.mark.parametrize("kind", sorted(SPLITTERS))
def test_every_splitter_accounts_for_every_row(pairs, kind):
    """Rule 5: nothing vanishes. Every input row gets a fold or DROPPED."""
    kwargs = {"cutoff_year": 2015} if kind == "time" else {"n_folds": 2}
    result = SPLITTERS[kind](pairs, **kwargs)
    assert len(result.folds) == len(pairs)
    assert int(result.kept().sum()) + int((~result.kept()).sum()) == len(pairs)
    assert "pairs kept" in result.summary()


@pytest.mark.parametrize("kind", sorted(SPLITTERS))
def test_every_splitter_passes_its_own_checks(pairs, kind):
    kwargs = {"cutoff_year": 2015} if kind == "time" else {"n_folds": 3}
    assert check_split(pairs, SPLITTERS[kind](pairs, **kwargs))


def test_empty_table_does_not_crash():
    empty = build_pairs().iloc[0:0]
    for kind in sorted(SPLITTERS):
        kwargs = {"cutoff_year": 2015} if kind == "time" else {"n_folds": 2}
        result = SPLITTERS[kind](empty, **kwargs)
        assert len(result.folds) == 0
