"""Tests for app/headgroup_properties.py and the two baselines built on it.

The one that matters most is
test_every_capped_fragment_contains_its_own_headgroup: a capped SMILES that is
quietly the wrong molecule would give the baseline plausible numbers for the
wrong chemistry, and nothing downstream would notice. It is checked against the
SMARTS in app/headgroups.py, which is the project's own definition of each
group, rather than against my reading of the name.
"""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, RDLogger

from app.baselines import (
    DELTA, HeadgroupPropertyBaseline, NearestSwapClassMedianBaseline,
)
from app.headgroup_properties import (
    CAPPED_SMILES, CAP_CARBONS, PKA, PKA_UNCERTAIN, PROPERTY_NAMES,
    difference_vector, missing_headgroups, properties, property_vector,
)
from app.headgroups import HEADGROUPS, count_headgroup

RDLogger.DisableLog("rdApp.*")


# --- the chemistry -----------------------------------------------------------

def test_every_headgroup_in_the_project_has_properties():
    """A group added to app/headgroups.py without properties here would make
    the baseline silently fall back to predicting zero for every pair using it."""
    assert missing_headgroups() == []


@pytest.mark.parametrize("key", sorted(CAPPED_SMILES))
def test_every_capped_fragment_parses(key):
    assert Chem.MolFromSmiles(CAPPED_SMILES[key]) is not None


@pytest.mark.parametrize("key", sorted(CAPPED_SMILES))
def test_every_capped_fragment_contains_its_own_headgroup(key):
    """THE test of this module. The fragment must match the SMARTS the project
    uses to find that headgroup in a real molecule - exactly once."""
    molecule = Chem.MolFromSmiles(CAPPED_SMILES[key])
    assert count_headgroup(molecule, key) == 1, \
        f"{CAPPED_SMILES[key]} is not one {key}"


@pytest.mark.parametrize("key", sorted(CAPPED_SMILES))
def test_a_capped_fragment_is_not_some_other_headgroup(key):
    """A fragment should not read as a different acidic group as well. The
    known and acceptable overlaps are listed, so a NEW overlap fails here."""
    allowed = {
        # A cyclic acylsulfonamide-like or diketone-like group can legitimately
        # contain a smaller pattern; these are the ones that actually overlap.
        "thiazolidine_2_4_dione": {"thiazolidine_2_4_dione"},
        "tetramic_acid": {"tetramic_acid"},
        "cyclopentane_1_3_dione": {"cyclopentane_1_3_dione"},
    }
    molecule = Chem.MolFromSmiles(CAPPED_SMILES[key])
    found = {other for other in HEADGROUPS
             if count_headgroup(molecule, other) > 0}
    assert found == allowed.get(key, {key}), f"{key} also reads as {found - {key}}"


def test_every_pka_is_a_plausible_number():
    for key, value in PKA.items():
        assert -2.0 <= value <= 14.0, f"{key} pKa {value} is off the scale"


def test_the_uncertain_pkas_are_the_ones_the_headgroup_table_doubts():
    """PKA_UNCERTAIN must track the notes in app/headgroups.py, or it stops
    being a usable list of which numbers to distrust."""
    for key in PKA_UNCERTAIN:
        note = HEADGROUPS[key].pka.lower()
        assert "uncertain" in note or "poorly characterized" in note


def test_heavy_atoms_excludes_the_cap():
    """Acetic acid is 4 heavy atoms, so the carboxylic acid itself is 3:
    C(=O)OH. Dimethylphosphinic acid is 5 and carries TWO caps, because two
    carbons on phosphorus are part of its definition, so the group is also 3:
    P(=O)OH. Without the correction it would read as 5 and look twice the size
    of the group it replaces."""
    assert properties("carboxylic_acid")["heavy_atoms"] == 3.0
    assert CAP_CARBONS["phosphinic_acid"] == 2
    assert properties("phosphinic_acid")["heavy_atoms"] == 3.0


def test_properties_are_in_a_stable_order_and_complete():
    vector = property_vector("tetrazole")
    assert len(vector) == len(PROPERTY_NAMES)
    assert all(isinstance(value, float) for value in vector)


def test_an_absent_headgroup_has_no_properties():
    assert properties("none") is None
    assert property_vector("none") is None
    assert difference_vector("carboxylic_acid", "none") is None
    assert difference_vector("none", "none") is None


def test_the_difference_vector_is_antisymmetric():
    forward = difference_vector("carboxylic_acid", "tetrazole")
    backward = difference_vector("tetrazole", "carboxylic_acid")
    assert forward == [-value for value in backward]


def test_swapping_a_group_for_itself_is_all_zeros():
    assert difference_vector("tetrazole", "tetrazole") == [0.0] * len(PROPERTY_NAMES)


def test_the_acid_to_tetrazole_swap_has_the_chemistry_right():
    """A sanity check with known signs: a tetrazole is bigger than a carboxylic
    acid, slightly less acidic, and has a similar donor count."""
    values = dict(zip(PROPERTY_NAMES,
                      difference_vector("carboxylic_acid", "tetrazole")))
    assert values["heavy_atoms"] > 0        # 5-ring is bigger than COOH
    assert values["pka"] > 0                # 4.7 against 4.5
    assert values["hba"] > 0                # the ring nitrogens accept


# --- the baselines -----------------------------------------------------------

def pair(hg_a, hg_b, delta, environment="[c]"):
    return {"headgroup_a": hg_a, "headgroup_b": hg_b, DELTA: delta,
            "env_radius2_smarts": environment}


@pytest.fixture
def train():
    """Swaps whose delta is built to depend on the pKa difference, so a model
    keyed on properties has something real to find."""
    rows = []
    for index in range(12):
        rows.append(pair("carboxylic_acid", "tetrazole", 0.20 + 0.01 * index))
        rows.append(pair("tetrazole", "carboxylic_acid", -0.20 - 0.01 * index))
        rows.append(pair("carboxylic_acid", "hydroxamic_acid", 1.00 + 0.01 * index))
        rows.append(pair("hydroxamic_acid", "carboxylic_acid", -1.00 - 0.01 * index))
    return pd.DataFrame(rows)


def test_the_property_baseline_predicts_an_unseen_swap_class(train):
    """THE reason this baseline exists. acid -> primary_sulfonamide is not in
    the training data at all, and the prediction must still be a real number
    from the model rather than a fallback to zero."""
    model = HeadgroupPropertyBaseline().fit(train)
    unseen = pd.DataFrame([pair("carboxylic_acid", "primary_sulfonamide", 0.0)])
    assert model.coverage(unseen) == {"headgroup_properties": 1, "global": 0}
    assert model.describe_collapse(unseen) is None
    assert model.predict(unseen)[0] != 0.0


def test_the_property_baseline_is_antisymmetric(train):
    """Fitted without an intercept on difference features, so swapping the two
    headgroups must negate the prediction exactly - the symmetry delta has."""
    model = HeadgroupPropertyBaseline().fit(train)
    forward = pd.DataFrame([pair("carboxylic_acid", "primary_sulfonamide", 0.0)])
    backward = pd.DataFrame([pair("primary_sulfonamide", "carboxylic_acid", 0.0)])
    assert model.predict(forward)[0] == pytest.approx(-model.predict(backward)[0])


def test_the_property_baseline_learns_the_sign_of_a_built_in_effect(train):
    """In the fixture, swapping an acid for a hydroxamic acid (pKa 4.5 -> 9.0)
    gains about 1 log unit and for a tetrazole (4.5 -> 4.7) gains about 0.2. A
    model on these features should put the hydroxamic swap higher."""
    model = HeadgroupPropertyBaseline().fit(train)
    test = pd.DataFrame([pair("carboxylic_acid", "tetrazole", 0.0),
                         pair("carboxylic_acid", "hydroxamic_acid", 0.0)])
    tetrazole, hydroxamic = model.predict(test)
    assert hydroxamic > tetrazole > 0


def test_the_property_baseline_falls_back_when_a_headgroup_is_absent(train):
    model = HeadgroupPropertyBaseline().fit(train)
    test = pd.DataFrame([pair("carboxylic_acid", "", 0.0)])
    assert model.coverage(test) == {"headgroup_properties": 0, "global": 1}
    assert model.predict(test)[0] == 0.0


def test_the_property_baseline_survives_an_untrainable_training_set():
    model = HeadgroupPropertyBaseline().fit(
        pd.DataFrame([pair("carboxylic_acid", "", 0.0)]))
    assert model.model is None
    test = pd.DataFrame([pair("carboxylic_acid", "tetrazole", 0.0)])
    assert list(model.predict(test)) == [0.0]


def test_the_property_baseline_reports_readable_coefficients(train):
    coefficients = HeadgroupPropertyBaseline().fit(train).coefficients()
    assert set(coefficients) == set(PROPERTY_NAMES)
    assert all(np.isfinite(value) for value in coefficients.values())


def test_nearest_swap_class_borrows_a_median_instead_of_the_global_one(train):
    """acid -> sulfonic is unseen. The nearest seen class by properties must be
    used, not the global median - which in this fixture is about 0."""
    model = NearestSwapClassMedianBaseline().fit(train)
    unseen = pd.DataFrame([pair("carboxylic_acid", "primary_sulfonamide", 0.0)])
    assert model.coverage(unseen) == {"transformation": 0,
                                      "nearest_swap_class": 1, "global": 0}
    prediction = model.predict(unseen)[0]
    assert prediction in set(model.medians.values())
    assert prediction != 0.0


def test_nearest_swap_class_reduces_to_the_transformation_median_when_seen(train):
    model = NearestSwapClassMedianBaseline().fit(train)
    seen = pd.DataFrame([pair("carboxylic_acid", "tetrazole", 0.0)])
    assert model.coverage(seen)["transformation"] == 1
    assert model.predict(seen)[0] == pytest.approx(
        model.medians[("carboxylic_acid", "tetrazole")])


def test_nearest_swap_class_keeps_the_direction_right(train):
    """The difference vector is signed, so the backward swap must borrow a
    negative median, not the forward one."""
    model = NearestSwapClassMedianBaseline().fit(train)
    forward = pd.DataFrame([pair("carboxylic_acid", "primary_sulfonamide", 0.0)])
    backward = pd.DataFrame([pair("primary_sulfonamide", "carboxylic_acid", 0.0)])
    assert model.predict(forward)[0] > 0 > model.predict(backward)[0]


# --- the environment-crossed baseline ----------------------------------------

def crossed_pair(hg_a, hg_b, delta, environment, a="A", b="B", constant="C"):
    return {"headgroup_a": hg_a, "headgroup_b": hg_b, DELTA: delta,
            "env_radius2_smarts": environment, "parent_a_id": a,
            "parent_b_id": b, "constant_smiles": constant}


@pytest.fixture
def crossed_train():
    """The same swap behaves differently in two environments, which is the
    within-class variation the plain property ridge cannot see at all."""
    rows = []
    for index in range(30):
        # In an aromatic environment the swap gains; in an aliphatic one it loses.
        rows.append(crossed_pair("carboxylic_acid", "tetrazole", 1.0,
                                 "[c]", a=f"ar{index}", b=f"at{index}",
                                 constant=f"S{index}"))
        rows.append(crossed_pair("tetrazole", "carboxylic_acid", -1.0,
                                 "[c]", a=f"at{index}", b=f"ar{index}",
                                 constant=f"S{index}"))
        rows.append(crossed_pair("carboxylic_acid", "tetrazole", -1.0,
                                 "[C]", a=f"br{index}", b=f"bt{index}",
                                 constant=f"T{index}"))
        rows.append(crossed_pair("tetrazole", "carboxylic_acid", 1.0,
                                 "[C]", a=f"bt{index}", b=f"br{index}",
                                 constant=f"T{index}"))
    return pd.DataFrame(rows)


def test_the_crossed_baseline_separates_two_environments(crossed_train):
    """THE reason it exists. One swap class, opposite deltas in two
    environments: the plain property ridge must average them to nothing, and
    the crossed one must recover both signs."""
    from app.baselines import EnvironmentPropertyBaseline
    plain = HeadgroupPropertyBaseline().fit(crossed_train)
    crossed = EnvironmentPropertyBaseline(min_count=20).fit(crossed_train)
    test = pd.DataFrame([
        crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[c]"),
        crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[C]")])
    aromatic_plain, aliphatic_plain = plain.predict(test)
    aromatic, aliphatic = crossed.predict(test)
    assert aromatic_plain == pytest.approx(aliphatic_plain)   # cannot tell them apart
    assert aromatic > 0 > aliphatic                           # the crossed one can
    assert crossed.environments_fitted() == 2


def test_the_crossed_baseline_is_antisymmetric(crossed_train):
    from app.baselines import EnvironmentPropertyBaseline
    model = EnvironmentPropertyBaseline(min_count=20).fit(crossed_train)
    forward = pd.DataFrame([crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[c]")])
    backward = pd.DataFrame([crossed_pair("tetrazole", "carboxylic_acid", 0.0, "[c]")])
    assert model.predict(forward)[0] == pytest.approx(-model.predict(backward)[0])


def test_the_crossed_baseline_falls_back_on_a_thin_environment(crossed_train):
    """An environment with fewer than min_count rows gets no model of its own
    and must use the global property ridge, reported at the transformation
    level rather than pretending to be environment-specific."""
    from app.baselines import EnvironmentPropertyBaseline
    thin = pd.concat([crossed_train, pd.DataFrame([
        crossed_pair("carboxylic_acid", "tetrazole", 5.0, "[N]", a="x", b="y",
                     constant="U")])], ignore_index=True)
    model = EnvironmentPropertyBaseline(min_count=20).fit(thin)
    test = pd.DataFrame([crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[N]")])
    assert model.coverage(test)["transformation"] == 1
    assert model.coverage(test)["transformation_environment"] == 0


def test_the_crossed_baseline_reports_its_coverage_levels(crossed_train):
    from app.baselines import EnvironmentPropertyBaseline
    model = EnvironmentPropertyBaseline(min_count=20).fit(crossed_train)
    test = pd.DataFrame([
        crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[c]"),   # fitted
        crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[N]"),   # unseen env
        crossed_pair("carboxylic_acid", "", 0.0, "[c]")])           # no properties
    assert model.coverage(test) == {"transformation_environment": 1,
                                    "transformation": 1, "global": 1}


# --- the shrinkage baseline -----------------------------------

def shrink_rows(n_big, n_small, big_delta, small_delta):
    """One well-measured cell and one thin cell inside the same swap class."""
    rows = []
    for index in range(n_big):
        rows.append(crossed_pair("carboxylic_acid", "tetrazole", big_delta, "[c]",
                                 a=f"p{index}", b=f"q{index}", constant=f"S{index}"))
    for index in range(n_small):
        rows.append(crossed_pair("carboxylic_acid", "tetrazole", small_delta, "[C]",
                                 a=f"r{index}", b=f"s{index}", constant=f"T{index}"))
    return pd.DataFrame(rows)


def test_a_thin_cell_is_pulled_harder_than_a_well_measured_one():
    """THE property of partial pooling. Both cells differ from the class mean;
    the one with less data must end up closer to it."""
    from app.baselines import ShrunkEnvironmentMeanBaseline
    # 60 rows at +1 and 3 rows at -3, so both cells are far from the class mean.
    rows = pd.concat([shrink_rows(60, 3, 1.0, -3.0),
                      # a little spread inside each cell so sigma_within > 0
                      pd.DataFrame([
                          crossed_pair("carboxylic_acid", "tetrazole", 1.4, "[c]",
                                       a="x", b="y", constant="X"),
                          crossed_pair("carboxylic_acid", "tetrazole", -2.6, "[C]",
                                       a="z", b="w", constant="Z")])],
                     ignore_index=True)
    model = ShrunkEnvironmentMeanBaseline(strength=1.0).fit(rows)
    swap = ("carboxylic_acid", "tetrazole")
    class_mean = model.group[swap]
    big = model.cell[(swap, "[c]")]
    small = model.cell[(swap, "[C]")]
    big_pull = abs(big - class_mean)
    small_pull = abs(small - class_mean)
    # Both are pulled toward the class mean, the thin one proportionally more.
    raw_big, raw_small = 1.0, -3.0
    assert big_pull / abs(raw_big - class_mean) > \
        small_pull / abs(raw_small - class_mean)


def test_zero_strength_is_the_raw_cell_mean():
    """strength -> 0 must reproduce partitioning exactly, which is the endpoint
    the sweep needs in order to mean anything."""
    from app.baselines import ShrunkEnvironmentMeanBaseline
    rows = shrink_rows(6, 6, 1.0, -1.0)
    model = ShrunkEnvironmentMeanBaseline(strength=0.0).fit(rows)
    swap = ("carboxylic_acid", "tetrazole")
    assert model.cell[(swap, "[c]")] == pytest.approx(1.0)
    assert model.cell[(swap, "[C]")] == pytest.approx(-1.0)


def test_infinite_strength_is_the_class_mean_everywhere():
    from app.baselines import ShrunkEnvironmentMeanBaseline
    rows = shrink_rows(6, 6, 1.0, -1.0)
    model = ShrunkEnvironmentMeanBaseline(strength=float("inf")).fit(rows)
    swap = ("carboxylic_acid", "tetrazole")
    for environment in ("[c]", "[C]"):
        assert model.cell[(swap, environment)] == pytest.approx(model.group[swap])


def test_shrinkage_is_antisymmetric():
    from app.baselines import ShrunkEnvironmentMeanBaseline
    rows = shrink_rows(20, 5, 1.0, -2.0)
    mirrored = rows.rename(columns={
        "headgroup_a": "headgroup_b", "headgroup_b": "headgroup_a",
        "parent_a_id": "parent_b_id", "parent_b_id": "parent_a_id"})
    mirrored[DELTA] = -mirrored[DELTA]
    both = pd.concat([rows, mirrored], ignore_index=True)
    model = ShrunkEnvironmentMeanBaseline(strength=1.0).fit(both)
    forward = pd.DataFrame([crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[c]")])
    backward = pd.DataFrame([crossed_pair("tetrazole", "carboxylic_acid", 0.0, "[c]")])
    assert model.predict(forward)[0] == pytest.approx(-model.predict(backward)[0])


def test_shrinkage_falls_back_through_both_levels():
    from app.baselines import ShrunkEnvironmentMeanBaseline
    model = ShrunkEnvironmentMeanBaseline().fit(shrink_rows(10, 10, 1.0, -1.0))
    test = pd.DataFrame([
        crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[c]"),   # known cell
        crossed_pair("carboxylic_acid", "tetrazole", 0.0, "[N]"),   # unseen env
        crossed_pair("carboxylic_acid", "hydroxamic_acid", 0.0, "[c]")])  # unseen swap
    assert model.coverage(test) == {"transformation_environment": 1,
                                    "transformation": 1, "global": 1}


def test_no_between_cell_signal_means_total_pooling():
    """If every cell of a class has the same mean there is nothing to learn
    from the cells, and alpha must say so rather than quietly partitioning."""
    from app.baselines import ShrunkEnvironmentMeanBaseline
    rows = shrink_rows(10, 10, 1.0, 1.0)
    # Identical spread inside each cell, so the cell means are exactly equal
    # and the between-cell component is zero.
    rows.loc[0, DELTA] = 1.5
    rows.loc[1, DELTA] = 0.5
    rows.loc[10, DELTA] = 1.5
    rows.loc[11, DELTA] = 0.5
    model = ShrunkEnvironmentMeanBaseline(strength=1.0).fit(rows)
    swap = ("carboxylic_acid", "tetrazole")
    assert model.alpha == float("inf")
    assert model.cell[(swap, "[c]")] == pytest.approx(model.group[swap])


def test_a_noiseless_training_set_trusts_its_cells():
    """The mirror image of total pooling, and the bug the first version had:
    with no within-cell noise the cell means are exact, so alpha must be 0 and
    the cells must NOT be pulled toward the class mean."""
    from app.baselines import ShrunkEnvironmentMeanBaseline
    model = ShrunkEnvironmentMeanBaseline(strength=1.0).fit(
        shrink_rows(6, 6, 1.0, -1.0))
    swap = ("carboxylic_acid", "tetrazole")
    assert model.alpha == 0.0
    assert model.cell[(swap, "[c]")] == pytest.approx(1.0)
    assert model.cell[(swap, "[C]")] == pytest.approx(-1.0)
