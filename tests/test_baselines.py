"""Tests for app/baselines.py on a synthetic pair table.

The important ones here are not "does the median work" but:
  - the median is keyed on the DIRECTED headgroup pair, so the sign survives,
    and NOT on the mmpdb fragment string, so one swap written two ways pools
  - the fallback chain goes environment -> transformation -> global
  - a transformation split makes baseline 2 collapse to the global median, and
    that collapse is reported rather than looking like a score
"""

import numpy as np
import pandas as pd
import pytest

from app.baselines import (
    DELTA,
    TrainingMeanBaseline,
    TransformationEnvironmentMedianBaseline,
    TransformationMedianBaseline,
    ZeroBaseline,
    all_baselines,
    run_baselines,
)
from app.splits import swap_class, transformation_split

# Directed headgroup pairs - the key the median baselines use.
ACID_TO_TETRAZOLE = ("carboxylic_acid", "tetrazole")
TETRAZOLE_TO_ACID = ("tetrazole", "carboxylic_acid")
ACID_TO_SULFONAMIDE = ("carboxylic_acid", "primary_sulfonamide")
ACID_TO_NITRILE = ("carboxylic_acid", "")       # to_non_acid: no headgroup on B
ACID_TO_PHOSPHONIC = ("carboxylic_acid", "phosphonic_acid")
ARYL = "[#0;X1:1]-[c]"
ALKYL = "[#0;X1:1]-[C]"


def frame(rows):
    table = pd.DataFrame(rows, columns=["headgroup_a", "headgroup_b", "transformation",
                                        "env_radius2_smarts", "delta_p",
                                        "parent_a_id", "parent_b_id"])
    # The target and the constant arrive with the step-1.5 join; the fixture
    # carries them so every baseline can run, with one target throughout so
    # target conditioning adds nothing and the ordering assertions still hold.
    table["target_chembl_id"] = "T1"
    table["constant_smiles"] = table["parent_a_id"] + "|" + table["parent_b_id"]
    return table


def make_rows(swap, environment, deltas, tag="X", transformation=None):
    """Rows for one directed headgroup swap. The fragment string defaults to a
    made-up one; it is carried along but must not affect any baseline."""
    transformation = transformation or f"{swap[0]}>>{swap[1]}"
    return [(swap[0], swap[1], transformation, environment, delta,
             f"{tag}_A{i}", f"{tag}_B{i}")
            for i, delta in enumerate(deltas)]


@pytest.fixture
def train():
    rows = []
    # acid -> tetrazole costs potency on aryl, less so on alkyl
    rows += make_rows(ACID_TO_TETRAZOLE, ARYL, [-1.0, -1.2, -0.8], "a")
    rows += make_rows(ACID_TO_TETRAZOLE, ALKYL, [-0.2, 0.0, -0.1], "b")
    # the reverse direction must come out with the opposite sign
    rows += make_rows(TETRAZOLE_TO_ACID, ARYL, [1.0, 1.2, 0.8], "c")
    rows += make_rows(ACID_TO_SULFONAMIDE, ARYL, [0.4, 0.6], "d")
    return frame(rows)


# --- baseline 1: zero -----------------------------------------------------

def test_zero_predicts_zero(train):
    predictions = ZeroBaseline().fit(train).predict(train)
    assert len(predictions) == len(train)
    assert np.all(predictions == 0.0)


def test_zero_needs_no_columns():
    empty = pd.DataFrame({"anything": [1, 2]})
    assert np.all(ZeroBaseline().fit(empty).predict(empty) == 0.0)


# --- baseline 2: transformation median -----------------------------------

def test_transformation_median_uses_the_median(train):
    model = TransformationMedianBaseline().fit(train)
    # acid -> tetrazole over both environments: -1.0,-1.2,-0.8,-0.2,0.0,-0.1
    assert model.median_by_swap[ACID_TO_TETRAZOLE] == pytest.approx(-0.5)
    assert model.median_by_swap[ACID_TO_SULFONAMIDE] == pytest.approx(0.5)


def test_transformation_median_keeps_direction_separate(train):
    """delta is antisymmetric. If the two directions were merged the medians
    would cancel to about zero and both baselines would be destroyed."""
    model = TransformationMedianBaseline().fit(train)
    forward = model.median_by_swap[ACID_TO_TETRAZOLE]
    backward = model.median_by_swap[TETRAZOLE_TO_ACID]
    assert forward < 0 < backward
    # and the split helper really does merge them, which is the other half
    assert swap_class(*ACID_TO_TETRAZOLE) == swap_class(*TETRAZOLE_TO_ACID)


def test_one_swap_written_two_ways_is_one_median():
    """The bug this key fixes. mmpdb writes acid -> tetrazole as a different
    string depending on how much scaffold the cut left in the fragment. Keyed
    on the string these were two medians of one pair each; keyed on the
    headgroup pair they are one median of both."""
    short_cut = "[*:1]C(=O)O>>[*:1]c1nnn[nH]1"
    long_cut = "[*:1]c1ccc(C(=O)O)cc1>>[*:1]c1ccc(-c2nnn[nH]2)cc1"
    rows = (make_rows(ACID_TO_TETRAZOLE, ARYL, [-1.0], "s", transformation=short_cut)
            + make_rows(ACID_TO_TETRAZOLE, ARYL, [-0.2], "l", transformation=long_cut))
    model = TransformationMedianBaseline().fit(frame(rows))
    assert list(model.median_by_swap) == [ACID_TO_TETRAZOLE]
    assert model.median_by_swap[ACID_TO_TETRAZOLE] == pytest.approx(-0.6)


def test_a_missing_headgroup_is_its_own_swap(train):
    """to_non_acid and deletion pairs have a blank headgroup on one side. That
    must become a named key, not crash and not merge with anything else."""
    model = TransformationMedianBaseline().fit(
        frame(make_rows(ACID_TO_NITRILE, ARYL, [-2.0, -2.0], "n")))
    assert model.median_by_swap == {("carboxylic_acid", "none"): -2.0}


def test_transformation_median_falls_back_to_global(train):
    model = TransformationMedianBaseline().fit(train)
    unseen = frame(make_rows(ACID_TO_PHOSPHONIC, ARYL, [0.0], "z"))
    assert model.predict(unseen)[0] == pytest.approx(model.global_median)
    assert model.coverage(unseen) == {"transformation": 0, "global": 1}


def test_transformation_median_ignores_unusable_deltas(train):
    dirty = pd.concat([train, frame(make_rows(ACID_TO_SULFONAMIDE, ARYL, [None], "e"))],
                      ignore_index=True)
    model = TransformationMedianBaseline().fit(dirty)
    assert model.median_by_swap[ACID_TO_SULFONAMIDE] == pytest.approx(0.5)


def test_transformation_median_requires_its_columns(train):
    with pytest.raises(ValueError, match="needs columns"):
        TransformationMedianBaseline().fit(train.drop(columns=["delta_p"]))


# --- baseline 3: transformation + environment ----------------------------

def test_environment_median_is_more_specific_than_transformation(train):
    model = TransformationEnvironmentMedianBaseline().fit(train)
    aryl = frame(make_rows(ACID_TO_TETRAZOLE, ARYL, [0.0], "q"))
    alkyl = frame(make_rows(ACID_TO_TETRAZOLE, ALKYL, [0.0], "r"))
    assert model.predict(aryl)[0] == pytest.approx(-1.0)
    assert model.predict(alkyl)[0] == pytest.approx(-0.1)
    # and they differ from the transformation-level median of -0.5
    assert model.predict(aryl)[0] != pytest.approx(-0.5)


def test_environment_median_falls_back_through_both_levels(train):
    model = TransformationEnvironmentMedianBaseline().fit(train)
    # known transformation, unseen environment -> transformation median
    new_environment = frame(make_rows(ACID_TO_TETRAZOLE, "[#0;X1:1]-[N]", [0.0], "s"))
    assert model.predict(new_environment)[0] == pytest.approx(-0.5)
    assert model.coverage(new_environment)["transformation"] == 1
    # unseen transformation -> global median
    unseen = frame(make_rows(ACID_TO_NITRILE, ARYL, [0.0], "t"))
    assert model.predict(unseen)[0] == pytest.approx(
        model.transformation_baseline.global_median)
    assert model.coverage(unseen)["global"] == 1


def test_environment_median_can_require_a_minimum_count(train):
    """With min_count=3 the alkyl cell (3 rows) survives but sulfonamide/aryl
    (2 rows) does not, so the latter falls back."""
    model = TransformationEnvironmentMedianBaseline(min_count=3).fit(train)
    assert (ACID_TO_TETRAZOLE, ALKYL) in model.median_by_pair
    assert (ACID_TO_SULFONAMIDE, ARYL) not in model.median_by_pair


def test_environment_median_requires_its_columns(train):
    with pytest.raises(ValueError, match="needs columns"):
        TransformationEnvironmentMedianBaseline().fit(
            train.drop(columns=["env_radius2_smarts"]))


# --- the collapse on a transformation split ------------------------------

def test_transformation_median_collapses_on_a_transformation_split(train):
    """Every held-out transformation is unseen by construction, so baseline 2
    becomes 'predict the global median'. It must say so."""
    pairs = train.assign(constant_smiles="[*:1]c1ccccc1")
    result = transformation_split(pairs, n_folds=2)
    kept = result.kept()
    train_rows = pairs[kept & (result.folds == 0)]
    test_rows = pairs[kept & (result.folds == 1)]
    assert len(train_rows) and len(test_rows)

    model = TransformationMedianBaseline().fit(train_rows)
    counts = model.coverage(test_rows)
    assert counts["transformation"] == 0, "a test transformation leaked into training"
    assert counts["global"] == len(test_rows)

    message = model.describe_collapse(test_rows)
    assert message is not None
    assert "GLOBAL median" in message
    assert "transformation split" in message


def test_collapse_message_is_absent_when_the_baseline_is_working(train):
    model = TransformationMedianBaseline().fit(train)
    assert model.describe_collapse(train) is None


def test_partial_collapse_is_reported(train):
    """More than half falling back is worth saying out loud too."""
    model = TransformationMedianBaseline().fit(train)
    mostly_unseen = frame(
        make_rows(ACID_TO_NITRILE, ARYL, [0.0] * 5, "u")
        + make_rows(ACID_TO_TETRAZOLE, ARYL, [0.0], "v"))
    message = model.describe_collapse(mostly_unseen)
    assert message is not None and "83%" in message


def test_environment_baseline_also_collapses(train):
    model = TransformationEnvironmentMedianBaseline().fit(train)
    unseen = frame(make_rows(ACID_TO_NITRILE, "[#0;X1:1]-[S]", [0.0] * 3, "w"))
    assert model.coverage(unseen) == {"transformation_environment": 0,
                                     "transformation": 0, "global": 3}
    assert "GLOBAL median" in model.describe_collapse(unseen)


# --- running them together ------------------------------------------------

def test_all_baselines_are_in_rule_2_order():
    assert [b.name for b in all_baselines()] == [
        "zero", "training_mean", "transformation_median",
        "transformation_environment_median", "headgroup_property_ridge",
        "environment_property_ridge", "shrunk_environment_mean",
        "shrunk_target_mean", "nearest_swap_class_median"]


def test_run_baselines_returns_predictions_and_notes(train):
    predictions, notes = run_baselines(train, train)
    assert set(predictions) == {"zero", "training_mean", "transformation_median",
                                "transformation_environment_median",
                                "headgroup_property_ridge",
                                "environment_property_ridge",
                                "shrunk_environment_mean",
                                "shrunk_target_mean",
                                "nearest_swap_class_median"}
    for values in predictions.values():
        assert len(values) == len(train)
    assert notes == []          # nothing collapsed when test == train


def test_run_baselines_reports_collapse(train):
    """ACID_TO_NITRILE has a BLANK headgroup on side B, so four baselines
    collapse on it, for two different reasons: the two median baselines have
    never seen the swap, and the two property baselines cannot describe a
    headgroup that is not there. The zero and constant baselines never
    collapse. The property baselines not collapsing on an unseen swap is
    tested separately - that is the whole point of them."""
    unseen = frame(make_rows(ACID_TO_NITRILE, ARYL, [0.0] * 4, "y"))
    _predictions, notes = run_baselines(train, unseen)
    assert len(notes) == 7
    assert all("GLOBAL median" in note for note in notes)
    collapsed = {note.split(":")[0] for note in notes}
    assert collapsed == {"transformation_median",
                         "transformation_environment_median",
                         "headgroup_property_ridge",
                         "environment_property_ridge",
                         "shrunk_environment_mean",
                         "shrunk_target_mean",
                         "nearest_swap_class_median"}


def test_baselines_are_ordered_by_strength_on_data_they_fit(train):
    """On the training data itself the more specific lookup must fit better.
    Not a generalisation claim - just proof the fallback chain is wired up."""
    from app.metrics import rmse
    predictions, _notes = run_baselines(train, train)
    truth = train["delta_p"]
    errors = {name: rmse(truth, values) for name, values in predictions.items()}
    assert errors["transformation_environment_median"] <= errors["transformation_median"]
    assert errors["transformation_median"] <= errors["zero"]


# --- the constant baseline and the "best non-degenerate" reporting ------------

def test_training_mean_predicts_one_constant():
    train = pd.DataFrame({DELTA: [1.0, 2.0, 3.0, 6.0]})
    model = TrainingMeanBaseline().fit(train)
    predictions = model.predict(pd.DataFrame({DELTA: [0.0, 0.0]}))
    assert list(predictions) == [3.0, 3.0]


def test_training_mean_is_zero_when_both_directions_are_present():
    """The finding this baseline exists to demonstrate: with
    mmpdb's mirrored pairs, the best constant is 0, so the zero baseline is not
    an arbitrary choice."""
    train = pd.DataFrame({DELTA: [0.8, -0.8, 1.6, -1.6, 0.1, -0.1]})
    assert TrainingMeanBaseline().fit(train).value == pytest.approx(0.0)


def test_training_mean_never_reports_a_collapse():
    model = TrainingMeanBaseline().fit(pd.DataFrame({DELTA: [1.0]}))
    assert model.describe_collapse(pd.DataFrame({DELTA: [1.0]})) is None
    assert model.coverage(pd.DataFrame({DELTA: [1.0]})) == {}


def test_training_mean_handles_an_empty_training_set():
    model = TrainingMeanBaseline().fit(pd.DataFrame({DELTA: []}))
    assert model.value == 0.0



# --- comparison weighting -------------------------------------

def test_comparison_weights_split_one_comparison_across_its_rows():
    from app.baselines import comparison_weights
    table = pd.DataFrame([
        # one compound pair measured on three targets: three rows, weight 1/3
        {"parent_a_id": "M1", "parent_b_id": "M2", "constant_smiles": "C"},
        {"parent_a_id": "M1", "parent_b_id": "M2", "constant_smiles": "C"},
        {"parent_a_id": "M1", "parent_b_id": "M2", "constant_smiles": "C"},
        # a different pair measured once: weight 1
        {"parent_a_id": "M3", "parent_b_id": "M4", "constant_smiles": "C"},
    ])
    assert list(comparison_weights(table)) == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3, 1.0])


def test_a_pair_and_its_mirror_are_one_comparison():
    from app.baselines import comparison_weights
    table = pd.DataFrame([
        {"parent_a_id": "M1", "parent_b_id": "M2", "constant_smiles": "C"},
        {"parent_a_id": "M2", "parent_b_id": "M1", "constant_smiles": "C"},
    ])
    assert list(comparison_weights(table)) == pytest.approx([0.5, 0.5])


def test_comparison_weights_are_all_ones_without_the_id_columns():
    from app.baselines import comparison_weights
    assert list(comparison_weights(pd.DataFrame({DELTA: [1.0, 2.0]}))) == [1.0, 1.0]


def test_weighted_median_agrees_with_numpy_when_weights_are_equal():
    """Adopting a pick-a-point convention here would have moved every median
    baseline the moment weighting was switched on."""
    from app.baselines import weighted_median
    rng = np.random.default_rng(42)
    for size in (1, 2, 3, 4, 5, 6, 7, 20, 21):
        values = rng.normal(size=size)
        assert weighted_median(values, np.ones(size)) == pytest.approx(
            float(np.median(values)))


def test_weighted_median_follows_the_weights():
    from app.baselines import weighted_median
    # The 10.0 carries almost all the weight, so it is the median.
    assert weighted_median([0.0, 1.0, 10.0], [0.01, 0.01, 10.0]) == 10.0


def test_a_heavily_measured_pair_no_longer_dominates_a_median():
    """One compound pair measured on five targets used to count five times.
    Weighted, it counts once, so the median follows the four single-target
    pairs instead of the one repeated pair."""
    rows = []
    for index in range(5):
        rows.append({"headgroup_a": "carboxylic_acid", "headgroup_b": "tetrazole",
                     DELTA: 3.0, "env_radius2_smarts": "[c]",
                     "parent_a_id": "HEAVY_A", "parent_b_id": "HEAVY_B",
                     "constant_smiles": "C"})
    for index in range(4):
        rows.append({"headgroup_a": "carboxylic_acid", "headgroup_b": "tetrazole",
                     DELTA: 0.0, "env_radius2_smarts": "[c]",
                     "parent_a_id": f"A{index}", "parent_b_id": f"B{index}",
                     "constant_smiles": "C"})
    model = TransformationMedianBaseline().fit(pd.DataFrame(rows))
    # Unweighted this is median([3,3,3,3,3,0,0,0,0]) = 3.0; weighted, the
    # repeated pair carries total weight 1 against the four others' 4.
    assert model.median_by_swap[("carboxylic_acid", "tetrazole")] == 0.0


def test_a_baseline_without_its_columns_is_skipped_with_a_note():
    """On a pre-1.5 table there is no target, and that must be said rather than
    crashing the whole run or quietly dropping the baseline."""
    rows = frame(make_rows(ACID_TO_TETRAZOLE, ARYL, [-1.0, -0.5], "a"))
    no_target = rows.drop(columns=["target_chembl_id"])
    predictions, notes = run_baselines(no_target, no_target)
    assert "shrunk_target_mean" not in predictions
    skips = [note for note in notes if "SKIPPED" in note]
    assert len(skips) == 1 and "target_chembl_id" in skips[0]
