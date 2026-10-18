"""Training on several pair classes: leakage filter, vocabularies,
the pair-class input, and that a bioisostere-only run is unchanged."""

import pandas as pd
import pytest
import torch

from app.model import SiameseDelta
from app.pair_dataset import (
    UNKNOWN, PairDataset, build_vocabs, collate_pairs, extra_training_rows,
    pair_class_tag, parse_pair_classes,
)
from tests.test_model import ACID, TETRAZOLE, ACYLSULFONAMIDE, batch, swapped


def rows(spec):
    """spec: (a_id, b_id, constant, pair_class, delta)."""
    smiles = {"A": ACID, "B": TETRAZOLE, "C": ACYLSULFONAMIDE, "D": "OC(=O)c1ccncc1",
              "E": "NC(=O)c1ccccc1", "F": "c1ccccc1"}
    out = []
    for a, b, constant, pair_class, delta in spec:
        out.append({"parent_a_id": a, "smiles_a": smiles[a], "parent_b_id": b,
                    "smiles_b": smiles[b], "headgroup_a": "carboxylic_acid",
                    "headgroup_b": "tetrazole" if pair_class == "bioisostere" else None,
                    "pair_class": pair_class, "constant_smiles": constant,
                    "target_chembl_id": "T1", "family_l2": "F1", "delta_p": delta,
                    "censored_a": "exact", "censored_b": "exact"})
    return pd.DataFrame(out)


def test_extra_rows_touching_a_test_molecule_are_dropped():
    test = rows([("A", "B", "c1", "bioisostere", 0.5)])
    extra = rows([("A", "E", "c9", "to_non_acid", 1.0),    # shares molecule A
                  ("E", "B", "c8", "same_headgroup", 0.2),  # shares molecule B
                  ("D", "E", "c7", "to_non_acid", -0.3)])   # clean
    kept, counts = extra_training_rows(extra, test)
    assert list(kept["parent_a_id"]) == ["D"]
    assert counts["share a molecule with the test fold"] == 2


def test_extra_rows_from_a_test_series_are_dropped_even_without_a_shared_molecule():
    # D -> E cut at the same place as the test series c1: the series itself.
    test = rows([("A", "B", "c1", "bioisostere", 0.5)])
    extra = rows([("D", "E", "c1", "to_non_acid", 1.0), ("D", "F", "c2", "deletion", 0.1)])
    kept, counts = extra_training_rows(extra, test)
    assert list(kept["constant_smiles"]) == ["c2"]
    assert counts["same constant part as a test series (no shared molecule)"] == 1


def test_bioisostere_only_vocabularies_are_unchanged():
    train = rows([("A", "B", "c1", "bioisostere", 0.5), ("C", "D", "c2", "bioisostere", 0.1)])
    assert set(build_vocabs(train)) == {"swap", "family", "target"}


def test_several_classes_get_a_pair_class_vocabulary_and_index():
    train = rows([("A", "B", "c1", "bioisostere", 0.5), ("C", "E", "c2", "to_non_acid", 0.1)])
    vocabs = build_vocabs(train)
    assert vocabs["pair_class"] == {"bioisostere": 1, "to_non_acid": 2}
    dataset = PairDataset(train, vocabs, antisymmetry=True, identity_fraction=0.0)
    by_kind = {(item["id_a"], item["kind"]): item["pair_class_idx"] for item in dataset.items}
    # The mirror keeps its class: the class of (B, A) is the class of (A, B).
    assert by_kind[("A", "forward")] == by_kind[("B", "mirror")] == 1
    assert by_kind[("C", "forward")] == by_kind[("E", "mirror")] == 2
    assert collate_pairs([dataset[0], dataset[1]])["pair_class_idx"].dtype == torch.long


def test_a_class_unseen_in_training_gets_index_zero():
    train = rows([("A", "B", "c1", "bioisostere", 0.5), ("C", "E", "c2", "to_non_acid", 0.1)])
    vocabs = build_vocabs(train)
    other = rows([("D", "F", "c3", "deletion", 0.0)])
    assert PairDataset(other, vocabs, antisymmetry=False,
                       identity_fraction=0.0).items[0]["pair_class_idx"] == UNKNOWN


def test_pair_class_input_keeps_exact_antisymmetry_in_train_mode():
    torch.manual_seed(42)
    model = SiameseDelta(8, 5, 6, n_pair_classes=5).train()
    data = batch([ACID, TETRAZOLE], [TETRAZOLE, ACYLSULFONAMIDE], swap=[1, 2],
                 reverse_swap=[3, 4], family=[1, 2], target=[1, 0])
    data["pair_class_idx"] = torch.tensor([1, 3])
    back = swapped(data)
    back["pair_class_idx"] = data["pair_class_idx"]
    torch.manual_seed(0)
    forward = model(data)["delta"]
    torch.manual_seed(0)
    backward = model(back)["delta"]
    assert torch.equal(forward, -backward)


def test_pair_class_changes_the_prediction():
    torch.manual_seed(42)
    model = SiameseDelta(8, 5, 6, n_pair_classes=5).eval()
    data = batch([ACID], [TETRAZOLE], swap=[1], reverse_swap=[2], family=[1], target=[1])
    data["pair_class_idx"] = torch.tensor([1])
    first = model(data)["delta"]
    data["pair_class_idx"] = torch.tensor([2])
    assert not torch.equal(first, model(data)["delta"])


def test_a_model_without_pair_classes_has_the_old_parameter_count():
    torch.manual_seed(42)
    old = SiameseDelta(8, 5, 6)
    new = SiameseDelta(8, 5, 6, n_pair_classes=5)
    assert old.pair_class_embedding is None
    assert new.parameter_counts()["total"] - old.parameter_counts()["total"] == 5 * 8 + 8 * 300


def test_parse_and_tag():
    assert parse_pair_classes("bioisostere") == ["bioisostere"]
    assert pair_class_tag(parse_pair_classes("bioisostere")) == ""
    assert pair_class_tag(parse_pair_classes("all")) == "_allpc"
    assert pair_class_tag(parse_pair_classes("deletion,bioisostere")) == "_pc-bioisostere-deletion"
    with pytest.raises(ValueError):
        parse_pair_classes("to_non_acid")
    with pytest.raises(ValueError):
        parse_pair_classes("bioisostere,not_a_swap")


def test_extra_rows_are_filtered_against_validation_and_test_together():
    # The trainer passes val and test rows concatenated; a row touching either goes.
    val = rows([("A", "B", "c1", "bioisostere", 0.5)])
    test = rows([("C", "D", "c2", "bioisostere", 0.1)])
    extra = rows([("A", "E", "c9", "to_non_acid", 1.0),     # validation molecule
                  ("F", "C", "c8", "same_headgroup", 0.2),  # test molecule
                  ("E", "F", "c2", "deletion", 0.0),        # test series
                  ("E", "F", "c7", "to_non_acid", -0.3)])   # clean
    kept, _ = extra_training_rows(extra, pd.concat([val, test]))
    assert list(kept["constant_smiles"]) == ["c7"]


def keeps_checkpoint():
    """app/train.py's checkpoint-selection rule (a numbered script
    cannot be imported by name)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "app/train.py"
    spec = importlib.util.spec_from_file_location("train_12", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.keeps_checkpoint


def test_no_validation_fold_keeps_every_epoch():
    """--no-val: every epoch overwrites, so the file ends at the final epoch."""
    keeps = keeps_checkpoint()
    assert keeps(has_val_fold=False, keep_final=False, val_rmse=float("nan"),
                 best_rmse=float("inf"))
    assert keeps(has_val_fold=False, keep_final=False, val_rmse=float("nan"),
                 best_rmse=0.1)


def test_validation_fold_keeps_only_an_improvement():
    """The default: the file ends holding the best validation RMSE."""
    keeps = keeps_checkpoint()
    assert keeps(has_val_fold=True, keep_final=False, val_rmse=0.9, best_rmse=1.0)
    assert not keeps(has_val_fold=True, keep_final=False, val_rmse=1.1, best_rmse=1.0)
    assert not keeps(has_val_fold=True, keep_final=False, val_rmse=1.0, best_rmse=1.0)


def test_keep_final_keeps_every_epoch_with_a_validation_fold():
    """--keep-final: the validation fold is held out of training
    but never selects the model, so a worse epoch is still written."""
    keeps = keeps_checkpoint()
    assert keeps(has_val_fold=True, keep_final=True, val_rmse=1.1, best_rmse=1.0)
    assert keeps(has_val_fold=True, keep_final=True, val_rmse=0.9, best_rmse=1.0)
