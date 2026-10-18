"""Prediction backend for the Gradio interface.

Loads the five seeds of one fold's all-pair-class model, blends their mean with
the shrunk_target_mean baseline exactly as the published headline does, and
refuses to answer when the two molecules are not a single headgroup swap.

Reads:  models/constant_fold<k>_4fold_cw0[_ms<n>]_allpc_e3.pt   five seeds
        data/pairs_bioisostere.csv                             baseline + target list
        data/target_properties.csv                              protein family

WHY A FOLD AT ALL
-----------------
There is no model trained on all the data. Every checkpoint was trained on four
of five molecule-disjoint folds so that the fifth could score it honestly, and
the published RMSE is the mean of those five held-out scores. The interface has
to pick one of the five, and it picks fold `FOLD` (0 by default, settable with
HEADGROUP_FOLD). The consequence is stated in the interface: molecules in fold
`FOLD` are genuinely unseen by these five models, and molecules in the other
four folds were in their training data, so a prediction on a molecule that is
already in ChEMBL is likely to look better than the published error bar.

WHY THE THREE-CLASS CALL IS A THRESHOLD, NOT A HEAD
---------------------------------------------------
The network has a classification head, but the published ensemble was trained
with --class-weight 0, so that head never learned anything and its logits are
ignored here. The call is the predicted Delta p put through the same +-1 log unit
thresholds that the reported MCC of +0.369 is computed on
(`app.pair_dataset.three_class`), so the number and the call can never disagree.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

from app.baselines import ShrunkTargetMeanBaseline
from app.headgroups import HEADGROUPS, count_headgroup
from app.model import SiameseDelta
from app.pair_dataset import (
    PairDataset, _split_result, collate_pairs, load_bioisostere_pairs, three_class,
)

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]

# The published configuration. Changing any of these makes the error bar below
# a lie, so they are constants rather than arguments.
SPLIT = "constant"
N_FOLDS = 5
SEED = 42
MODEL_SEEDS = (42, 1, 2, 3, 4)
BLEND_WEIGHT = 0.5

FOLD = int(os.environ.get("HEADGROUP_FOLD", "0"))

# The headline, for display. results/ensemble_constant_allpc_e3.csv and
# results/ensemble_predictions_constant_allpc_e3.csv are where these come from.
HEADLINE_RMSE = 0.9339
HEADLINE_FOLD_ERROR = 8.6
HEADLINE_MCC = 0.369
NOISE_FLOOR = 0.55
NOISE_FLOOR_FOLD = 3.5

CLASS_NAMES = {0: "worse", 1: "about the same", 2: "better"}

# The largest heavy-atom size difference in the 40,420 real single-swap
# comparisons this model trained on (measured, not assumed; p50 is 0.059 and
# p99 is 0.273). Past it, the two molecules cannot be a matched pair of the
# kind the model has seen.
#
# WHAT THIS DOES NOT DO, and what nothing cheap here can do: it does not check
# that the rest of the molecule is unchanged. Two tried alternatives both
# failed on this project's own data. Counting connected regions outside a
# maximum-common-substructure fired on 39% of 250 genuine pairs while calling
# aspirin-versus-caffeine a clean single-site swap, and taking 1.3 s a pair.
# Morgan Tanimoto is worse on small molecules: a real 4-chlorobenzoic acid ->
# tetrazole swap scores 0.250, below the 0.5th percentile of real pairs, and a
# swap with an unrelated Cl -> F elsewhere scores 0.081, below aspirin versus
# caffeine. Deciding that two molecules differ ONLY at the headgroup needs the
# fragmentation the training pipeline used, not a similarity score, so the
# interface says plainly that it has not checked rather than pretending to.
MAX_TRAINING_SIZE_GAP = 0.4583


@dataclass
class SwapReport:
    """Whether this molecule pair is a single headgroup swap the model can read."""

    ok: bool
    message: str
    headgroup_a: str = ""
    headgroup_b: str = ""
    notes: list = field(default_factory=list)


def _pretty(key):
    return HEADGROUPS[key].name if key in HEADGROUPS else key


@lru_cache(maxsize=1)
def _reference():
    """The pair table, its fold assignment, and the target properties."""
    pairs = load_bioisostere_pairs(REPO_ROOT / "data/pairs_bioisostere.csv",
                                   REPO_ROOT / "data/target_properties.csv")
    result = _split_result(pairs, SPLIT, N_FOLDS, SEED)
    properties = pd.read_csv(REPO_ROOT / "data/target_properties.csv")
    return pairs, result, properties


def checkpoint_path(fold, model_seed):
    tag = "" if model_seed == 42 else f"_ms{model_seed}"
    return REPO_ROOT / f"models/{SPLIT}_fold{fold}_4fold_cw0{tag}_allpc_e3.pt"


@lru_cache(maxsize=2)
def _models(fold):
    """The five seeds of this fold, each with the vocabularies it trained on."""
    loaded = []
    for model_seed in MODEL_SEEDS:
        path = checkpoint_path(fold, model_seed)
        if not path.exists():
            raise FileNotFoundError(
                f"missing checkpoint {path.name}. The interface needs all five "
                f"seeds of fold {fold}; see README.md for where they come from.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["fold"] != fold or checkpoint["train_folds"] != 4:
            raise ValueError(f"{path.name} is not a 4-fold model of fold {fold}")
        if checkpoint["model_seed"] != model_seed:
            raise ValueError(f"{path.name} records model_seed "
                             f"{checkpoint['model_seed']}, not {model_seed}")
        vocabs = checkpoint["vocabs"]
        model = SiameseDelta(len(vocabs["swap"]) + 1, len(vocabs["family"]) + 1,
                             len(vocabs["target"]) + 1, fusion=checkpoint["fusion"],
                             n_pair_classes=checkpoint.get("n_pair_classes", 0))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        loaded.append((model_seed, model, vocabs))
    return loaded


@lru_cache(maxsize=2)
def _baseline(fold):
    """shrunk_target_mean fitted on this fold's four TRAINING folds only."""
    pairs, result, _ = _reference()
    train = pairs[result.kept() & (result.folds != fold)]
    return ShrunkTargetMeanBaseline().fit(train)


@lru_cache(maxsize=2)
def target_choices(fold):
    """(label, chembl_id) for every target the models actually trained on.

    A target the model never saw has no embedding of its own: it falls back to
    the reserved "unknown" row, which carries no information about that protein.
    Those targets are left out of the list rather than offered with a warning,
    because the honest answer for them is the baseline, not this model.
    """
    pairs, result, properties = _reference()
    trained = set(_models(fold)[0][2]["target"])
    train = pairs[result.kept() & (result.folds != fold)].copy()
    train = train[train["target_chembl_id"].isin(trained)]
    # One comparison is one unordered molecule pair; the file stores both
    # directions, so counting rows would double every number here.
    train["pair_key"] = [frozenset((a, b)) for a, b
                         in zip(train["parent_a_id"], train["parent_b_id"])]
    support = train.groupby("target_chembl_id")["pair_key"].nunique()
    names = properties.set_index("target_chembl_id")["target_pref_name"].to_dict()
    organisms = properties.set_index("target_chembl_id")["organism"].to_dict()
    rows = []
    for target_id, n in support.sort_values(ascending=False).items():
        name = names.get(target_id, "unnamed target")
        organism = organisms.get(target_id, "")
        suffix = f" [{organism}]" if isinstance(organism, str) and organism \
            and organism != "Homo sapiens" else ""
        rows.append((f"{name}{suffix} - {n:,} training pairs ({target_id})", target_id))
    return rows


def _size_gap(mol_a, mol_b):
    """Difference in heavy-atom count as a fraction of the larger molecule."""
    n_a, n_b = mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms()
    if max(n_a, n_b) == 0:
        return 0.0
    return abs(n_a - n_b) / max(n_a, n_b)


def swap_between(smiles_a, smiles_b):
    """Is this pair one headgroup swapped for another? A SwapReport either way.

    The gate is counting, the same test the training data was filtered with
    (`app.swap_check`): in a genuine swap of X for Y, molecule A carries more
    copies of X than B does, and B carries more copies of Y than A does.
    Counting rather than presence is what lets a diacid lose one acid to a
    tetrazole and still count as a swap.
    """
    if not smiles_a or not smiles_a.strip() or not smiles_b or not smiles_b.strip():
        return SwapReport(False, "Enter a SMILES string for both molecules.")
    mol_a = Chem.MolFromSmiles(smiles_a.strip())
    mol_b = Chem.MolFromSmiles(smiles_b.strip())
    unreadable = [name for name, mol in (("A", mol_a), ("B", mol_b)) if mol is None]
    if unreadable:
        return SwapReport(False, f"Molecule {' and '.join(unreadable)}: RDKit cannot "
                                 f"read that SMILES.")

    # COPIES, not kinds. Two acids becoming two tetrazoles is one kind of change
    # made twice, and it is still two headgroups' worth of difference, which the
    # model has never been shown. A diacid losing ONE acid to a tetrazole is a
    # genuine single swap and has to survive this.
    removed, added = [], []
    for key in HEADGROUPS:
        in_a, in_b = count_headgroup(mol_a, key), count_headgroup(mol_b, key)
        removed.extend([key] * max(0, in_a - in_b))
        added.extend([key] * max(0, in_b - in_a))

    if len(removed) > 1 or len(added) > 1:
        changed = sorted({_pretty(k) for k in removed + added})
        return SwapReport(
            False,
            f"These two molecules differ by more than one headgroup: "
            f"{len(removed)} removed and {len(added)} added ({', '.join(changed)}). "
            f"Every training pair differed at exactly one headgroup, so the model "
            f"has nothing to say about this pair and no prediction is given.")
    if not removed and not added:
        present = sorted({_pretty(k) for k in
                          [k for k in HEADGROUPS if count_headgroup(mol_a, k)]})
        detail = f"Both carry: {', '.join(present)}." if present else \
            "Neither molecule contains an acidic headgroup this model knows."
        return SwapReport(False, "No acidic headgroup differs between these two "
                                 f"molecules, so there is no swap to predict. {detail}")
    if not added and removed:
        return SwapReport(False, f"Molecule B has lost A's {_pretty(removed[0])} "
                                 f"without gaining another headgroup. That is a "
                                 f"deletion, not a bioisostere swap, and it is not "
                                 f"what the published error bar covers.")
    if not removed:
        return SwapReport(False, f"Molecule B has gained a {_pretty(added[0])} "
                                 f"without losing a headgroup from A. That is an "
                                 f"addition, not a bioisostere swap, and it is not "
                                 f"what the published error bar covers.")

    report = SwapReport(True, f"{_pretty(removed[0])} -> {_pretty(added[0])}",
                        headgroup_a=removed[0], headgroup_b=added[0])
    gap = _size_gap(mol_a, mol_b)
    if gap > MAX_TRAINING_SIZE_GAP:
        report.notes.append(
            f"These two molecules differ in size by {gap:.0%} of their heavy atoms. "
            f"No pair in the training data differed by more than "
            f"{MAX_TRAINING_SIZE_GAP:.0%}, so whatever else is true, this is not the "
            f"kind of pair the model was shown. Check that you meant these two.")
    return report


def predict(smiles_a, smiles_b, target_id, fold=None):
    """Predict Delta p for one swap. Returns a dict; never raises on bad input."""
    fold = FOLD if fold is None else fold
    report = swap_between(smiles_a, smiles_b)
    if not report.ok:
        return {"ok": False, "message": report.message}
    if not target_id:
        return {"ok": False, "message": "Choose a protein target from the list."}

    _, _, properties = _reference()
    families = properties.set_index("target_chembl_id")["family_l2"].to_dict()
    row = {
        "parent_a_id": "query_a", "smiles_a": smiles_a.strip(),
        "parent_b_id": "query_b", "smiles_b": smiles_b.strip(),
        "headgroup_a": report.headgroup_a, "headgroup_b": report.headgroup_b,
        "target_chembl_id": target_id,
        "family_l2": families.get(target_id, "unclassified"),
        "delta_p": 0.0,            # a placeholder; nothing reads it for a prediction
        "censored_a": "exact", "censored_b": "exact",
        "constant_smiles": "query", "pair_class": "bioisostere",
    }
    query = pd.DataFrame([row])

    per_seed = {}
    for model_seed, model, vocabs in _models(fold):
        dataset = PairDataset(query, vocabs, antisymmetry=False, identity_fraction=0.0,
                              seed=SEED, name="query")
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=1, shuffle=False,
                                    collate_fn=collate_pairs):
                per_seed[model_seed] = float(model(batch)["delta"].numpy()[0])

    values = np.array([per_seed[s] for s in MODEL_SEEDS], dtype=float)
    ensemble = float(values.mean())
    baseline = float(_baseline(fold).predict(query)[0])
    blended = BLEND_WEIGHT * ensemble + (1 - BLEND_WEIGHT) * baseline

    return {
        "ok": True, "swap": report.message, "notes": report.notes,
        "target_id": target_id, "fold": fold,
        "per_seed": per_seed, "ensemble": ensemble, "baseline": baseline,
        "delta_p": blended,
        "seed_spread": float(values.max() - values.min()),
        "seed_sd": float(values.std(ddof=1)),
        "seed_low": float(values.min()), "seed_high": float(values.max()),
        "call": CLASS_NAMES[three_class(blended)],
        "fold_change": float(10 ** abs(blended)),
    }
