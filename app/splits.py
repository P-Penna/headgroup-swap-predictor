"""Fold splitters for the matched-pair dataset.

THE RULE THESE EXIST TO ENFORCE
-------------------------------
Never split randomly over pairs. One molecule appears in many pairs, so a
random pair split puts the SAME molecule in train and test, and the model can
score well by remembering that molecule instead of learning anything about
headgroup swaps. Every splitter here therefore works in two steps:

  1. assign whole GROUPS (scaffolds, targets, transformations, years) to folds
  2. keep only the pairs whose two ends landed in the same fold

Step 2 throws pairs away - a pair whose molecules went to different folds
cannot be used by either - and every splitter reports how many it lost, so the
cost of a split is always visible (rule 5). A split that keeps very few pairs
is telling you the dataset is one connected lump, which is worth knowing
before you trust any held-out score from it.

WHAT EACH SPLIT ANSWERS
-----------------------
  constant        can it predict swaps on a chemical series it has never seen?
                  THE PRIMARY series split.
  scaffold        the same question asked a different way, kept as a SECONDARY
                  split and reported alongside so its bias stays visible.
  target          can it predict swaps on a protein it has never seen?
  transformation  can it predict a swap it has never seen (acid -> tetrazole
                  when it only ever saw acid -> acylsulfonamide)?
  time            can it predict the future from the past, which is the only
                  split that matches how the model would really be used?

These get harder in roughly that order, and they fail in different ways, which
is why all five exist rather than one.

WHY THE CONSTANT IS THE PRIMARY SERIES SPLIT
--------------------------------------------
The constant is the part of the molecule that did NOT change - mmpdb pairs are
defined by sharing one - so it is this dataset's own definition of a series,
and both molecules in a pair always share it. Grouping on it therefore drops
no pair at all.

Bemis-Murcko asks a related but different question, and on this dataset it
answers it in a biased way: a tetrazole, thiazolidinedione, oxadiazolone and
squaramide ARE rings, so swapping them changes the Murcko scaffold and the
pair gets dropped, while acid to acylsulfonamide keeps the scaffold and
survives. A scaffold split is then not measuring generalisation to unseen
series; it is measuring generalisation to unseen series among the swaps that
happen not to touch a ring - which excludes much of the chemistry this project
exists to model. `compare_series_splits()` reports both so that bias is
visible, and if they disagree substantially on real data that disagreement is
itself a result.

THE COLUMNS EACH SPLITTER NEEDS
-------------------------------
The matched-pair table carries parent ids, SMILES,
headgroups, pair_class, the constant and the transformation - so the scaffold
and transformation splits work on it directly. The transformation split groups
on the headgroup pair, not the transformation string (see swap_class).

It does NOT carry a target or a year. A training example is (molecule A,
molecule B, target T), and the target and the measurement year only appear
once pairs are joined to the activity table in step 1.5. So the target and
time splitters need that joined table, and each splitter validates its own
required columns with a readable error rather than failing deep inside pandas.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED = 42

# Columns every splitter needs: the two molecules that make up the pair.
PAIR_COLUMNS = ["parent_a_id", "parent_b_id"]
# Extra columns per splitter, on top of PAIR_COLUMNS.
EXTRA_COLUMNS = {
    "constant": ["constant_smiles"],
    "scaffold": ["smiles_a", "smiles_b"],
    "target": ["target_chembl_id"],
    "transformation": ["headgroup_a", "headgroup_b"],
    "time": ["year"],
}

DROPPED = -1          # fold label for a pair that no fold can use


@dataclass
class SplitResult:
    """Which fold each input row belongs to, and what the split cost.

    folds: one entry per input row, in input order. DROPPED (-1) means the
           pair spans folds (or lacks the information to be placed) and must
           not be used by any fold.
    groups: the group each row was assigned by (scaffold, target, ...), kept
            so a split can be audited and so leakage checks can be re-run.
    """

    kind: str
    n_folds: int
    folds: pd.Series
    groups: pd.Series
    stats: dict = field(default_factory=dict)
    # True when this split guarantees no molecule appears in two folds. Only the
    # scaffold split gives that for free; target and transformation splits group
    # by something else and share molecules unless asked not to (see their
    # docstrings). The checker reads this flag rather than guessing from `kind`.
    molecules_disjoint: bool = False
    # True when this split guarantees no chemical series (constant_smiles)
    # appears in two folds. The constant split gives it for free; the others
    # only when asked (require_disjoint_series=True).
    series_disjoint: bool = False

    def kept(self):
        """Mask of rows that survived the split."""
        return self.folds != DROPPED

    def fold_sizes(self):
        counts = self.folds[self.kept()].value_counts().sort_index()
        return {int(fold): int(n) for fold, n in counts.items()}

    def summary(self):
        lines = [f"{self.kind} split into {self.n_folds} folds",
                 f"  pairs in:   {len(self.folds):,}",
                 f"  pairs kept: {int(self.kept().sum()):,}",
                 f"  pairs dropped (span folds): {int((~self.kept()).sum()):,}"]
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value:,}" if isinstance(value, int)
                         else f"  {key}: {value}")
        lines.append(f"  fold sizes: {self.fold_sizes()}")
        return "\n".join(lines)


def require_columns(pairs, kind):
    """Fail early and readably if the table lacks what this splitter needs."""
    if kind not in EXTRA_COLUMNS:
        raise ValueError(f"unknown split kind {kind!r}; "
                         f"expected one of {sorted(EXTRA_COLUMNS)}")
    needed = PAIR_COLUMNS + EXTRA_COLUMNS[kind]
    missing = [column for column in needed if column not in pairs.columns]
    if missing:
        hint = ""
        if kind in ("target", "time"):
            hint = (" These come from joining pairs to the activity table in "
                    "the pairing step; the raw matched pairs alone do not have them.")
        raise ValueError(f"{kind} split needs columns {missing} which are not in the "
                         f"table (it has {sorted(pairs.columns)}).{hint}")


def assign_groups_to_folds(group_sizes, n_folds, seed=SEED):
    """Spread groups over folds, keeping the folds about equal in size.

    Greedy: take the biggest group first and put it in whichever fold is
    currently smallest. Groups of equal size are shuffled with `seed` first,
    so the result is deterministic but not alphabetical (alphabetical order
    would correlate with chemistry, since similar compounds get similar names).
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    names = list(group_sizes.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    # Sort by size descending; the shuffle above breaks ties reproducibly.
    names.sort(key=lambda name: -int(group_sizes[name]))

    fold_of_group = {}
    load = [0] * n_folds
    for name in names:
        smallest = min(range(n_folds), key=lambda fold: load[fold])
        fold_of_group[name] = smallest
        load[smallest] += int(group_sizes[name])
    return fold_of_group


def molecule_table(pairs):
    """One row per parent molecule: its id and SMILES, taken from both sides.

    Raises if the same id appears with two different structures, which would
    mean the pair table is inconsistent and any molecule-level split built
    from it would be meaningless.
    """
    left = pairs[["parent_a_id", "smiles_a"]].rename(
        columns={"parent_a_id": "parent_id", "smiles_a": "smiles"})
    right = pairs[["parent_b_id", "smiles_b"]].rename(
        columns={"parent_b_id": "parent_id", "smiles_b": "smiles"})
    molecules = pd.concat([left, right], ignore_index=True).drop_duplicates()
    clashes = molecules["parent_id"].value_counts()
    clashes = clashes[clashes > 1]
    if len(clashes):
        raise ValueError(f"{len(clashes)} parent id(s) appear with more than one "
                         f"SMILES, e.g. {list(clashes.index[:3])}")
    return molecules.set_index("parent_id")["smiles"]


def murcko_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES: rings plus the linkers joining them.

    Side chains are stripped, so two analogues that differ only in their
    substituents share a scaffold. A molecule with no ring has an empty
    scaffold; those are grouped together under "" rather than each becoming
    its own group, which would defeat the point of the split.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)


def _pair_group_from_molecules(pairs, group_of_molecule):
    """Group label per pair when both molecules agree, else None."""
    groups = []
    for id_a, id_b in zip(pairs["parent_a_id"], pairs["parent_b_id"]):
        group_a = group_of_molecule.get(id_a)
        group_b = group_of_molecule.get(id_b)
        groups.append(group_a if (group_a is not None and group_a == group_b) else None)
    return pd.Series(groups, index=pairs.index)


def _folds_from_groups(pairs, pair_groups, n_folds, seed):
    """Assign groups to folds, then label each pair with its group's fold."""
    usable = pair_groups.dropna()
    sizes = usable.value_counts()
    fold_of_group = assign_groups_to_folds(sizes, n_folds, seed) if len(sizes) else {}
    folds = [fold_of_group.get(group, DROPPED) if group is not None else DROPPED
             for group in pair_groups]
    return pd.Series(folds, index=pairs.index, dtype=int), fold_of_group


def constant_split(pairs, n_folds=5, seed=SEED, require_disjoint_molecules=False):
    """Split by the shared constant: the PRIMARY series split.

    The constant is what the pair holds fixed, so both molecules share it by
    construction and no pair is ever dropped for its two ends disagreeing -
    unlike the scaffold split, which loses every swap that changes a ring.

    Molecules can still appear in more than one fold, because one cut is made
    per fragmentation and a molecule with several cuttable bonds belongs to
    several constants. So the same compound can sit in series C1 (train) and
    series C2 (test). Pass require_disjoint_molecules=True to remove that, and
    compare the kept counts to see what it costs.
    """
    require_columns(pairs, "constant")
    constants = pairs["constant_smiles"]
    sizes = constants.dropna().value_counts()
    fold_of_constant = (assign_groups_to_folds(sizes, n_folds, seed)
                        if len(sizes) else {})
    folds = pd.Series(
        [fold_of_constant.get(constant, DROPPED) if isinstance(constant, str) else DROPPED
         for constant in constants], index=pairs.index, dtype=int)

    stats = {"distinct constants (series)": int(constants.nunique()),
             "pairs with no constant": int(constants.isna().sum())}
    if require_disjoint_molecules:
        folds, dropped = _enforce_disjoint_molecules(pairs, folds, n_folds)
        stats["pairs dropped to keep molecules disjoint"] = int(dropped)
    stats["molecules shared between folds"] = int(_shared_molecule_count(pairs, folds))
    # The constant IS the series, so a constant split is series-disjoint for free.
    return SplitResult("constant", n_folds, folds, constants, stats,
                       molecules_disjoint=require_disjoint_molecules,
                       series_disjoint=True)


def scaffold_split(pairs, n_folds=5, seed=SEED):
    """Split by Bemis-Murcko scaffold: a chemical series is never split.

    A headgroup swap usually leaves the scaffold alone, so both molecules in a
    pair normally share one scaffold and the pair survives. It does not always:
    if the headgroup is itself part of a ring (a tetrazole, a thiazolidinedione),
    swapping it CHANGES the scaffold, the two molecules land in different
    groups, and the pair is dropped. That is not a bug - those pairs genuinely
    cannot be placed without leaking a scaffold across folds - but it means a
    scaffold split systematically loses ring-headgroup swaps, which is exactly
    the chemistry we care about. The dropped count makes that visible.
    """
    require_columns(pairs, "scaffold")
    smiles_of = molecule_table(pairs)
    scaffold_of = {}
    unparsed = 0
    for parent_id, smiles in smiles_of.items():
        scaffold = murcko_scaffold(smiles)
        if scaffold is None:
            unparsed += 1
            continue
        scaffold_of[parent_id] = scaffold

    pair_groups = _pair_group_from_molecules(pairs, scaffold_of)
    folds, fold_of_group = _folds_from_groups(pairs, pair_groups, n_folds, seed)
    stats = {
        "molecules": int(len(smiles_of)),
        "molecules whose SMILES would not parse": int(unparsed),
        "distinct scaffolds": int(len(set(scaffold_of.values()))),
        "scaffolds used (both ends agreed)": int(len(fold_of_group)),
        "pairs whose two molecules have different scaffolds":
            int(pair_groups.isna().sum()),
    }
    return SplitResult("scaffold", n_folds, folds, pair_groups, stats,
                       molecules_disjoint=True)


def target_split(pairs, n_folds=5, seed=SEED, require_disjoint_molecules=False):
    """Split by protein target: a target is never split across folds.

    Unlike the other splits this one does NOT make the molecule sets disjoint
    by default, because the question it asks is about targets: can the model
    predict a swap on a protein it has not seen? The same compound pair may
    appear in train on target T1 and in test on target T2.

    That is a real, if milder, leak: the model can remember "this swap was
    worth +0.5 log units" and be right on a new target whenever the effect is
    target-independent. That is precisely what baseline 2 (the median delta per
    transformation) measures, so a target split only beats that baseline if it
    has learned something target-specific. Pass
    require_disjoint_molecules=True for the stricter version, which also
    insists the molecules differ between folds and drops whatever does not fit.
    """
    require_columns(pairs, "target")
    targets = pairs["target_chembl_id"]
    sizes = targets.value_counts()
    fold_of_target = assign_groups_to_folds(sizes, n_folds, seed) if len(sizes) else {}
    folds = pd.Series([fold_of_target.get(target, DROPPED) for target in targets],
                      index=pairs.index, dtype=int)

    stats = {"distinct targets": int(targets.nunique())}
    if require_disjoint_molecules:
        folds, dropped = _enforce_disjoint_molecules(pairs, folds, n_folds)
        stats["pairs dropped to keep molecules disjoint"] = int(dropped)
    stats["molecules shared between folds"] = int(_shared_molecule_count(pairs, folds))
    return SplitResult("target", n_folds, folds, targets, stats,
                       molecules_disjoint=require_disjoint_molecules)


def _molecules_per_fold(pairs, folds):
    per_fold = {}
    for fold in sorted(set(folds[folds != DROPPED])):
        rows = folds == fold
        per_fold[fold] = set(pairs.loc[rows, "parent_a_id"]) | set(
            pairs.loc[rows, "parent_b_id"])
    return per_fold


def _shared_molecule_count(pairs, folds):
    per_fold = _molecules_per_fold(pairs, folds)
    seen = {}
    for fold, molecules in per_fold.items():
        for molecule in molecules:
            seen[molecule] = seen.get(molecule, 0) + 1
    return sum(1 for count in seen.values() if count > 1)


def _enforce_disjoint_molecules(pairs, folds, n_folds):
    """Keep a molecule in the fold where it has the most pairs; drop the rest.

    Deterministic: folds are visited in order and ties go to the lower fold.
    """
    counts = {}
    for row, fold in zip(pairs.itertuples(index=False), folds):
        if fold == DROPPED:
            continue
        for molecule in (row.parent_a_id, row.parent_b_id):
            counts.setdefault(molecule, {})
            counts[molecule][fold] = counts[molecule].get(fold, 0) + 1
    home = {}
    for molecule, per_fold in counts.items():
        best = max(sorted(per_fold), key=lambda fold: per_fold[fold])
        home[molecule] = best

    new_folds = []
    dropped = 0
    for row, fold in zip(pairs.itertuples(index=False), folds):
        if fold == DROPPED:
            new_folds.append(DROPPED)
            continue
        if home.get(row.parent_a_id) == fold and home.get(row.parent_b_id) == fold:
            new_folds.append(fold)
        else:
            new_folds.append(DROPPED)
            dropped += 1
    return pd.Series(new_folds, index=pairs.index, dtype=int), dropped


def _series_per_fold(pairs, folds):
    per_fold = {}
    for fold in sorted(set(folds[folds != DROPPED])):
        per_fold[fold] = set(pairs.loc[folds == fold, "constant_smiles"].dropna())
    return per_fold


def _shared_series_count(pairs, folds):
    if "constant_smiles" not in pairs.columns:
        return 0
    seen = {}
    for molecules in _series_per_fold(pairs, folds).values():
        for series in molecules:
            seen[series] = seen.get(series, 0) + 1
    return sum(1 for count in seen.values() if count > 1)


def _enforce_disjoint_series(pairs, folds, n_folds):
    """Keep a series in the fold where it has the most pairs; drop the rest.

    The series is `constant_smiles` - the part of the molecule the pair holds
    fixed, which is this dataset's own definition of a chemical series (see
    "WHY THE CONSTANT IS THE PRIMARY SERIES SPLIT" above).

    Why this is worth dropping pairs for: a held-out-chemistry fold claims the
    model has not seen the thing being held out. If series S appears in
    training as acid -> acylsulfonamide and in test as acid -> tetrazole, the
    model has learned something real about how S tolerates headgroup changes,
    and the fold name ("hold out tetrazole") claims more than the number
    measures. Removing the overlap is expensive; the point is to see the two
    numbers side by side rather than to pick one blind.

    Deterministic: ties go to the lower fold, as in _enforce_disjoint_molecules.
    """
    if "constant_smiles" not in pairs.columns:
        raise ValueError("require_disjoint_series needs a constant_smiles column; "
                         f"the table has {sorted(pairs.columns)}")
    counts = {}
    for series, fold in zip(pairs["constant_smiles"], folds):
        if fold == DROPPED or not isinstance(series, str):
            continue
        counts.setdefault(series, {})
        counts[series][fold] = counts[series].get(fold, 0) + 1
    home = {}
    for series, per_fold in counts.items():
        home[series] = max(sorted(per_fold), key=lambda fold: per_fold[fold])

    new_folds = []
    dropped = 0
    for series, fold in zip(pairs["constant_smiles"], folds):
        if fold == DROPPED:
            new_folds.append(DROPPED)
        elif isinstance(series, str) and home.get(series) == fold:
            new_folds.append(fold)
        else:
            new_folds.append(DROPPED)
            dropped += 1
    return pd.Series(new_folds, index=pairs.index, dtype=int), dropped


def _drop_familiar_series(pairs, train, test):
    """Remove training pairs whose series also appears in the test set.

    The train/test masks go in, a narrower train mask and a count come out.
    Shared by the two mask splitters so the rule is written once: a series is
    familiar if ANY test pair in this fold has it, whatever swap that test pair
    is of.
    """
    if "constant_smiles" not in pairs.columns:
        raise ValueError("require_disjoint_series needs a constant_smiles column; "
                         f"the table has {sorted(pairs.columns)}")
    test_series = set(pairs.loc[test, "constant_smiles"].dropna())
    familiar = train & pairs["constant_smiles"].isin(test_series)
    return train & ~familiar, int(familiar.sum())


def fold_overlap(pairs, train, test):
    """How much of a fold's test set the training set has already met.

    Both arguments are boolean masks over `pairs`. Returns the share of test
    pairs that share a MOLECULE with some training pair, and the share that
    share a SERIES (constant_smiles). These are the numbers that say whether a
    fold measures what its name claims: a fold where 87% of test series are in
    training is mostly a test on familiar chemistry.
    """
    test_rows = pairs[test]
    train_rows = pairs[train]
    n_test = len(test_rows)
    if n_test == 0:
        return {"test pairs": 0, "train pairs": len(train_rows),
                "molecule in train": 0.0, "series in train": 0.0}
    train_molecules = set(train_rows["parent_a_id"]) | set(train_rows["parent_b_id"])
    shares_molecule = (test_rows["parent_a_id"].isin(train_molecules)
                       | test_rows["parent_b_id"].isin(train_molecules))
    out = {"test pairs": n_test, "train pairs": len(train_rows),
           "molecule in train": float(shares_molecule.sum()) / n_test,
           "series in train": 0.0}
    if "constant_smiles" in pairs.columns:
        train_series = set(train_rows["constant_smiles"].dropna())
        shares_series = test_rows["constant_smiles"].isin(train_series)
        out["series in train"] = float(shares_series.sum()) / n_test
    return out


NO_HEADGROUP = "none"   # the label for the bare side of a deletion or to_non_acid pair


def _headgroup_label(headgroup):
    """A headgroup name, or NO_HEADGROUP for a blank or missing one."""
    if not isinstance(headgroup, str) or not headgroup.strip():
        return NO_HEADGROUP
    return headgroup.strip()


def swap_class(headgroup_a, headgroup_b):
    """The UNORDERED headgroup pair: the group the transformation split uses.

    Why not the mmpdb transformation string: it carries however much scaffold
    the cut happened to leave in the variable fragment, so one chemical swap
    appears under thousands of names. On the real data acid <-> tetrazole is
    14,286 pairs over 8,974 distinct strings. Keyed on the string,
    `C(=O)O>>c1nnn[nH]1` could be held out while
    `c1ccc(C(=O)O)cc1>>c1ccc(-c2nnn[nH]2)cc1` - the same swap - trained, so
    the "unseen" swap had been seen. Keyed on the headgroup pair, both are
    ('carboxylic_acid', 'tetrazole').

    Sorted, so A>>B and B>>A are one group: `mmpdb index --symmetric` stores
    both directions, and splitting them would put the training swap in the
    test set written backwards. The fragment string stays in the data as a
    feature; it is just not the grouping key.

    Returns None when neither side has a headgroup (cannot happen in
    the pair table, which requires one), so the caller can drop and count it.
    """
    left = _headgroup_label(headgroup_a)
    right = _headgroup_label(headgroup_b)
    if left == NO_HEADGROUP and right == NO_HEADGROUP:
        return None
    return tuple(sorted([left, right]))


def directed_swap(headgroup_a, headgroup_b):
    """The ORDERED headgroup pair: the key the median baselines use.

    The deliberate twin of swap_class. delta is antisymmetric (acid >> tetrazole
    at +0.8 means tetrazole >> acid at -0.8), so a median over the sorted key
    would average the two directions to about zero. Splitting must merge the
    directions; predicting must not.
    """
    left = _headgroup_label(headgroup_a)
    right = _headgroup_label(headgroup_b)
    if left == NO_HEADGROUP and right == NO_HEADGROUP:
        return None
    return (left, right)


def transformation_split(pairs, n_folds=5, seed=SEED, require_disjoint_molecules=False,
                         require_disjoint_series=False):
    """Split by swap class: a headgroup pair in test was never seen in training.

    The group is swap_class(headgroup_a, headgroup_b), e.g. ('carboxylic_acid',
    'tetrazole') - NOT the mmpdb transformation string, which fragments one
    swap into thousands of names and leaked (see swap_class). There are only
    92 swap classes among the gated bioisostere pairs and the biggest is 21%
    of them, so with 5 folds one fold is acid <-> tetrazole alone. That is the
    experiment: "predict a whole headgroup pair you never trained on".

    The hardest and most honest of the four for this project, because it is the
    question a medicinal chemist actually asks - "I have not tried this
    bioisostere, what will it do?" It also breaks baseline 2 by construction:
    the training median for a transformation in the test set does not exist, so
    a model must generalise across transformations rather than memorise each.

    LIKE THE TARGET SPLIT, THIS SHARES MOLECULES BETWEEN FOLDS by default, and
    it is worth being clear why. One acid takes part in many transformations -
    acid to tetrazole, acid to acylsulfonamide, acid to sulfonamide - so holding
    out a transformation cannot also hold out its molecules unless almost every
    pair is thrown away. The first version of this module asserted molecule
    disjointness for every split; it failed here, and the assertion was the
    thing that was wrong, not the splitter.

    The residual leak is real: the model may see molecule A in training (as
    A -> acylsulfonamide) and again in test (as A -> tetrazole), so it can
    learn something about A itself. What it cannot do is look up the answer for
    the transformation, which is the thing being tested. Pass
    require_disjoint_molecules=True for the strict version, which removes the
    leak and keeps far fewer pairs; compare the two to see what that costs.

    THE SAME GOES FOR THE SERIES, and it matters more. Measured on the 43,054
    genuine gated pairs, 37.6-61.0% of each fold's test pairs have their series
    (constant_smiles) in the training set. Training on "this series tolerates
    acid -> acylsulfonamide" is real information about the held-out
    acid -> tetrazole on the same series, so the fold measures less than its
    name claims. require_disjoint_series=True gives every series one home fold
    and drops the rest.
    """
    require_columns(pairs, "transformation")
    keys = pd.Series([swap_class(a, b) for a, b in
                      zip(pairs["headgroup_a"], pairs["headgroup_b"])],
                     index=pairs.index, dtype=object)
    sizes = keys.dropna().value_counts()
    fold_of_key = assign_groups_to_folds(sizes, n_folds, seed) if len(sizes) else {}
    folds = pd.Series([fold_of_key.get(key, DROPPED) if key is not None else DROPPED
                       for key in keys], index=pairs.index, dtype=int)
    stats = {
        "distinct swap classes (headgroup pairs, directions merged)":
            int(sizes.shape[0]),
        "largest swap class (share of pairs)":
            f"{sizes.iloc[0] / sizes.sum():.1%}" if len(sizes) else "n/a",
        "pairs with no headgroup on either side (dropped)": int(keys.isna().sum()),
    }
    # Series first, then molecules: both only ever drop rows, so the order does
    # not change the final mask, but doing series first makes its cost - the
    # larger of the two - readable on its own line.
    if require_disjoint_series:
        folds, dropped = _enforce_disjoint_series(pairs, folds, n_folds)
        stats["pairs dropped to keep series disjoint"] = int(dropped)
    if require_disjoint_molecules:
        folds, dropped = _enforce_disjoint_molecules(pairs, folds, n_folds)
        stats["pairs dropped to keep molecules disjoint"] = int(dropped)
    stats["molecules shared between folds"] = int(_shared_molecule_count(pairs, folds))
    stats["series shared between folds"] = int(_shared_series_count(pairs, folds))
    return SplitResult("transformation", n_folds, folds, keys, stats,
                       molecules_disjoint=require_disjoint_molecules,
                       series_disjoint=require_disjoint_series)


def time_split(pairs, cutoff_year, seed=SEED):
    """Train on everything up to cutoff_year, test on everything after.

    Two folds only (0 = train, 1 = test), because time has a direction and
    k-fold shuffling would train on the future to predict the past.

    A pair gets the LATER of its two molecules' years: the pair could not have
    been made until both halves existed. Pairs with a missing year cannot be
    placed and are dropped, not guessed - 167,602 activity records have no
    year, so this is not a rare case.
    """
    require_columns(pairs, "time")
    years = pd.to_numeric(pairs["year"], errors="coerce")
    folds = []
    for year in years:
        if pd.isna(year):
            folds.append(DROPPED)
        elif year <= cutoff_year:
            folds.append(0)
        else:
            folds.append(1)
    folds = pd.Series(folds, index=pairs.index, dtype=int)
    stats = {
        "cutoff year": int(cutoff_year),
        "pairs with no year (dropped)": int(years.isna().sum()),
        "train pairs (<= cutoff)": int((folds == 0).sum()),
        "test pairs (> cutoff)": int((folds == 1).sum()),
        "molecules shared between folds": int(_shared_molecule_count(pairs, folds)),
    }
    return SplitResult("time", 2, folds, years, stats)


# The primary series split. scaffold is kept as a secondary view of the same
# question; see compare_series_splits() and the module docstring.
# --- the transformation split as per-fold masks -------------------------------
#
# WHY THIS EXISTS ALONGSIDE transformation_split
#
# transformation_split returns one fold label per row, so "strict" there has to
# give each series a single home fold and drop every pair whose series lives
# elsewhere. That keeps the dataset a clean partition, but it removes TEST pairs
# as well as training ones: fold 0 goes from 9,424 test pairs to 7,802. Loose
# and strict are then scored on different test sets, and the headline number
# this project is built to report - "performance drops by X when the series is
# held out too" - would mix two effects: removing the series leakage, and
# testing on a different 83% of the pairs. A reviewer asks which one it is and
# there is no answer.
#
# So the version that goes in the results table keeps every test fold exactly
# as the loose split drew it and only ever removes TRAINING pairs, the same way
# the leave-one-headgroup-out split does. Train is then not "everything else",
# which one fold label per row cannot express - hence masks.
#
# transformation_split(require_disjoint_series=True) is kept for the cases that
# want a true partition.


@dataclass
class MaskFold:
    """One held-out fold as three masks, where train is not "everything else".

    name:     what the fold holds out, for the results table
    test:     the fold's pairs - identical loose or strict, by construction
    train:    pairs used to fit, after any series removal
    excluded: everything else, so the three masks partition every input row
    """

    name: str
    train: pd.Series
    test: pd.Series
    excluded: pd.Series
    fold: int = -1
    train_dropped_for_series: int = 0


@dataclass
class MaskSplitResult:
    kind: str
    folds: list
    stats: dict = field(default_factory=dict)
    series_disjoint: bool = False

    def fold(self, name):
        for fold in self.folds:
            if fold.name == name or fold.fold == name:
                return fold
        raise KeyError(f"no fold named {name!r}")

    def summary(self):
        mode = "STRICT (series disjoint)" if self.series_disjoint else "loose"
        lines = [f"{self.kind} split as per-fold masks, {mode}: "
                 f"{len(self.folds)} folds"]
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value:,}" if isinstance(value, int)
                         else f"  {key}: {value}")
        for fold in self.folds:
            strict = (f"  train dropped for series {fold.train_dropped_for_series:>6,}"
                      if self.series_disjoint else "")
            lines.append(f"  fold {fold.fold}: test {int(fold.test.sum()):>6,}  "
                         f"train {int(fold.train.sum()):>6,}  "
                         f"excluded {int(fold.excluded.sum()):>6,}{strict}")
        return "\n".join(lines)


def transformation_folds(pairs, n_folds=5, seed=SEED, require_disjoint_series=False):
    """The transformation split with per-fold training masks.

    The folds are drawn exactly as transformation_split draws them - groups are
    swap classes, so a held-out headgroup pair is unseen - and the TEST masks do
    not depend on require_disjoint_series at all. Only training changes:

      loose   train = every pair outside the fold (identical to
              score_one_split on transformation_split, so loose numbers are
              unchanged by this refactor)
      strict  train = those pairs minus any whose series appears in this fold's
              test set

    Because the test set is fixed, the loose and strict rows of the results
    table are the same experiment on the same pairs with a cleaner training set,
    and their difference is the cost of the series leak and nothing else.
    """
    base = transformation_split(pairs, n_folds=n_folds, seed=seed)
    kept = base.kept()
    folds = []
    for number in sorted(set(base.folds[kept])):
        test = kept & (base.folds == number)
        train = kept & (base.folds != number)
        dropped = 0
        if require_disjoint_series:
            train, dropped = _drop_familiar_series(pairs, train, test)
        excluded = ~test & ~train
        counts = base.groups[test].value_counts()
        name = " / ".join(f"{a} <-> {b}" for a, b in counts.index[:1])
        if len(counts) > 1:
            name += f" (+{len(counts) - 1} smaller)"
        folds.append(MaskFold(name=name, train=train, test=test, excluded=excluded,
                              fold=int(number), train_dropped_for_series=dropped))
    stats = dict(base.stats)
    stats.pop("molecules shared between folds", None)
    stats.pop("series shared between folds", None)
    stats["pairs never placed in any fold"] = int((~kept).sum())
    return MaskSplitResult("transformation", folds, stats,
                           series_disjoint=require_disjoint_series)


PRIMARY_SERIES_SPLIT = "constant"
SECONDARY_SERIES_SPLIT = "scaffold"

SPLITTERS = {
    "constant": constant_split,
    "scaffold": scaffold_split,
    "target": target_split,
    "transformation": transformation_split,
    "time": time_split,
}


# --- leakage checks ---------------------------------------------------------
#
# These are the assertions that make a split trustworthy. They are written to
# be run on real data too, not only in tests: a split that passes its tests on
# synthetic rows can still leak on the real table if an assumption about the
# data is wrong.

def assert_molecules_disjoint(pairs, result):
    """No molecule may appear in two folds - but only for splits that promise it.

    Only the scaffold split (and the strict variants of the target and
    transformation splits) guarantee this. The others group by target or by
    transformation, where one molecule necessarily belongs to several groups,
    so applying this check to them would be asserting something false rather
    than catching a bug.
    """
    if not result.molecules_disjoint:
        return
    per_fold = _molecules_per_fold(pairs, result.folds)
    for fold, molecules in per_fold.items():
        others = set()
        for other_fold, other in per_fold.items():
            if other_fold != fold:
                others |= other
        overlap = molecules & others
        if overlap:
            raise AssertionError(
                f"{result.kind} split: {len(overlap)} molecule(s) appear in fold "
                f"{fold} and another fold, e.g. {sorted(overlap)[:3]}")


def assert_series_disjoint(pairs, result):
    """No series may appear in two folds - but only for splits that promise it.

    The twin of assert_molecules_disjoint. Reads result.series_disjoint rather
    than guessing from result.kind, so a loose split is not asserted to be
    something it never claimed.
    """
    if not result.series_disjoint:
        return
    if "constant_smiles" not in pairs.columns:
        raise ValueError("a series-disjoint split cannot be checked without a "
                         "constant_smiles column")
    per_fold = _series_per_fold(pairs, result.folds)
    for fold, series in per_fold.items():
        others = set()
        for other_fold, other in per_fold.items():
            if other_fold != fold:
                others |= other
        overlap = series & others
        if overlap:
            raise AssertionError(
                f"{result.kind} split: {len(overlap)} series appear in fold "
                f"{fold} and another fold, e.g. {sorted(overlap)[:2]}")


# --- mirrored rows ------------------------------------------------------------
#
# `mmpdb index --symmetric` stores every pair in BOTH directions, so the joined
# table holds each comparison twice: A -> B with delta +d and B -> A with -d.
# Measured on the full pairs table, 100% of the 1,291,070 rows have their
# mirror present. That makes a new kind of leak possible: train on
# A -> B and test on B -> A and the label is simply the negation of one already
# seen, which is the purest form of the rule-1 failure.
#
# Every splitter here is safe from it by construction - the two directions share
# both molecules, the constant, the target and (because swap_class sorts) the
# swap class, so nothing can separate them. This checks it anyway, because
# "safe by construction" is an argument and the assertion is a measurement.

# `year` is part of what makes two rows the same comparison, not just a column
# that happens to be there: the time split is MEANT to separate the same
# comparison measured in different years, and on the real data a pair's year is
# max(year_a, year_b), which is symmetric, so true mirrors always share it.
MIRROR_EXTRA_COLUMNS = ("constant_smiles", "target_chembl_id", "standard_type",
                        "year")


def _pair_identity(pairs, swap=False):
    """What makes a row the same comparison, with the two molecules optionally
    swapped. Only the columns the table actually has are used, so this works
    before and after the step-1.5 join."""
    first, second = ("parent_b_id", "parent_a_id") if swap \
        else ("parent_a_id", "parent_b_id")
    names = [first, second] + [name for name in MIRROR_EXTRA_COLUMNS
                               if name in pairs.columns]
    # Stringified because a missing value must match a missing value: 15,320
    # rows of pairs.csv have no year, and NaN != NaN would make their mirrors
    # invisible to this check - a silent false pass, which is worse than a
    # false alarm.
    columns = [pairs[name].astype(str) for name in names]
    return pd.Series(list(zip(*columns)), index=pairs.index)


def one_row_per_comparison(pairs):
    """Mask keeping ONE row per comparison, with the mirrors collapsed.

    For SCORING, not for training. The file keeps both
    directions - a model should learn that Delta p is antisymmetric - but a
    metric computed over both counts every comparison twice, so `n` is not a
    count of independent comparisons and the sign distribution is symmetric by
    construction. On the real data this takes the 57,184 bioisostere rows down
    to the 20,210 compound pairs they describe.

    Which direction survives is decided by row order, not by chemistry: the
    first row of each unordered identity is kept. That is deterministic and
    makes no assumption about which headgroup "should" be the parent. The
    alternative - always putting the carboxylic acid on side A - would make
    the surviving signs directly readable ("acid -> tetrazole loses 0.3") but
    has no answer for the pairs where neither side is an acid.

    Robust to a filtered table: rows are collapsed by identity rather than by
    keeping `parent_a_id < parent_b_id`, so if one direction has already been
    removed the other is still kept rather than both being dropped.
    """
    names = [name for name in MIRROR_EXTRA_COLUMNS if name in pairs.columns]
    columns = [pairs[name].astype(str) for name in ["parent_a_id", "parent_b_id"]
               + names]
    keep = []
    seen = set()
    for values in zip(*columns):
        molecules, extras = values[:2], values[2:]
        identity = (tuple(sorted(molecules)), extras)
        keep.append(identity not in seen)
        seen.add(identity)
    return pd.Series(keep, index=pairs.index)


def assert_mirrors_together(pairs, train, test, label="fold"):
    """No test row's mirror (the same comparison written backwards) may train."""
    forward = _pair_identity(pairs)
    reverse = _pair_identity(pairs, swap=True)
    trained = set(forward[train])
    straddle = reverse[test].isin(trained)
    if straddle.any():
        example = pairs.loc[straddle[straddle].index[0],
                            ["parent_a_id", "parent_b_id"]].tolist()
        raise AssertionError(
            f"{label}: {int(straddle.sum())} test row(s) have their mirror "
            f"(B -> A) in training, e.g. {example}. The label of a mirror is "
            f"just the negation, so this is leakage.")
    return True


def assert_groups_not_split(result):
    """A group must live in exactly one fold, or the split means nothing."""
    frame = pd.DataFrame({"group": result.groups, "fold": result.folds})
    frame = frame[(frame["fold"] != DROPPED) & frame["group"].notna()]
    per_group = frame.groupby("group")["fold"].nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise AssertionError(f"{result.kind} split: {len(bad)} group(s) span more than "
                             f"one fold, e.g. {list(bad.index[:3])}")


def assert_time_order(result):
    """Nothing in the time test fold may predate the train fold's cutoff."""
    if result.kind != "time":
        return
    years = pd.to_numeric(result.groups, errors="coerce")
    cutoff = result.stats["cutoff year"]
    train_max = years[result.folds == 0].max()
    test_min = years[result.folds == 1].min()
    if pd.notna(train_max) and train_max > cutoff:
        raise AssertionError(f"time split: a train pair is from {train_max}, "
                             f"after the cutoff {cutoff}")
    if pd.notna(test_min) and test_min <= cutoff:
        raise AssertionError(f"time split: a test pair is from {test_min}, "
                             f"at or before the cutoff {cutoff}")


def compare_series_splits(pairs, n_folds=5, seed=SEED, top_n=10):
    """Run both series splits and report where they disagree.

    The headline number is how many pairs the scaffold split throws away that
    the constant split keeps. Those are the swaps that change a ring, so the
    transformations listed are expected to be dominated by ring headgroups
    (tetrazole, thiazolidinedione, oxadiazolone, squaramide). Reporting it
    keeps the scaffold split's bias in view instead of hidden inside a score.
    """
    primary = constant_split(pairs, n_folds=n_folds, seed=seed)
    secondary = scaffold_split(pairs, n_folds=n_folds, seed=seed)
    lost = pairs[primary.kept() & ~secondary.kept()]
    report = {
        "pairs": len(pairs),
        "constant kept": int(primary.kept().sum()),
        "scaffold kept": int(secondary.kept().sum()),
        "kept by constant but dropped by scaffold": len(lost),
        "share of constant-kept pairs lost by scaffold":
            (len(lost) / int(primary.kept().sum()) if primary.kept().any() else 0.0),
        "transformations scaffold loses most":
            lost["transformation"].value_counts().head(top_n).to_dict()
            if "transformation" in lost.columns else {},
    }
    return primary, secondary, report


def check_split(pairs, result):
    """Run every check that applies to this split. Raises on the first failure."""
    assert_groups_not_split(result)
    assert_molecules_disjoint(pairs, result)
    assert_series_disjoint(pairs, result)
    assert_time_order(result)
    kept = result.kept()
    for fold in sorted(set(result.folds[kept])):
        assert_mirrors_together(pairs, kept & (result.folds != fold),
                                kept & (result.folds == fold),
                                label=f"{result.kind} split, fold {fold}")
    return True


# --- leave one headgroup out ---------------------------------------------------
#
# Not in SPLITTERS, because it does not fit SplitResult's one-fold-per-row
# shape: a tetrazole <-> phosphonic pair is test data for BOTH the tetrazole
# fold and the phosphonic fold, which is correct - it tests both claims. So
# each fold carries its own train and test masks.

# Held out, one fold each. Chosen by the number of distinct MOLECULES bearing
# the headgroup in genuine gated bioisostere pairs (mmps_collapsed.csv), not
# by pairs: one tetrazole compound paired with 20 acids is 20 pairs but one
# example of a tetrazole. The first five have 500+ molecules each; sulfonic
# (229) and oxadiazolone (196) are borderline and reported as such. Every
# other headgroup has 70 or fewer and a fold for it would be noise.
HELD_OUT_HEADGROUPS = [
    "tetrazole", "acylsulfonamide", "phosphonic_acid", "hydroxamic_acid",
    "primary_sulfonamide", "sulfonic_acid", "oxadiazol_5_one",
]
BORDERLINE_HEADGROUPS = ["sulfonic_acid", "oxadiazol_5_one"]
# Never held out: the acid is on ~85% of pairs, so removing it leaves almost
# nothing to train on, and "none" is the bare side of a deletion, not a group.
NEVER_HELD_OUT = ["carboxylic_acid", NO_HEADGROUP]


@dataclass
class HeldOutFold:
    """One leave-one-headgroup-out fold, as boolean masks over the input rows.

    test:     pairs whose swap involves the headgroup (headgroup_a or _b is it)
    train:    pairs where NEITHER molecule carries the headgroup anywhere
    excluded: the rest - a molecule carries the headgroup but it was not the
              thing swapped. Kept out of training so the model never sees the
              group at all, and out of test because they do not test a swap of it.
              Under require_disjoint_series=True this also holds the training
              pairs removed for sharing a series with the test set, so the three
              masks still partition every input row.
    """

    headgroup: str
    train: pd.Series
    test: pd.Series
    excluded: pd.Series
    borderline: bool = False
    molecules_with_headgroup: int = 0
    # Set when require_disjoint_series=True: training pairs moved into
    # `excluded` because their series also appears in this fold's test set.
    train_dropped_for_series: int = 0


@dataclass
class HeadgroupSplitResult:
    folds: list
    not_held_out: dict = field(default_factory=dict)
    kind: str = "headgroup"
    series_disjoint: bool = False

    def fold(self, headgroup):
        for fold in self.folds:
            if fold.headgroup == headgroup:
                return fold
        raise KeyError(f"no fold holds out {headgroup!r}")

    def summary(self):
        mode = "STRICT (series disjoint)" if self.series_disjoint else "loose"
        lines = [f"leave-one-headgroup-out, {mode}: {len(self.folds)} folds"]
        for fold in self.folds:
            note = "  (BORDERLINE: few molecules)" if fold.borderline else ""
            strict = (f"  train dropped for series {fold.train_dropped_for_series:>6,}"
                      if self.series_disjoint else "")
            lines.append(
                f"  hold out {fold.headgroup:<20} test {int(fold.test.sum()):>6,}  "
                f"train {int(fold.train.sum()):>6,}  "
                f"excluded {int(fold.excluded.sum()):>5,}  "
                f"molecules with it {fold.molecules_with_headgroup:>5,}{strict}{note}")
        if self.not_held_out:
            lines.append("  not held out (too few molecules for a fold to mean anything):")
            for headgroup, molecules in self.not_held_out.items():
                lines.append(f"    {headgroup:<24} {molecules:>4} molecules")
        return "\n".join(lines)


def headgroup_split(pairs, headgroups=None, require_disjoint_series=False):
    """Leave one headgroup out: can the model predict a group it has NEVER seen?

    Stricter than the transformation split. Holding out acid <-> tetrazole
    there still trains on acid <-> acylsulfonamide AND acylsulfonamide <->
    tetrazole, so the model has seen tetrazoles, just never swapped for an
    acid. Here, for headgroup H:
      test  = every pair whose swap involves H
      train = every pair in which neither molecule contains H anywhere
    "Anywhere" is checked by copy count on the SMILES (count_headgroup), not
    by the labels, because ~10% of molecules carry two headgroups: a molecule
    with a tetrazole and an acid can appear as an acid <-> sulfonamide swap
    with the tetrazole sitting unchanged, and training on it would show the
    model a tetrazole.

    The acid partner of a test pair can still appear in training (paired with
    another group), as in the transformation split. What the model cannot
    have seen is H.

    AND SO CAN ITS SERIES, which is the bigger hole. Measured on the 43,054
    genuine gated pairs, the share of each fold's test pairs whose series
    (constant_smiles) also appears in training runs 39.3-56.8% for the five
    clear folds and 75.4% (sulfonic) and 85.3% (oxadiazolone) for the two
    borderline ones. If training shows that series S tolerates
    acid -> acylsulfonamide, the model has learned something real about S
    before meeting it again at test time as acid -> tetrazole, so "hold out
    oxadiazolone" is largely a test on series it already knows.

    require_disjoint_series=True moves every training pair whose series appears
    in this fold's test set into `excluded`. It costs a lot of training data and
    it is not the default; run both and read them side by side.
    """
    from rdkit import Chem

    from app.headgroups import count_headgroup

    needed = ["parent_a_id", "parent_b_id", "smiles_a", "smiles_b",
              "headgroup_a", "headgroup_b"]
    missing = [column for column in needed if column not in pairs.columns]
    if missing:
        raise ValueError(f"headgroup split needs columns {missing}")
    headgroups = HELD_OUT_HEADGROUPS if headgroups is None else list(headgroups)
    forbidden = [h for h in headgroups if h in NEVER_HELD_OUT]
    if forbidden:
        raise ValueError(f"{forbidden} must never be held out (see NEVER_HELD_OUT)")

    labels_a = pairs["headgroup_a"].map(_headgroup_label)
    labels_b = pairs["headgroup_b"].map(_headgroup_label)

    # Parse each distinct molecule once.
    molecules = {}
    for side in ("a", "b"):
        for parent, smiles in zip(pairs[f"parent_{side}_id"], pairs[f"smiles_{side}"]):
            if parent not in molecules:
                molecules[parent] = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) \
                    else None

    folds = []
    for headgroup in headgroups:
        carriers = set()
        for parent, mol in molecules.items():
            if mol is not None and count_headgroup(mol, headgroup) > 0:
                carriers.add(parent)
        test = (labels_a == headgroup) | (labels_b == headgroup)
        carries = pairs["parent_a_id"].isin(carriers) | pairs["parent_b_id"].isin(carriers)
        train = ~test & ~carries
        dropped_for_series = 0
        if require_disjoint_series:
            train, dropped_for_series = _drop_familiar_series(pairs, train, test)
        # Everything that is neither test nor train, so the three masks always
        # cover every input row exactly once.
        excluded = ~test & ~train
        swapped = set(pairs.loc[labels_a == headgroup, "parent_a_id"]) | \
            set(pairs.loc[labels_b == headgroup, "parent_b_id"])
        folds.append(HeldOutFold(headgroup, train, test, excluded,
                                 borderline=headgroup in BORDERLINE_HEADGROUPS,
                                 molecules_with_headgroup=len(swapped),
                                 train_dropped_for_series=dropped_for_series))

    not_held_out = {}
    everything = set(labels_a) | set(labels_b)
    for headgroup in sorted(everything - set(headgroups) - set(NEVER_HELD_OUT)):
        swapped = set(pairs.loc[labels_a == headgroup, "parent_a_id"]) | \
            set(pairs.loc[labels_b == headgroup, "parent_b_id"])
        not_held_out[headgroup] = len(swapped)
    not_held_out = dict(sorted(not_held_out.items(), key=lambda item: -item[1]))
    return HeadgroupSplitResult(folds, not_held_out,
                                series_disjoint=require_disjoint_series)


def assert_mask_folds(pairs, result):
    """Checks that apply to any per-fold-mask split (MaskFold or HeldOutFold).

    Shared by the transformation mask split and leave-one-headgroup-out: the
    three masks must partition every row, and if the split claims series
    disjointness it must actually have it. Reads result.series_disjoint rather
    than the data, so a loose split is not asserted into something it never
    promised.
    """
    for fold in result.folds:
        name = getattr(fold, "name", None) or getattr(fold, "headgroup", "fold")
        if (fold.train & fold.test).any():
            raise AssertionError(f"{name}: a row is in both train and test")
        covered = (fold.train.astype(int) + fold.test.astype(int)
                   + fold.excluded.astype(int))
        if not (covered == 1).all():
            raise AssertionError(f"{name}: train/test/excluded do not partition "
                                 f"the rows")
        if result.series_disjoint:
            test_series = set(pairs.loc[fold.test, "constant_smiles"].dropna())
            shared = set(pairs.loc[fold.train, "constant_smiles"].dropna()) & test_series
            if shared:
                raise AssertionError(
                    f"{name}: {len(shared)} series are in both train and test "
                    f"despite require_disjoint_series, e.g. {sorted(shared)[:2]}")
        assert_mirrors_together(pairs, fold.train, fold.test, label=name)
    return True


def assert_headgroup_never_seen(pairs, result):
    """No training row may carry the held-out headgroup, by label OR by structure."""
    from rdkit import Chem

    from app.headgroups import count_headgroup

    assert_mask_folds(pairs, result)
    for fold in result.folds:
        train = pairs[fold.train]
        for column in ("headgroup_a", "headgroup_b"):
            if (train[column] == fold.headgroup).any():
                raise AssertionError(f"{fold.headgroup} appears as a label in training")
        for smiles in set(train["smiles_a"]) | set(train["smiles_b"]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and count_headgroup(mol, fold.headgroup) > 0:
                raise AssertionError(
                    f"{fold.headgroup} is present in training molecule {smiles}")
    return True
