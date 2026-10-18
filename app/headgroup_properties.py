"""Physical properties of each acidic headgroup, for the property baseline.

WHY THIS EXISTS
---------------
Every baseline so far looks the swap up in the training data: the median delta
for this transformation, or for this transformation in this environment. On a
held-out-chemistry split that lookup cannot work - the swap class is unseen by
construction - so all of them fall back to the global median, which on this
mirror-symmetric dataset is exactly 0. Measured: across all 48 fold x series x
censoring groups of the transformation and headgroup splits, all four baselines
share one RMSE, which is the weakest part of that measurement. Those folds are
the ones the
project exists to report, and they currently have no reference point except
"predict nothing".

A baseline keyed on PROPERTIES of the two headgroups has a prediction for a
group it has never seen, because a tetrazole still has a pKa and a TPSA whether
or not any tetrazole was in training. That gives those folds a real second
reference point.

IT IS ALSO A TEST OF THE HYPOTHESIS THE PROJECT RESTS ON. Bioisosterism says a
headgroup swap is tolerated when the replacement matches the original on the
properties the protein actually sees: charge at physiological pH, size, shape,
hydrogen-bonding pattern. If a linear model on five such numbers predicts Delta
p usefully, that is evidence a physics-feature channel is worth
building. If it predicts nothing, that is worth knowing before building it.

THE PROPERTIES
--------------
Computed on a METHYL-CAPPED NEUTRAL FRAGMENT of each headgroup, e.g. acetic
acid for the carboxylic acid, 5-methyl-1H-tetrazole for the tetrazole. The cap
is needed because a bare fragment with a dangling valence is not a molecule
RDKit will give sensible descriptors for.

  pka          the FIRST ionization, in pH units. Hand-curated (see PKA below).
  heavy_atoms  heavy atoms in the group itself, with the cap carbons subtracted
               so a group is not credited for its methyl. Phosphinic acid is
               the reason this matters: its definition requires two carbons on
               phosphorus, so its fragment carries two caps where every other
               carries one.
  hbd          Lipinski H-bond donors, on the neutral form.
  hba          Lipinski H-bond acceptors, on the neutral form.
  tpsa         topological polar surface area of the capped fragment. A methyl
               contributes 0, so the cap does not distort it.

THE pKa NUMBERS ARE HAND-CURATED, AND THAT IS A REAL LIMITATION
---------------------------------------------------------------
RDKit does not predict pKa and the project's stack has no pKa predictor, so
these are single numbers read off the literature ranges already recorded in
`app/headgroups.py`. Where that field gives a range, the midpoint is used;
where it says the value is uncertain, PKA_UNCERTAIN records that so a caller
can exclude those groups and see whether the model depends on them. They are
properties of the BARE headgroup and ignore the molecule it sits on, which can
move a pKa by a unit or more - a 5-aryl tetrazole and a 5-alkyl tetrazole are
not the same acid. So pka here is a class label more than a measurement, and a
coefficient on it should be read with that in mind.
"""

from app.headgroups import HEADGROUPS

# A methyl-capped neutral fragment per headgroup. Every one of these is checked
# by a test to (a) parse and (b) contain exactly one copy of its own headgroup
# according to the SMARTS in app/headgroups.py - which is what catches a
# plausible-looking SMILES that is quietly the wrong molecule.
CAPPED_SMILES = {
    "carboxylic_acid": "CC(=O)O",                  # acetic acid
    "tetrazole": "Cc1nnn[nH]1",                    # 5-methyl-1H-tetrazole
    "hydroxyisoxazole_3": "Cc1cc(O)no1",           # 3-hydroxy-5-methylisoxazole
    "hydroxypyrazole_3": "Cc1cc(O)n[nH]1",         # 3-hydroxy-5-methylpyrazole
    "oxadiazol_5_one": "CC1=NOC(=O)N1",            # 3-methyl-1,2,4-oxadiazol-5(4H)-one
    "thiadiazol_5_one": "CC1=NSC(=O)N1",           # the sulfur analogue
    "acylsulfonamide": "CC(=O)NS(C)(=O)=O",        # N-(methanesulfonyl)acetamide
    "hydroxamic_acid": "CC(=O)NO",                 # acetohydroxamic acid
    "acyl_cyanamide": "CC(=O)NC#N",                # N-cyanoacetamide
    "primary_sulfonamide": "CS(N)(=O)=O",          # methanesulfonamide
    "sulfonic_acid": "CS(=O)(=O)O",                # methanesulfonic acid
    "phosphonic_acid": "CP(=O)(O)O",               # methylphosphonic acid
    # Two carbons on phosphorus are part of the definition, so two caps.
    "phosphinic_acid": "CP(C)(=O)O",               # dimethylphosphinic acid
    "thiazolidine_2_4_dione": "CC1SC(=O)NC1=O",    # 5-methylthiazolidine-2,4-dione
    "cyclopentane_1_3_dione": "CC1C(=O)CCC1=O",    # 2-methylcyclopentane-1,3-dione
    "tetramic_acid": "CC1NC(=O)CC1=O",             # 5-methylpyrrolidine-2,4-dione
    "squaramide": "NC1=C(N)C(=O)C1=O",             # the diamino squarate itself
}

# How many capping carbons each fragment carries, subtracted from the heavy-atom
# count so the number describes the group and not the cap.
CAP_CARBONS = {key: 1 for key in CAPPED_SMILES}
CAP_CARBONS["phosphinic_acid"] = 2
CAP_CARBONS["squaramide"] = 0      # the aminos are the group, nothing is a cap

# First ionization, pH units. Midpoints of the ranges in app/headgroups.py,
# which is the single source for the literature basis - keep them in step.
PKA = {
    "carboxylic_acid": 4.5,          # ~4.2 benzoic to ~4.8 acetic
    "tetrazole": 4.7,                # ~4.5-4.9 (5-aryl)
    "hydroxyisoxazole_3": 5.4,       # ~4.8-6
    "hydroxypyrazole_3": 7.5,        # ~6-9, poorly characterized
    "oxadiazol_5_one": 6.5,          # ~6-7
    "thiadiazol_5_one": 6.5,         # ~6-7
    "acylsulfonamide": 4.0,          # ~3.5-4.5
    "hydroxamic_acid": 9.0,          # ~8.5-9.4
    "acyl_cyanamide": 3.5,           # ~3-4
    "primary_sulfonamide": 10.0,     # ~10 (benzenesulfonamide)
    "sulfonic_acid": 0.0,            # ~ -1 to 1, fully ionized at 7.4
    "phosphonic_acid": 1.5,          # first of two (~1.5 and ~6.5)
    "phosphinic_acid": 2.5,          # ~2-3
    "thiazolidine_2_4_dione": 6.8,
    "cyclopentane_1_3_dione": 4.8,   # ~4.3-5.3
    "tetramic_acid": 6.4,
    "squaramide": 8.0,               # ~7-9 for the N-H
}

# Groups whose pKa the headgroup table itself calls poorly characterized or
# uncertain. Kept as data so a run can exclude them and show whether the
# model's behaviour depends on the shakiest numbers.
PKA_UNCERTAIN = ("hydroxypyrazole_3", "squaramide")

PROPERTY_NAMES = ("pka", "heavy_atoms", "hbd", "hba", "tpsa")

_CACHE = {}


def properties(key):
    """The five properties of one headgroup, or None if it has none.

    None is the answer for an absent headgroup - the bare side of a deletion
    pair, labelled "none" - so a caller can count those rows rather than
    silently scoring them on zeros.
    """
    if key in _CACHE:
        return _CACHE[key]
    if key not in CAPPED_SMILES or key not in PKA:
        return None

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")

    molecule = Chem.MolFromSmiles(CAPPED_SMILES[key])
    if molecule is None:
        raise ValueError(f"the capped SMILES for {key!r} does not parse: "
                         f"{CAPPED_SMILES[key]!r}")
    values = {
        "pka": float(PKA[key]),
        "heavy_atoms": float(molecule.GetNumHeavyAtoms() - CAP_CARBONS[key]),
        "hbd": float(Lipinski.NumHDonors(molecule)),
        "hba": float(Lipinski.NumHAcceptors(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
    }
    _CACHE[key] = values
    return values


def property_vector(key):
    """The five properties in PROPERTY_NAMES order, or None."""
    values = properties(key)
    if values is None:
        return None
    return [values[name] for name in PROPERTY_NAMES]


def difference_vector(key_a, key_b):
    """properties(B) - properties(A): the features the baseline regresses on.

    A DIFFERENCE, because the label is a difference. This is what makes the
    baseline antisymmetric when fitted without an intercept: swapping the two
    headgroups negates every feature, so it negates the prediction, which is
    the behaviour Delta p actually has. The dataset contains both directions of
    every pair, so a model that was not antisymmetric would be
    fighting its own training data.

    Returns None when either side has no properties.
    """
    left, right = property_vector(key_a), property_vector(key_b)
    if left is None or right is None:
        return None
    return [b - a for a, b in zip(left, right)]


def missing_headgroups():
    """Headgroups in app/headgroups.py that have no properties here.

    Used by a test: a headgroup added to the project without properties would
    otherwise make the baseline quietly fall back to the global median for
    every pair involving it.
    """
    return sorted(key for key in HEADGROUPS
                  if key not in CAPPED_SMILES or key not in PKA)
