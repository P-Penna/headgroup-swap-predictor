"""Tests for app/headgroups.py.

Every headgroup needs at least one POSITIVE (a named real molecule that
really contains the group) and at least one NEAR-MISS NEGATIVE (a molecule
that looks similar and must not match). Unrelated molecules are a weak test;
the negatives here are the things that actually went wrong or could.

The negatives marked "regression" are real over-matches an earlier draft of
the patterns produced on ChEMBL data, before the patterns required the
acidic hydrogen to still be present.
"""

import pytest
from rdkit import Chem, RDLogger

from app.headgroups import (
    HEADGROUPS,
    PATTERNS,
    compile_patterns,
    count_headgroup,
    find_headgroups,
    has_headgroup,
    is_reporting_only,
    matching_forms,
)

RDLogger.DisableLog("rdApp.*")


def mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"test SMILES does not parse: {smiles}"
    return m


# (headgroup key, molecule name, SMILES) - the group IS present
POSITIVES = [
    ("carboxylic_acid", "ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("carboxylic_acid", "aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("carboxylic_acid", "naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("carboxylic_acid", "diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("tetrazole", "losartan (1H tautomer)",
     "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
    ("tetrazole", "valsartan",
     "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1)C(C(C)C)C(=O)O"),
    ("tetrazole", "irbesartan",
     "CCCCC1=NC2(CCCC2)C(=O)N1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"),
    ("acylsulfonamide", "glibenclamide (a sulfonylurea: N-H between C=O and SO2)",
     "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1"),
    ("acylsulfonamide", "N-(methanesulfonyl)benzamide", "CS(=O)(=O)NC(=O)c1ccccc1"),
    ("hydroxamic_acid", "vorinostat", "O=C(NO)CCCCCCC(=O)Nc1ccccc1"),
    ("hydroxamic_acid", "belinostat", "O=C(NO)/C=C/c1cccc(S(=O)(=O)Nc2ccccc2)c1"),
    ("primary_sulfonamide", "furosemide", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl"),
    ("primary_sulfonamide", "acetazolamide", "CC(=O)Nc1nnc(S(N)(=O)=O)s1"),
    ("primary_sulfonamide", "celecoxib",
     "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
    ("sulfonic_acid", "mesna", "OS(=O)(=O)CCS"),
    ("sulfonic_acid", "taurine", "NCCS(=O)(=O)O"),
    ("phosphonic_acid", "alendronate", "NCCCC(O)(P(=O)(O)O)P(=O)(O)O"),
    ("phosphonic_acid", "fosfomycin", "C[C@@H]1O[C@@H]1P(=O)(O)O"),
    ("phosphinic_acid", "fosinopril",
     "CCCCC(=O)OC(P(=O)(O)CCCCc1ccccc1)N1CCCC1C(=O)O"),
    ("phosphinic_acid", "glufosinate", "CP(=O)(O)CC[C@H](N)C(=O)O"),
    ("hydroxyisoxazole_3", "muscimol (oxo tautomer)", "NCc1cc(=O)[nH]o1"),
    ("hydroxyisoxazole_3", "ibotenic acid", "NC(C(=O)O)c1cc(=O)[nH]o1"),
    ("hydroxypyrazole_3", "3-hydroxypyrazole", "Oc1cc[nH]n1"),
    ("oxadiazol_5_one", "azilsartan",
     "CCOc1nc2cccc(C(=O)O)c2n1Cc1ccc(-c2ccccc2-c2noc(=O)[nH]2)cc1"),
    ("thiadiazol_5_one", "3-phenyl-1,2,4-thiadiazol-5(4H)-one (no marketed drug known)",
     "O=c1[nH]c(-c2ccccc2)ns1"),
    ("thiazolidine_2_4_dione", "pioglitazone",
     "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1"),
    ("thiazolidine_2_4_dione", "rosiglitazone",
     "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1"),
    ("cyclopentane_1_3_dione", "cyclopentane-1,3-dione", "O=C1CC(=O)CC1"),
    ("cyclopentane_1_3_dione", "2-methylcyclopentane-1,3-dione", "O=C1C(C)C(=O)CC1"),
    ("cyclopentane_1_3_dione", "enol form (3-hydroxycyclopent-2-enone)", "OC1=CC(=O)CC1"),
    ("tetramic_acid", "pyrrolidine-2,4-dione (tetramic acid core)", "O=C1CC(=O)NC1"),
    ("squaramide", "a ChEMBL squaramide (kinase inhibitor series)",
     "CC(C)(C)Nc1c(Nc2ccnc(Nc3ccc(N4CCOCC4)cc3)n2)c(=O)c1=O"),
    ("acyl_cyanamide", "N-cyanobenzamide", "N#CNC(=O)c1ccccc1"),
]

# (headgroup key, molecule name, SMILES, why it is a near miss) - must NOT match
NEGATIVES = [
    ("carboxylic_acid", "methyl benzoate", "COC(=O)c1ccccc1",
     "ester: the oxygen has no hydrogen"),
    ("carboxylic_acid", "benzamide", "NC(=O)c1ccccc1",
     "amide: nitrogen, not oxygen"),
    ("carboxylic_acid", "benzaldehyde", "O=Cc1ccccc1", "aldehyde: no second oxygen"),
    ("tetrazole", "1-methyl-5-phenyltetrazole", "Cn1nnnc1-c1ccccc1",
     "N-substituted, so there is no acidic hydrogen left"),
    ("tetrazole", "1,2,4-triazole", "c1nc[nH]n1", "only three nitrogens"),
    ("acylsulfonamide", "acetanilide", "CC(=O)Nc1ccccc1",
     "plain amide: no sulfonyl"),
    ("acylsulfonamide", "benzenesulfonamide", "NS(=O)(=O)c1ccccc1",
     "plain sulfonamide: no acyl group"),
    ("acylsulfonamide", "N-methyl-N-(methanesulfonyl)benzamide",
     "CS(=O)(=O)N(C)C(=O)c1ccccc1", "N-alkylated: no acidic hydrogen"),
    ("hydroxamic_acid", "N-methoxybenzamide", "CONC(=O)c1ccccc1",
     "O-methylated, so not acidic"),
    ("hydroxamic_acid", "benzamide", "NC(=O)c1ccccc1", "amide with no N-O bond"),
    ("hydroxamic_acid", "benzohydroxamate O-ester", "CC(=O)ONC(=O)c1ccccc1",
     "the N-O oxygen is acylated, not a free OH"),
    ("primary_sulfonamide", "probenecid", "CCCN(CCC)S(=O)(=O)c1ccc(C(=O)O)cc1",
     "tertiary sulfonamide: no N-H"),
    ("primary_sulfonamide", "N-methylbenzenesulfonamide", "CNS(=O)(=O)c1ccccc1",
     "secondary sulfonamide: only one N-H, not acidic enough to count"),
    ("sulfonic_acid", "methyl benzenesulfonate", "COS(=O)(=O)c1ccccc1",
     "sulfonate ester: the oxygen has no hydrogen"),
    ("sulfonic_acid", "sulfamic acid", "NS(=O)(=O)O",
     "no carbon on sulfur, so not a sulfonic acid"),
    ("phosphonic_acid", "phenyl dihydrogen phosphate", "Oc1ccccc1OP(=O)(O)O",
     "phosphate: oxygen on phosphorus, not carbon"),
    ("phosphonic_acid", "fosinopril (phosphinic)",
     "CCCCC(=O)OC(P(=O)(O)CCCCc1ccccc1)N1CCCC1C(=O)O",
     "phosphinic acid: only one acidic OH on phosphorus"),
    ("phosphinic_acid", "alendronate (phosphonic)", "NCCCC(O)(P(=O)(O)O)P(=O)(O)O",
     "phosphonic acid: two OH and only one carbon on phosphorus"),
    ("hydroxyisoxazole_3", "3-methoxyisoxazole", "COc1ccno1",
     "methyl ether: no acidic hydrogen"),
    ("hydroxyisoxazole_3", "2-methylisoxazol-3(2H)-one", "Cc1cc(=O)n(C)o1",
     "N-methylated oxo form: no acidic hydrogen"),
    ("hydroxyisoxazole_3", "leflunomide", "Cc1oncc1C(=O)Nc1ccc(C(F)(F)F)cc1",
     "isoxazole with no oxygen on the ring carbon next to N"),
    ("hydroxypyrazole_3", "1,2-dimethylpyrazol-3-one", "Cc1cc(=O)n(C)n1C",
     "both nitrogens substituted: no acidic hydrogen"),
    ("hydroxypyrazole_3", "a ChEMBL fused N-substituted pyrazolone (regression)",
     "Cc1cc(-c2c(F)cc3c4nn(CCN(C)C)c(=O)c-4cn(C4CC4)c3c2F)cc(C)n1",
     "an earlier draft counted 248 of these as 3-hydroxypyrazoles"),
    ("oxadiazol_5_one", "ataluren", "OC(=O)c1ccccc1-c1noc(-c2ccc(F)cc2)n1",
     "a 1,2,4-oxadiazole with no 5-one and no N-H"),
    ("thiadiazol_5_one", "3-phenyl-1,2,4-oxadiazol-5(4H)-one", "O=c1[nH]c(-c2ccccc2)no1",
     "the oxygen analogue: ring sulfur is required here"),
    ("thiazolidine_2_4_dione", "rhodanine (2-thioxo-thiazolidin-4-one)", "O=C1CSC(=S)N1",
     "C2 is a thione, not a carbonyl"),
    ("thiazolidine_2_4_dione", "5,5-dimethylthiazolidine-2,4-dione", "O=C1NC(=O)C(C)(C)S1",
     "C5 fully substituted: the acidic C-H is gone"),
    ("cyclopentane_1_3_dione", "PGE2 (regression)",
     "CCCCC[C@H](O)/C=C/[C@H]1[C@H](O)CC(=O)[C@@H]1C/C=C\\CCCC(=O)O",
     "a saturated ring hydroxyl plus a ketone; an earlier draft counted "
     "39 prostaglandins here"),
    ("cyclopentane_1_3_dione", "cyclopentane-1,2-dione", "O=C1CCCC1=O",
     "adjacent carbonyls: no CH between them"),
    ("cyclopentane_1_3_dione", "2,2-dimethylcyclopentane-1,3-dione",
     "O=C1C(C)(C)C(=O)CC1", "no acidic C-H between the carbonyls"),
    ("cyclopentane_1_3_dione", "dimedone (six-membered)", "O=C1CC(C)(C)CC(=O)C1",
     "cyclohexane-1,3-dione, not cyclopentane"),
    ("tetramic_acid", "succinimide", "O=C1CCC(=O)N1",
     "pyrrolidine-2,5-dione: both carbonyls touch the nitrogen"),
    ("tetramic_acid", "glutarimide", "O=C1CCCC(=O)N1", "six-membered 2,6-dione"),
    ("squaramide", "squaric acid dimethyl ester", "COC1=C(OC)C(=O)C1=O",
     "no nitrogen on the ring, so no acidic N-H"),
    ("acyl_cyanamide", "benzonitrile", "N#Cc1ccccc1", "nitrile with no amide"),
    ("acyl_cyanamide", "cimetidine (a cyanoguanidine)",
     "CC1=C(CSCCNC(=NC#N)NC)N=CN1", "cyanoguanidine: no carbonyl"),
    ("acyl_cyanamide", "N-cyano-N-methylbenzamide", "CN(C#N)C(=O)c1ccccc1",
     "N-alkylated: no acidic hydrogen"),
]


# --- the patterns themselves ------------------------------------------------

def test_every_smarts_compiles():
    compiled = compile_patterns()
    assert set(compiled) == set(HEADGROUPS)
    for key, forms in compiled.items():
        assert forms, key
        for form, pattern in forms.items():
            assert pattern is not None, f"{key}/{form}"


def test_every_headgroup_records_a_pka():
    for key, headgroup in HEADGROUPS.items():
        assert headgroup.pka.strip(), key
        assert any(char.isdigit() for char in headgroup.pka), key


def test_the_requested_vocabulary_is_present():
    expected = {
        "carboxylic_acid", "tetrazole", "acylsulfonamide", "hydroxamic_acid",
        "primary_sulfonamide", "sulfonic_acid", "phosphonic_acid", "phosphinic_acid",
        "hydroxyisoxazole_3", "oxadiazol_5_one", "thiadiazol_5_one", "squaramide",
        "thiazolidine_2_4_dione", "cyclopentane_1_3_dione", "tetramic_acid",
        "hydroxypyrazole_3", "acyl_cyanamide",
    }
    assert set(HEADGROUPS) == expected


# --- positives and negatives ------------------------------------------------

@pytest.mark.parametrize("key,name,smiles", POSITIVES, ids=[f"{k}-{n}" for k, n, _ in POSITIVES])
def test_positive_matches(key, name, smiles):
    assert has_headgroup(mol(smiles), key), f"{name} should contain {key}"


@pytest.mark.parametrize("key,name,smiles,why", NEGATIVES,
                         ids=[f"{k}-{n}" for k, n, _, _ in NEGATIVES])
def test_negative_does_not_match(key, name, smiles, why):
    assert not has_headgroup(mol(smiles), key), f"{name} must NOT match {key} ({why})"


def test_every_headgroup_has_a_positive_and_a_negative_test():
    """Guard: a new headgroup added without tests should fail here."""
    with_positive = {key for key, _, _ in POSITIVES}
    with_negative = {key for key, _, _, _ in NEGATIVES}
    assert set(HEADGROUPS) - with_positive == set()
    assert set(HEADGROUPS) - with_negative == set()


# --- pitfall 1: protonation state -------------------------------------------

@pytest.mark.parametrize("key,neutral,anion", [
    ("carboxylic_acid", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "CC(C)Cc1ccc(C(C)C(=O)[O-])cc1"),
    ("tetrazole", "c1ccc(-c2nnn[nH]2)cc1", "c1ccc(-c2nnn[n-]2)cc1"),
    ("sulfonic_acid", "OS(=O)(=O)c1ccccc1", "[O-]S(=O)(=O)c1ccccc1"),
    ("acylsulfonamide", "CS(=O)(=O)NC(=O)c1ccccc1", "CS(=O)(=O)[N-]C(=O)c1ccccc1"),
    ("primary_sulfonamide", "NS(=O)(=O)c1ccccc1", "[NH-]S(=O)(=O)c1ccccc1"),
])
def test_both_protonation_states_match(key, neutral, anion):
    """ChEMBL stores some structures already deprotonated; both must be found."""
    assert has_headgroup(mol(neutral), key), f"{key}: neutral form missed"
    assert has_headgroup(mol(anion), key), f"{key}: anion form missed"


def membership_forms(smiles, key):
    """Matching forms, with the reporting-only ones removed."""
    return [f for f in matching_forms(mol(smiles), key) if not is_reporting_only(key, f)]


def test_the_two_protonation_states_are_reported_as_different_forms():
    assert membership_forms("CC(=O)O", "carboxylic_acid") == ["neutral"]
    assert membership_forms("CC(=O)[O-]", "carboxylic_acid") == ["anion"]


# --- pitfall 2: tautomers ---------------------------------------------------

def test_both_tetrazole_tautomers_match_and_are_told_apart():
    one_h = mol("c1ccc(-c2nnn[nH]2)cc1")
    two_h = mol("c1ccc(-c2nn[nH]n2)cc1")
    # RDKit does not canonicalize tautomers, so these really are different molecules.
    assert Chem.MolToSmiles(one_h) != Chem.MolToSmiles(two_h)
    assert has_headgroup(one_h, "tetrazole")
    assert has_headgroup(two_h, "tetrazole")
    assert "NH_adjacent_1H" in matching_forms(one_h, "tetrazole")
    assert "NH_distal_2H" in matching_forms(two_h, "tetrazole")


@pytest.mark.parametrize("key,hydroxy,oxo", [
    ("hydroxyisoxazole_3", "NCc1cc(O)no1", "NCc1cc(=O)[nH]o1"),
    ("hydroxypyrazole_3", "Oc1cc[nH]n1", "O=c1cc[nH][nH]1"),
    ("oxadiazol_5_one", "Oc1nc(-c2ccccc2)no1", "O=c1[nH]c(-c2ccccc2)no1"),
])
def test_both_hydroxy_and_oxo_tautomers_match(key, hydroxy, oxo):
    assert has_headgroup(mol(hydroxy), key), f"{key}: hydroxy tautomer missed"
    assert has_headgroup(mol(oxo), key), f"{key}: oxo tautomer missed"


def test_the_dione_enol_tautomer_matches():
    assert "diketo" in matching_forms(mol("O=C1CC(=O)CC1"), "cyclopentane_1_3_dione")
    assert "enol" in matching_forms(mol("OC1=CC(=O)CC1"), "cyclopentane_1_3_dione")


# --- pitfall 3: aromaticity perception --------------------------------------

@pytest.mark.parametrize("key,aliphatic,aromatic", [
    ("oxadiazol_5_one", "O=C1NC(=NO1)c1ccccc1", "O=c1[nH]c(-c2ccccc2)no1"),
    ("thiadiazol_5_one", "O=C1NC(=NS1)c1ccccc1", "O=c1[nH]c(-c2ccccc2)ns1"),
    ("tetrazole", "C1=NNN=N1", "c1nn[nH]n1"),
])
def test_matches_whichever_way_the_ring_was_written(key, aliphatic, aromatic):
    """RDKit re-perceives aromaticity on parsing, and for these rings it returns
    the aromatic form either way. The patterns use ~ bonds so both still match."""
    assert has_headgroup(mol(aliphatic), key), f"{key}: aliphatic input missed"
    assert has_headgroup(mol(aromatic), key), f"{key}: aromatic input missed"


def test_aromaticity_sensitive_rings_use_any_bond_queries():
    """The rings RDKit may perceive as aromatic must not pin bond orders."""
    for key, headgroup in HEADGROUPS.items():
        if headgroup.aromaticity_sensitive:
            for form, smarts in headgroup.forms.items():
                ring_part = smarts[smarts.index("1"):] if "1" in smarts else smarts
                assert "=" not in ring_part.replace("=,:", ""), f"{key}/{form}: {smarts}"


def test_non_aromatic_carbon_acids_do_pin_bond_orders():
    """These three must pin bond orders, or saturated look-alikes slip in
    (the PGE2 regression). Guard so nobody loosens them back."""
    for key in ["thiazolidine_2_4_dione", "cyclopentane_1_3_dione", "tetramic_acid"]:
        assert not HEADGROUPS[key].aromaticity_sensitive, key
        assert "=[OX1]" in HEADGROUPS[key].forms["diketo"], key


# --- the whole vocabulary together ------------------------------------------

def test_find_headgroups_reports_every_group_in_a_molecule():
    # Azilsartan has both a carboxylic acid and an oxadiazolone.
    found = find_headgroups(mol(
        "CCOc1nc2cccc(C(=O)O)c2n1Cc1ccc(-c2ccccc2-c2noc(=O)[nH]2)cc1"))
    assert found == ["carboxylic_acid", "oxadiazol_5_one"]
    # Furosemide has a carboxylic acid and a primary sulfonamide.
    found = find_headgroups(mol("NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl"))
    assert found == ["carboxylic_acid", "primary_sulfonamide"]


def test_a_molecule_with_no_acidic_group_finds_nothing():
    for name, smiles in [("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
                         ("benzene", "c1ccccc1"),
                         ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
                         ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1")]:
        found = find_headgroups(mol(smiles))
        assert found == [], f"{name} should have no headgroup, found {found}"


def test_extra_forms_do_not_make_a_molecule_count_as_having_the_group():
    """1-methyltetrazole matches the skeleton pattern but is not acidic."""
    methyl_tetrazole = mol("Cn1nnnc1-c1ccccc1")
    assert "skeleton" in matching_forms(methyl_tetrazole, "tetrazole")
    assert not has_headgroup(methyl_tetrazole, "tetrazole")


def test_compiled_patterns_cover_every_declared_form():
    assert set(PATTERNS) == set(HEADGROUPS)
    for key, headgroup in HEADGROUPS.items():
        assert set(PATTERNS[key]) == set(headgroup.patterns()), key


# --- every anion form must be reachable (regression) -------------------------
#
# A pattern that can never match is silent: it just quietly contributes zero.
# [NX3H0-] was exactly that bug - deprotonating drops the connection count to
# X2, so the anion forms of acylsulfonamide, primary sulfonamide, hydroxamic
# acid and acyl cyanamide matched nothing at all until this was caught.
# One deprotonated example per anion form, with no exceptions allowed.

ANION_EXAMPLES = {
    ("carboxylic_acid", "anion"): "CC(=O)[O-]",
    ("tetrazole", "anion_adjacent"): "c1ccc(-c2nnn[n-]2)cc1",
    ("tetrazole", "anion_distal"): "c1ccc(-c2nn[n-]n2)cc1",
    ("hydroxyisoxazole_3", "anion"): "NCc1cc([O-])no1",
    ("hydroxypyrazole_3", "anion"): "[O-]c1cc[nH]n1",
    ("oxadiazol_5_one", "anion"): "O=c1[n-]c(-c2ccccc2)no1",
    ("thiadiazol_5_one", "anion"): "O=c1[n-]c(-c2ccccc2)ns1",
    ("acylsulfonamide", "anion"): "CS(=O)(=O)[N-]C(=O)c1ccccc1",
    ("hydroxamic_acid", "O_anion"): "[O-]NC(=O)c1ccccc1",
    ("hydroxamic_acid", "N_anion"): "O[N-]C(=O)c1ccccc1",
    ("acyl_cyanamide", "anion"): "N#C[N-]C(=O)c1ccccc1",
    ("primary_sulfonamide", "anion"): "[NH-]S(=O)(=O)c1ccccc1",
    ("sulfonic_acid", "anion"): "[O-]S(=O)(=O)c1ccccc1",
    ("phosphonic_acid", "neutral_or_anion"): "[O-]P(=O)([O-])CCc1ccccc1",
    ("phosphinic_acid", "neutral_or_anion"): "CP(=O)([O-])CC[C@H](N)C(=O)O",
    ("squaramide", "anion"): "CN[C]1=C([N-]C)C(=O)C1=O",
    ("thiazolidine_2_4_dione", "enolate"): "O=C1NC([O-])=CS1",
    ("cyclopentane_1_3_dione", "enolate"): "O=C1CC([O-])=CC1",
    ("tetramic_acid", "enolate"): "O=C1NC(=O)C=C1[O-]",
}


@pytest.mark.parametrize("key,form,smiles",
                         [(k, f, smi) for (k, f), smi in ANION_EXAMPLES.items()],
                         ids=[f"{k}-{f}" for k, f in ANION_EXAMPLES])
def test_each_anion_form_matches_a_deprotonated_example(key, form, smiles):
    assert form in matching_forms(mol(smiles), key), f"{key}/{form} matched nothing"


def test_every_anion_form_has_an_example():
    """Guard: a new anion form added without an example fails here."""
    declared = {(key, form) for key, headgroup in HEADGROUPS.items()
                for form in headgroup.forms
                if "anion" in form or "enolate" in form}
    assert declared - set(ANION_EXAMPLES) == set()


# --- the skeleton upper bound -------------------------------------------------
#
# Every group has a reporting-only `skeleton`: heavy-atom connectivity with
# hydrogens, charges and bond orders forgotten. It bounds what the real
# patterns could be missing, so a headgroup written in an unanticipated
# tautomer shows up as a gap instead of silently shrinking a count. For that
# to mean anything the skeleton must be a genuine SUPERSET of every
# membership form - that is the invariant tested here.

def test_every_group_has_a_skeleton():
    for key, headgroup in HEADGROUPS.items():
        assert "skeleton" in headgroup.extra_forms, key
        assert "skeleton" not in headgroup.forms, f"{key}: skeleton must not decide membership"


@pytest.mark.parametrize("key,name,smiles", POSITIVES, ids=[f"{k}-{n}" for k, n, _ in POSITIVES])
def test_skeleton_also_matches_every_positive(key, name, smiles):
    """If the skeleton missed a real positive it would not be an upper bound."""
    assert "skeleton" in matching_forms(mol(smiles), key), \
        f"{name}: skeleton misses a molecule the acidic forms match"


@pytest.mark.parametrize("key,form,smiles",
                         [(k, f, smi) for (k, f), smi in ANION_EXAMPLES.items()],
                         ids=[f"{k}-{f}" for k, f in ANION_EXAMPLES])
def test_skeleton_also_matches_every_anion(key, form, smiles):
    assert "skeleton" in matching_forms(mol(smiles), key), f"{key}/{form} escapes the skeleton"


def test_skeleton_is_looser_than_the_acidic_forms():
    """Spot-check that the skeleton really is the generous version: it is
    expected to match things that are NOT headgroups."""
    # An ester has the carboxylic acid skeleton but is not an acid.
    ester = mol("COC(=O)c1ccccc1")
    assert "skeleton" in matching_forms(ester, "carboxylic_acid")
    assert not has_headgroup(ester, "carboxylic_acid")
    # PGE2 has the 1,3-dione skeleton (this was the original bug) but is not one.
    pge2 = mol("CCCCC[C@H](O)/C=C/[C@H]1[C@H](O)CC(=O)[C@@H]1C/C=C\\CCCC(=O)O")
    assert "skeleton" in matching_forms(pge2, "cyclopentane_1_3_dione")
    assert not has_headgroup(pge2, "cyclopentane_1_3_dione")


# --- sub-forms -------------------------------------------------

def test_indane_dione_is_counted_and_labelled():
    """Indane-1,3-diones stay in the headgroup but are reported separately."""
    indandione = mol("O=C1c2ccccc2C(=O)C1c1ccncc1")      # 2-aryl indane-1,3-dione
    assert has_headgroup(indandione, "cyclopentane_1_3_dione")
    assert "sub_indane_diketo" in matching_forms(indandione, "cyclopentane_1_3_dione")
    indandione_enol = mol("C/C(=N\\C)C1=C(O)c2ccccc2C1=O")
    assert has_headgroup(indandione_enol, "cyclopentane_1_3_dione")
    assert "sub_indane_enol" in matching_forms(indandione_enol, "cyclopentane_1_3_dione")


def test_plain_cyclopentane_dione_is_not_labelled_as_indane():
    forms = matching_forms(mol("O=C1CC(=O)CC1"), "cyclopentane_1_3_dione")
    assert "diketo" in forms
    assert not [f for f in forms if f.startswith("sub_")]


def test_n_substituted_pyrazolone_is_counted_and_labelled():
    """Keeps an N-H, so it is still an acid and still counted."""
    pyrazolone = mol("Cc1[nH]n(-c2ccccc2)c(=O)c1")
    assert has_headgroup(pyrazolone, "hydroxypyrazole_3")
    assert "sub_N_substituted" in matching_forms(pyrazolone, "hydroxypyrazole_3")


def test_fully_n_substituted_pyrazolone_is_not_in_the_sub_form_either():
    """Regression: the sub-form must require the acidic hydrogen too. Without
    that it matched the non-acids as well, giving 160% of its own headgroup."""
    both_substituted = mol("Cc1cc(=O)n(C)n1C")
    assert "sub_N_substituted" not in matching_forms(both_substituted, "hydroxypyrazole_3")


def test_plain_hydroxypyrazole_is_not_labelled_as_n_substituted():
    forms = matching_forms(mol("Oc1cc[nH]n1"), "hydroxypyrazole_3")
    assert "hydroxy_tautomer" in forms
    assert not [f for f in forms if f.startswith("sub_")]


def test_fully_n_substituted_pyrazolone_is_still_excluded_entirely():
    """The sub-form must not sneak the non-acids back in."""
    both_substituted = mol("Cc1cc(=O)n(C)n1C")
    assert not has_headgroup(both_substituted, "hydroxypyrazole_3")


def test_reporting_only_is_decided_by_the_dict_not_the_name():
    """A sub_ form in `forms` adds population; one in `extra_forms` only labels."""
    for key, headgroup in HEADGROUPS.items():
        for form in headgroup.forms:
            assert not is_reporting_only(key, form), f"{key}/{form}"
        for form in headgroup.extra_forms:
            assert is_reporting_only(key, form), f"{key}/{form}"
    assert not is_reporting_only("thiazolidine_2_4_dione", "sub_arylidene")
    assert is_reporting_only("cyclopentane_1_3_dione", "sub_indane_enol")


# --- sub-forms that ADD population ------------------------------

def test_arylidene_tzd_is_counted_through_its_n_h():
    """5-benzylidene TZD has no acidic C5-H left, but the N3-H is still there
    and the ring is unchanged, so it is the same headgroup."""
    arylidene = mol("COc1ccc(COc2ccc(/C=C3\\SC(=O)NC3=O)cc2)cc1")
    assert has_headgroup(arylidene, "thiazolidine_2_4_dione")
    assert "sub_arylidene" in matching_forms(arylidene, "thiazolidine_2_4_dione")
    # The C5-H forms genuinely do not reach it; sub_arylidene is what counts it.
    assert "diketo" not in matching_forms(arylidene, "thiazolidine_2_4_dione")


def test_n_substituted_arylidene_tzd_has_no_acid_left_and_is_excluded():
    """N3 alkylated AND C5 arylidene: both acidic hydrogens gone (680 parents)."""
    both_gone = mol("COc1cc(N2CCCC2)c(OC)cc1/C=C1\\SC(=O)N(C)C1=O")
    assert not has_headgroup(both_gone, "thiazolidine_2_4_dione")


def test_tzd_iminol_tautomer_is_counted():
    """The N3 proton moved onto the C2 oxygen: same compound, same acid.
    Found by sampling the skeleton gap, not by guessing tautomers up front."""
    iminol = mol("O=C1N=C(O)S/C1=C\\c1ccc(OCC2CCCCC2)cc1")
    assert has_headgroup(iminol, "thiazolidine_2_4_dione")
    assert "iminol_tautomer" in matching_forms(iminol, "thiazolidine_2_4_dione")


def test_rhodanine_still_excluded_after_adding_the_iminol_form():
    """Guard: the iminol pattern must not drag in the 2-thioxo analogue."""
    assert not has_headgroup(mol("O=C1CSC(=S)N1"), "thiazolidine_2_4_dione")


def test_plain_tzd_is_not_labelled_arylidene():
    assert "sub_arylidene" not in matching_forms(
        mol("CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1"), "thiazolidine_2_4_dione")


def test_three_acyl_tetramic_acid_is_counted():
    """The exocyclic enol of a 3-acyl tetramic acid: same 1,3-dicarbonyl acid,
    more acidic than the plain group, but the in-ring enol pattern misses it."""
    acyl = mol("CC/C(O)=C1/C(=O)N[C@H](C)C1=O")
    assert has_headgroup(acyl, "tetramic_acid")
    assert "sub_acyl" in matching_forms(acyl, "tetramic_acid")


def test_two_acyl_cyclopentane_dione_is_counted():
    acyl = mol("CC/C(O)=C1/C(=O)CCC1=O")
    assert has_headgroup(acyl, "cyclopentane_1_3_dione")
    assert "sub_acyl" in matching_forms(acyl, "cyclopentane_1_3_dione")


def test_isoxazolidin_3_one_is_deliberately_excluded():
    """A saturated N-O ring is a cyclic hydroxamic acid, not a
    3-hydroxyisoxazole: different pKa and binding mode."""
    saturated = mol("O=C1NOC[C@H]1N")
    assert not has_headgroup(saturated, "hydroxyisoxazole_3")
    # while the aromatic oxo tautomer IS one
    assert has_headgroup(mol("NCc1cc(=O)[nH]o1"), "hydroxyisoxazole_3")


def test_every_sub_form_is_a_subset_of_its_headgroup():
    """INVARIANT: a sub-form names a slice of the headgroup, so anything it
    matches must also BE the headgroup. Violated once (see the regression
    above), and it would show up in the counts as a sub-form larger than the
    group that contains it."""
    probes = [smiles for _, _, smiles in POSITIVES]
    probes += [smiles for _, _, smiles, _ in NEGATIVES]
    probes += list(ANION_EXAMPLES.values())
    probes += ["Cc1cc(=O)n(C)n1C", "O=C1c2ccccc2C(=O)C1c1ccncc1", "O=C1CC(=O)CC1",
               "COC1=C(OC)C(=O)C1=O", "Cn1nnnc1-c1ccccc1"]
    for smiles in probes:
        molecule = mol(smiles)
        for key, headgroup in HEADGROUPS.items():
            for form in headgroup.extra_forms:   # membership sub-forms are exempt:
                if not form.startswith("sub_"):  # they define population, not a slice
                    continue
                if molecule.HasSubstructMatch(PATTERNS[key][form]):  # extra_forms only
                    assert has_headgroup(molecule, key), (
                        f"{smiles}: matches {key}/{form} but is not counted as {key}")


def test_acyl_cyanamide_skeleton_excludes_plain_ureas():
    """Regression: the skeleton once matched any N-C-N-C=O, so every urea in
    ChEMBL (79,677) landed in the gap and hid anything real."""
    urea = mol("CNC(=O)NC")
    assert "skeleton" not in matching_forms(urea, "acyl_cyanamide")
    assert "skeleton" in matching_forms(mol("N#CNC(=O)c1ccccc1"), "acyl_cyanamide")


# --- counting copies, which separates a real swap from a mislabel ------------

@pytest.mark.parametrize("smiles, key, expected", [
    ("OC(=O)c1ccccc1", "carboxylic_acid", 1),
    ("OC(=O)c1ccc(C(=O)O)cc1", "carboxylic_acid", 2),          # terephthalic acid
    ("OC(=O)C(=O)O", "carboxylic_acid", 2),                    # oxalic acid
    ("OC(=O)CC(C(=O)O)", "carboxylic_acid", 2),                # malonic acid
    ("c1ccccc1-c1nnn[nH]1", "tetrazole", 1),                   # 1H form
    ("c1ccccc1-c1nn[nH]n1", "tetrazole", 1),                   # 2H form
    ("c1ccc(-c2nnn[nH]2)cc1-c1nn[nH]n1", "tetrazole", 2),      # one of each
    ("CCOC(=O)c1ccccc1", "carboxylic_acid", 0),                # ester is not an acid
    ("c1ccccc1-c1nnnn1C", "tetrazole", 0),                     # N-methyl: no acidic H
])
def test_count_headgroup(smiles, key, expected):
    assert count_headgroup(mol(smiles), key) == expected
