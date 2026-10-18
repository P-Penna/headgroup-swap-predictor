"""Is a pair labelled bioisostere really a headgroup swap?

A pair is labelled from the variable fragment of one cut. If the cut is
large enough to include a headgroup that sits unchanged on BOTH molecules,
that headgroup gets credited as the swap. Example from the real data: A and
B both carry a phosphonic acid; A has a CH2COOH where B has a triazole. A big
cut puts the phosphonic acid in the fragment and calls it acid <-> phosphonic.

The test is counting. In a genuine swap of headgroup X (on A) for Y (on B):
  A has MORE copies of X than B   (X was removed)
  B has MORE copies of Y than A   (Y was added)
Counting rather than presence matters: a diacid where one acid becomes a
tetrazole is a genuine swap even though B still carries an acid.
"""

from rdkit import Chem

from app.headgroups import count_headgroup

GENUINE = "genuine"
NOT_REMOVED = "headgroup_a_not_removed"     # A's group is still there in B
NOT_ADDED = "headgroup_b_not_added"         # B's group was already there in A
NEITHER = "neither_removed_nor_added"
UNREADABLE = "unreadable"


def check_swap(smiles_a, smiles_b, headgroup_a, headgroup_b, mol_cache=None):
    """One of GENUINE, NOT_REMOVED, NOT_ADDED, NEITHER, UNREADABLE."""
    cache = {} if mol_cache is None else mol_cache
    mols = []
    for smiles in (smiles_a, smiles_b):
        if smiles not in cache:
            cache[smiles] = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        mols.append(cache[smiles])
    mol_a, mol_b = mols
    if mol_a is None or mol_b is None or not headgroup_a or not headgroup_b:
        return UNREADABLE
    removed = count_headgroup(mol_a, headgroup_a) > count_headgroup(mol_b, headgroup_a)
    added = count_headgroup(mol_b, headgroup_b) > count_headgroup(mol_a, headgroup_b)
    if removed and added:
        return GENUINE
    if added:
        return NOT_REMOVED
    if removed:
        return NOT_ADDED
    return NEITHER
