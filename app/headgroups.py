"""Acidic headgroups (carboxylic acid and its bioisosteres) as RDKit SMARTS.

A "headgroup" here is the acidic group a medicinal chemist swaps when they
replace a carboxylic acid with something that is still acidic but has
different permeability, metabolic stability or selectivity. This module is
the vocabulary: which groups we recognize and how we find them.

THE pKa VALUES ARE APPROXIMATE AND FROM MEMORY
----------------------------------------------
Each group carries a rough literature pKa, because that is why the group is
in this list: these are all acids of roughly comparable strength, so swapping
one for another keeps the molecule ionized at pH 7.4. The numbers are
order-of-magnitude guides for step 3.1, NOT measured values, and they were
written down from memory, not read off a paper. Verify any number before it
goes in the writeup. Values depend strongly on substitution: an
electron-poor aryl tetrazole can be a whole unit stronger than an alkyl one.

FOUR THINGS THAT SILENTLY BREAK PATTERNS LIKE THESE
----------------------------------------------------
1. PROTONATION STATE. ChEMBL stores some structures already deprotonated
   (C(=O)[O-]). A pattern written only for the neutral acid misses them
   silently. Every group that can ionize has an anion form here, and
   03_find_headgroups.py reports the forms separately so the split stays
   visible.

   A trap inside the trap, which the tests caught: losing a proton also
   lowers the atom's CONNECTION count, so X constraints copied from the
   neutral pattern silently never match. An [N-] bonded to a carbonyl and a
   sulfonyl has two connections (X2), not three, so the "obvious" [NX3H0-]
   is dead code. Anion patterns here therefore constrain only the element,
   hydrogen count and charge ([#7;H0;-1]) and never X.

2. TAUTOMERS. Tetrazole is written both 1H (c1nnn[nH]1) and 2H
   (c1nn[nH]n1) in ChEMBL, and RDKit keeps them as different molecules: it
   does NOT canonicalize tautomers. Hydroxy-heterocycles such as
   3-hydroxyisoxazole appear both as the hydroxy form (Oc1ccno1) and as the
   oxo form (O=c1cc[nH]o1). A pattern pinned to one tautomer misses the
   other, so each tautomer is a separate named form below.

3. AROMATICITY PERCEPTION. RDKit decides aromaticity itself when it parses
   a SMILES, and for these rings the answer is not what a textbook drawing
   suggests. Checked on RDKit in this environment:
     - 1,2,4-oxadiazol-5(4H)-one and the thiadiazolone are perceived
       AROMATIC even when written aliphatically: both "O=C1NC(=NO1)c1ccccc1"
       and "O=c1[nH]c(-c2ccccc2)no1" come back as the aromatic form.
     - A squaramide comes back PARTLY aromatic (Nc1c(N)c(=O)c1=O): 4 of its
       atoms are flagged aromatic.
     - Tetrazole is aromatic even from the aliphatic input "C1=NNN=N1".
     - Thiazolidine-2,4-dione, tetramic acid and cyclopentane-1,3-dione are
       NOT aromatic, from any input style tried.
   So a lowercase (aromatic) SMARTS and an uppercase (aliphatic) one would
   each miss part of the data. The rings in the first group above are
   therefore written with element queries ([#6], [#7]) and "any bond" (~),
   which match whatever RDKit decided. The three rings RDKit never calls
   aromatic DO specify bond orders - see pitfall 4 for why they must.

4. THE ACIDIC HYDROGEN MUST STILL BE THERE. This one bit us. A first draft
   described only the ring skeleton, and on 40,000 real structures it
   counted 248 N-substituted pyrazolones as 3-hydroxypyrazoles. If the
   nitrogen that should carry the acidic hydrogen is alkylated, the group is
   not an acid any more and must not be counted. So every pattern below
   pins the ionizable hydrogen: an -OH, a ring N-H, an acidic C-H, or the
   deprotonated anion. Where that hydrogen can sit in more than one place
   (tetrazole 1H vs 2H), each position is a separate named form.

   The same draft also counted 39 prostaglandins as cyclopentane-1,3-diones:
   with "any bond", a saturated ring carrying a ketone and a SEPARATE
   hydroxyl (PGE2's C(O)-C-C(=O)) looks exactly like the enol of a 1,3-dione.
   The non-aromatic carbon acids therefore specify bond orders, which is
   safe because RDKit never perceives those rings as aromatic.

Acyclic groups (the acid, the sulfonamides, the phosphorus acids) have no
aromaticity question, so they use ordinary explicit SMARTS with X (total
connections) and H (hydrogen count) constraints.

THE "skeleton" FORM: AN UPPER BOUND ON EVERY GROUP
--------------------------------------------------
Every group carries a reporting-only `skeleton` form: the heavy-atom
connectivity with hydrogens, charges and bond orders all forgotten. It is
deliberately too generous - the carboxylic acid skeleton matches esters too -
and it never decides membership.

Its job is to bound what the real patterns could be missing. A headgroup
written in a tautomer or charge state nobody anticipated would otherwise just
make a count slightly smaller, with nothing to notice. With a skeleton, it
shows up as a gap between the skeleton count and the acidic forms, which is a
place to look. The gap always contains genuine non-acids as well (esters for
the acid, N-alkylated rings for the heterocycles), so a gap is a prompt to
sample it, not a bug count. Tetrazole is the worked example: 9,047 rings,
3,109 acidic, and the 5,938 in the gap were sampled by hand and are all
N-substituted, so nothing is missing there.

WHAT COUNTS AS THE SAME HEADGROUP (the rule)
---------------------------------------------------------
A group is INCLUDED when the ionizable proton is still present and the
acidic motif is intact, even if the proton has MOVED to a different atom
inside the same group. A 5-arylidene thiazolidine-2,4-dione has lost its
acidic C5-H to the exocyclic double bond, but the N3-H is still there and
the ring is still a thiazolidine-2,4-dione, so it counts.

A group is EXCLUDED when the substitution removes the acid outright
(alkylating the only acidic nitrogen, methylating the only acidic OH), or
when the scaffold is really a different functional class.

Deliberate exclusion, so nobody re-adds it: saturated ISOXAZOLIDIN-3-ONES
are NOT 3-hydroxyisoxazoles. A saturated ring with an N-O bond next to the
carbonyl is a cyclic hydroxamic acid; its pKa and its binding mode differ
enough that counting it here would be wrong. It is kept out by requiring
the C4-C5 bond to be double or aromatic. 10 parents in ChEMBL 37 are
affected (29 match the saturated ring, 19 of which also carry a real
aromatic 3-hydroxyisoxazole elsewhere and are counted for that).

SUB-FORMS
---------
A `sub_` form names a structurally distinct slice of a headgroup. Whether
it decides membership depends only on which dict it is in:

  in `forms`        it ADDS population the other patterns do not reach.
                    sub_arylidene (thiazolidine-2,4-dione) and sub_acyl
                    (tetramic acid, cyclopentane-1,3-dione) are here: they
                    were being missed entirely before.
  in `extra_forms`  the slice is ALREADY counted by the patterns above and
                    is only being labelled. sub_indane_* and
                    sub_N_substituted are here, and each must be a strict
                    subset of its headgroup (there is a test).

Either way the point is the same: 1.5 can ask whether swaps involving a
fused ring, an N-substituent or an arylidene behave differently, instead of
having them silently mixed in.

[O;X1,X2H1] is used throughout for an oxygen that may be a carbonyl (=O,
one connection), an anion ([O-], one connection) or a hydroxyl (-OH, two
connections, one hydrogen). It deliberately does NOT match an ether or
ester oxygen (two connections, no hydrogen), which is what keeps
3-methoxyisoxazole out of the 3-hydroxyisoxazole count.
"""

from dataclasses import dataclass, field

from rdkit import Chem


@dataclass(frozen=True)
class Headgroup:
    """One acidic group, and the SMARTS that find it.

    forms: name -> SMARTS. A molecule has the group if ANY form matches.
           Forms are kept separate so counts per form can be reported
           (how many molecules are stored deprotonated, which tautomer, ...).
    extra_forms: reported alongside, but never used to decide membership.
    pka:   approximate literature pKa, from memory. See the module docstring.
    """

    name: str
    pka: str
    forms: dict
    notes: str = ""
    aromaticity_sensitive: bool = False
    extra_forms: dict = field(default_factory=dict)

    def patterns(self):
        """forms plus extra_forms."""
        return {**self.forms, **self.extra_forms}


# An oxygen that is a carbonyl, an anion, or a hydroxyl - but never an ether/ester.
ACID_O = "[O;X1,X2H1]"

HEADGROUPS = {
    # ---------------------------------------------------------------- the acid
    "carboxylic_acid": Headgroup(
        name="carboxylic acid",
        pka="~4.2 (benzoic) to ~4.8 (acetic)",
        forms={
            # The acid carbon must carry a carbon, which keeps carbamic acid
            # (N-COOH) and carbonic acid / carbonate fragments out.
            "neutral": "[#6][CX3](=[OX1])[OX2H1]",
            "anion": "[#6][CX3](=[OX1])[OX1H0-]",
        },
        extra_forms={
            # Also matches esters and anhydrides: an upper bound, not a count.
            "skeleton": "[#6][CX3](~[#8])~[#8]",
        },
        notes="An ester is excluded because its oxygen has no hydrogen (OX2H0); "
              "an amide because the heteroatom is nitrogen.",
    ),

    # ------------------------------------------------- five-membered aromatics
    "tetrazole": Headgroup(
        name="tetrazole (5-substituted 1H/2H-tetrazole)",
        pka="~4.5-4.9 (5-aryl)",
        forms={
            # C + 4 N in a 5-ring. The acidic H sits on a different nitrogen in
            # the 1H and 2H tautomers, so each position is its own form. An
            # N-substituted tetrazole (1-methyltetrazole, common in cephalosporin
            # side chains) has no acidic hydrogen and matches none of these.
            "NH_adjacent_1H": "[#6]1~[#7;H1]~[#7]~[#7]~[#7]~1",
            "NH_distal_2H": "[#6]1~[#7]~[#7;H1]~[#7]~[#7]~1",
            "anion_adjacent": "[#6]1~[#7;H0;-1]~[#7]~[#7]~[#7]~1",
            "anion_distal": "[#6]1~[#7]~[#7;H0;-1]~[#7]~[#7]~1",
        },
        extra_forms={
            # The bare ring, acidic or not. The gap between this and the forms
            # above is the N-substituted tetrazoles (sampled: all genuinely so).
            "skeleton": "[#6]1~[#7]~[#7]~[#7]~[#7]~1",
        },
        notes="The classic -COOH replacement: same pKa, flat, but more lipophilic "
              "and not glucuronidated the same way.",
        aromaticity_sensitive=True,
    ),

    "hydroxyisoxazole_3": Headgroup(
        name="3-hydroxyisoxazole",
        pka="~4.8-6 (the OH; muscimol's acidic OH is ~4.8)",
        forms={
            # Isoxazole ring O1-N2-C3-C4-C5 with an acidic O on C3 (next to N2).
            # The C4-C5 bond must be double or aromatic (=,:), which keeps the
            # saturated isoxazolidine-3,5-diones out: those are a different group.
            "hydroxy_tautomer": "[OX2H1]~[#6]1~[#7]~[#8]~[#6]=,:[#6]~1",
            "oxo_tautomer_NH": "[OX1]~[#6]1~[#7;H1]~[#8]~[#6]=,:[#6]~1",
            "anion": "[OX1;H0;-1]~[#6]1~[#7]~[#8]~[#6]=,:[#6]~1",
        },
        extra_forms={
            "skeleton": "[#8]~[#6]1~[#7]~[#8]~[#6]~[#6]~1",
        },
        notes="Matches both the hydroxy tautomer (Oc1ccno1) and the oxo tautomer "
              "(O=c1cc[nH]o1). 3-methoxyisoxazole is excluded (no acidic hydrogen), "
              "and so is an N-methylated oxo ring.",
        aromaticity_sensitive=True,
    ),

    "hydroxypyrazole_3": Headgroup(
        name="3-hydroxypyrazole",
        pka="~6-9, poorly characterized - treat as uncertain",
        forms={
            # Pyrazole ring N1-N2-C3-C4-C5 with an acidic O on C3 (next to N2).
            # Either the exocyclic OH or one of the two ring N-H is the acid, so
            # an N,N-substituted pyrazolone matches none of these.
            "hydroxy_tautomer": "[OX2H1]~[#6]1~[#7]~[#7]~[#6]=,:[#6]~1",
            "oxo_tautomer_NH_adjacent": "[OX1]~[#6]1~[#7;H1]~[#7]~[#6]=,:[#6]~1",
            "oxo_tautomer_NH_distal": "[OX1]~[#6]1~[#7]~[#7;H1]~[#6]=,:[#6]~1",
            "anion": "[OX1;H0;-1]~[#6]1~[#7]~[#7]~[#6]=,:[#6]~1",
        },
        extra_forms={
            "skeleton": "[#8]~[#6]1~[#7]~[#7]~[#6]~[#6]~1",
            # Sub-form: still counted as the headgroup by the forms
            # above, but named so the slice is visible. One ring nitrogen carries
            # a substituent (X3, no hydrogen) AND the group still keeps an acidic
            # hydrogen - either the exocyclic OH, or the other ring N-H. Both
            # halves are required, or this would also match the fully substituted
            # pyrazolones that are not acids at all (it did, and the count came
            # out at 160% of the headgroup, which is how the bug was spotted).
            # Four alternatives, because the exocyclic oxygen fixes which ring
            # nitrogen is which: OH with either N substituted, or =O with one
            # N-H and the other substituted.
            "sub_N_substituted": (
                "[$([OX2H1]~[#6]1~[#7;H0;X3]~[#7]~[#6]=,:[#6]~1),"
                "$([OX2H1]~[#6]1~[#7]~[#7;H0;X3]~[#6]=,:[#6]~1),"
                "$([OX1]~[#6]1~[#7;H1]~[#7;H0;X3]~[#6]=,:[#6]~1),"
                "$([OX1]~[#6]1~[#7;H0;X3]~[#7;H1]~[#6]=,:[#6]~1)]"
            ),
        },
        notes="Weaker and less used than the isoxazole, and the pKa depends "
              "heavily on substitution; verify before use. A pyrazolone with ALL "
              "its nitrogens substituted has no acidic hydrogen and is excluded "
              "(see pitfall 4); one that keeps an N-H is counted, and also "
              "reported under the sub_N_substituted_* forms.",
        aromaticity_sensitive=True,
    ),

    "oxadiazol_5_one": Headgroup(
        name="1,2,4-oxadiazol-5(4H)-one",
        pka="~6-7",
        forms={
            # Ring C3-N2-O1-C5(=O)-N4. The carbonyl sits between the ring oxygen
            # and the ring nitrogen, and the acidic hydrogen is the N4-H.
            "oxo_NH": "[#6]1~[#7]~[#8]~[#6](~[OX1])~[#7;H1]~1",
            "hydroxy_tautomer": "[#6]1~[#7]~[#8]~[#6](~[OX2H1])~[#7]~1",
            "anion": "[#6]1~[#7]~[#8]~[#6](~[OX1])~[#7;H0;-1]~1",
        },
        extra_forms={
            "skeleton": "[#6]1~[#7]~[#8]~[#6](~[#8])~[#7]~1",
        },
        notes="The azilsartan headgroup. RDKit perceives this ring as aromatic "
              "even from aliphatic input, which is why the pattern uses ~ bonds.",
        aromaticity_sensitive=True,
    ),

    "thiadiazol_5_one": Headgroup(
        name="1,2,4-thiadiazol-5(4H)-one",
        pka="~6-7",
        forms={
            "oxo_NH": "[#6]1~[#7]~[#16]~[#6](~[OX1])~[#7;H1]~1",
            "hydroxy_tautomer": "[#6]1~[#7]~[#16]~[#6](~[OX2H1])~[#7]~1",
            "anion": "[#6]1~[#7]~[#16]~[#6](~[OX1])~[#7;H0;-1]~1",
        },
        extra_forms={
            "skeleton": "[#6]1~[#7]~[#16]~[#6](~[#8])~[#7]~1",
        },
        notes="Sulfur analogue of the oxadiazolone above.",
        aromaticity_sensitive=True,
    ),

    # --------------------------------------------------- acyclic N-H acids
    "acylsulfonamide": Headgroup(
        name="acylsulfonamide (N-acyl sulfonamide)",
        pka="~3.5-4.5",
        forms={
            "neutral": "[CX3](=[OX1])[NX3H1][SX4](=[OX1])(=[OX1])",
            "anion": "[CX3](=[OX1])[#7;H0;-1][SX4](=[OX1])(=[OX1])",
        },
        extra_forms={
            "skeleton": "[CX3](~[#8])~[#7]~[SX4](~[#8])~[#8]",
        },
        notes="The N-H between a carbonyl and a sulfonyl is the acid. A plain "
              "amide has no sulfonyl and does not match; a plain sulfonamide has "
              "no acyl group and does not match; an N-alkylated acylsulfonamide "
              "has no acidic hydrogen and does not match.",
    ),

    "hydroxamic_acid": Headgroup(
        name="hydroxamic acid",
        pka="~8.5-9.4",
        forms={
            "neutral": "[CX3](=[OX1])[NX3H1][OX2H1]",
            "O_anion": "[CX3](=[OX1])[NX3H1][OX1H0-]",
            "N_anion": "[CX3](=[OX1])[#7;H0;-1][OX2H1]",
        },
        extra_forms={
            "skeleton": "[CX3](~[#8])~[#7]~[#8]",
        },
        notes="Weaker than the acid and a strong metal chelator (the reason "
              "vorinostat works on zinc HDACs). An N-methoxy amide is NOT a "
              "hydroxamic acid and is excluded: its oxygen has no hydrogen.",
    ),

    "acyl_cyanamide": Headgroup(
        name="acyl cyanamide (N-cyano amide)",
        pka="~3-4",
        forms={
            "neutral": "[NX1]#[CX2][NX3H1][CX3]=[OX1]",
            "anion": "[NX1]#[CX2][#7;H0;-1][CX3]=[OX1]",
        },
        extra_forms={
            # [#7;X1] is the nitrile's terminal nitrogen. Without it this
            # skeleton matched any N-C-N-C=O, i.e. every urea in ChEMBL (79,677).
            "skeleton": "[#7;X1]~[#6]~[#7]~[CX3]~[#8]",
        },
        notes="The nitrile pulls electrons off the N-H, making it about as acidic "
              "as a carboxylic acid.",
    ),

    "primary_sulfonamide": Headgroup(
        name="primary sulfonamide",
        pka="~10 (benzenesulfonamide)",
        forms={
            "neutral": "[SX4](=[OX1])(=[OX1])[NX3H2]",
            "anion": "[SX4](=[OX1])(=[OX1])[#7;H1;-1]",
        },
        extra_forms={
            # Also matches secondary and tertiary sulfonamides.
            "skeleton": "[SX4](~[#8])(~[#8])~[#7]",
        },
        notes="Only the PRIMARY form (-SO2NH2) is acidic enough to matter, so a "
              "secondary or tertiary sulfonamide (probenecid) does not match. "
              "Also the carbonic anhydrase zinc binder. Note this does not require "
              "a carbon on the sulfur, so a sulfamide (N-SO2-NH2) also matches.",
    ),

    # --------------------------------------------------- strong acyclic acids
    "sulfonic_acid": Headgroup(
        name="sulfonic acid",
        pka="~ -1 to 1 (fully ionized at pH 7.4)",
        forms={
            "neutral": "[#6][SX4](=[OX1])(=[OX1])[OX2H1]",
            "anion": "[#6][SX4](=[OX1])(=[OX1])[OX1H0-]",
        },
        extra_forms={
            "skeleton": "[#6][SX4](~[#8])(~[#8])~[#8]",
        },
        notes="Far stronger than a carboxylic acid, so it is permanently charged "
              "and rarely cell-permeable. The carbon on sulfur distinguishes it "
              "from sulfuric acid and from a sulfate ester; a sulfonate ester is "
              "excluded because its oxygen has no hydrogen.",
    ),

    "phosphonic_acid": Headgroup(
        name="phosphonic acid",
        pka="~1.5 and ~6.5 (two ionizations)",
        forms={
            # Carbon on phosphorus is what makes it a phosphonic acid rather
            # than a phosphate (which has oxygen there).
            "neutral_or_anion": f"[#6][PX4](=[OX1])({ACID_O}){ACID_O}",
        },
        extra_forms={
            "skeleton": "[#6][PX4](~[#8])(~[#8])~[#8]",
        },
        notes="Diprotic, so it carries up to two negative charges. The carbon-P "
              "bond distinguishes it from a phosphate ester.",
    ),

    "phosphinic_acid": Headgroup(
        name="phosphinic acid",
        pka="~2-3",
        forms={
            "neutral_or_anion": f"[#6][PX4](=[OX1])([#6]){ACID_O}",
        },
        extra_forms={
            "skeleton": "[#6][PX4](~[#8])(~[#6])~[#8]",
        },
        notes="Two carbons on phosphorus and one acidic OH, so monoprotic, "
              "unlike the phosphonic acid above.",
    ),

    # ------------------------------------------------- carbon (CH) acids
    "thiazolidine_2_4_dione": Headgroup(
        name="thiazolidine-2,4-dione",
        pka="~6.8",
        forms={
            # Ring S1-C2(=O)-N3-C4(=O)-C5. Bond orders are explicit here because
            # RDKit perceives this ring as aliphatic. The acidic hydrogen is the
            # C5-H, so a fully substituted C5 is not matched.
            "diketo": "[#16]1~[#6](=[OX1])~[#7]~[#6](=[OX1])~[#6;H1,H2]~1",
            "enol": "[#16]1~[#6](=[OX1])~[#7]~[#6](-[OX2H1])=[#6]~1",
            "enolate": "[#16]1~[#6](=[OX1])~[#7]~[#6](~[OX1;H0;-1])~[#6]~1",
            # 5-arylidene TZD (the benzylidene screening scaffold): C5 carries an
            # exocyclic double bond, so the acidic C5-H is gone, but the N3-H
            # remains and the ring is unchanged - still the same headgroup
            #. 659 parents, none of which the forms above reach.
            # Requiring [#7;H1] keeps out the N-substituted arylidenes (680
            # parents), which have no acidic hydrogen at all.
            "sub_arylidene": "[#16]1~[#6](=[OX1])~[#7;H1]~[#6](=[OX1])~[#6]~1=[#6]",
            # 2-hydroxy (iminol) tautomer: the N3 proton has moved onto the C2
            # oxygen, giving S-C(OH)=N- instead of S-C(=O)-N(H)-. Same compound,
            # drawn differently in ChEMBL, and the rule above says it counts.
            # Found by sampling the skeleton gap after the arylidene fix.
            "iminol_tautomer": "[#16]1~[#6](~[OX2H1])~[#7]~[#6](=[OX1])~[#6]~1",
        },
        extra_forms={
            "skeleton": "[#16]1~[#6](~[#8])~[#7]~[#6](~[#8])~[#6]~1",
        },
        notes="The glitazone headgroup. The acidic hydrogen is the ring CH "
              "between the sulfur and the C4 carbonyl.",
    ),

    "cyclopentane_1_3_dione": Headgroup(
        name="cyclopentane-1,3-dione",
        pka="~4.3-5.3",
        forms={
            # Ring C1(=O)-C2-C3(=O)-C4-C5. Bond orders are explicit: with "any
            # bond" this matched 39 prostaglandins, whose ring carries a ketone
            # and a SEPARATE saturated hydroxyl. The enol form requires the
            # C2=C3 double bond that a real enol has and PGE2 does not.
            "diketo": "[#6]1(=[OX1])~[#6;H1,H2]~[#6](=[OX1])~[#6]~[#6]~1",
            "enol": "[#6]1(=[OX1])~[#6]=[#6](-[OX2H1])~[#6]~[#6]~1",
            "enolate": "[#6]1(=[OX1])~[#6]~[#6](~[OX1;H0;-1])~[#6]~[#6]~1",
            # 2-acyl cyclopentane-1,3-dione: the all-carbon analogue of the
            # tetramic acid sub_acyl above. No parent in ChEMBL 37 matches it,
            # but it is here so the two groups stay symmetrical.
            "sub_acyl": f"[#6]1(=[OX1])~[#6](=[#6]~{ACID_O})~[#6](=[OX1])~[#6]~[#6]~1",
        },
        extra_forms={
            # This skeleton is the buggy first draft: its gap is the prostaglandins.
            "skeleton": "[#6]1(~[#8])~[#6]~[#6](~[#8])~[#6]~[#6]~1",
            # Sub-forms: benzo-fused, i.e. indane-1,3-diones (the
            # phenindione scaffold). Still counted as the headgroup - the acidic
            # CH between two carbonyls is the same functional unit - but the
            # fused ring shifts sterics and pKa, so it is named separately.
            "sub_indane_diketo":
                "[#6]1(=[OX1])~[#6;H1,H2]~[#6](=[OX1])~c~c~1",
            "sub_indane_enol": "[#6]1(=[OX1])~[#6]=[#6](-[OX2H1])~c~c~1",
        },
        notes="A carbon acid: the CH2 between two carbonyls. All-carbon, so no "
              "heteroatom for a metabolic handle.",
    ),

    "tetramic_acid": Headgroup(
        name="tetramic acid (pyrrolidine-2,4-dione)",
        pka="~6.4",
        forms={
            # Ring N1-C2(=O)-C3-C4(=O)-C5. The carbonyls are at 2 and 4, which
            # is what separates this from succinimide (carbonyls at 2 and 5).
            # Bond orders are explicit, for the same reason as the dione above.
            "diketo": "[#7]1~[#6](=[OX1])~[#6;H1,H2]~[#6](=[OX1])~[#6]~1",
            "enol": "[#7]1~[#6](=[OX1])~[#6]=[#6](-[OX2H1])~[#6]~1",
            "enolate": "[#7]1~[#6](=[OX1])~[#6]~[#6](~[OX1;H0;-1])~[#6]~1",
            # 3-acyl tetramic acid (the tenuazonic acid motif): C3 carries an
            # exocyclic acyl group and the enol sits OUTSIDE the ring, where the
            # enol form above expects it inside. Same 1,3-dicarbonyl acid, and
            # more acidic than the plain group, not less.
            "sub_acyl": f"[#7]1~[#6](=[OX1])~[#6](=[#6]~{ACID_O})~[#6](=[OX1])~[#6]~1",
        },
        extra_forms={
            "skeleton": "[#7]1~[#6](~[#8])~[#6]~[#6](~[#8])~[#6]~1",
        },
        notes="Nitrogen version of the cyclopentane-1,3-dione. Succinimide and "
              "glutarimide do not match: both their carbonyls touch the nitrogen.",
    ),

    "squaramide": Headgroup(
        name="squaramide (3,4-diamino-cyclobut-3-ene-1,2-dione)",
        pka="~7-9 for the N-H; parent squaric acid is ~0.5 and ~3.4 - uncertain",
        forms={
            # Four-membered ring: two carbons bearing N, two bearing acidic O.
            # One N-H is the acid. The pattern can map round the ring either way,
            # so requiring the hydrogen on the first nitrogen finds it on either.
            "NH": f"[#6]1(~[#7;H1,H2])~[#6](~[#7])~[#6](~{ACID_O})~[#6]~1~{ACID_O}",
            "anion": f"[#6]1(~[#7;H0;-1])~[#6](~[#7])~[#6](~{ACID_O})~[#6]~1~{ACID_O}",
        },
        extra_forms={
            "skeleton": "[#6]1(~[#7])~[#6](~[#7])~[#6](~[#8])~[#6]~1~[#8]",
        },
        notes="RDKit perceives part of this ring as aromatic, so the pattern uses "
              "~ bonds. The pKa is the least certain in this file: it moves a long "
              "way with substitution and the literature is thin. Verify before 3.1.",
        aromaticity_sensitive=True,
    ),
}


def compile_patterns(headgroups=None):
    """name -> {form name -> compiled SMARTS}. Raises if any SMARTS is invalid."""
    headgroups = HEADGROUPS if headgroups is None else headgroups
    compiled = {}
    for key, headgroup in headgroups.items():
        compiled[key] = {}
        for form, smarts in headgroup.patterns().items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise ValueError(f"{key}/{form}: invalid SMARTS {smarts!r}")
            compiled[key][form] = pattern
    return compiled


PATTERNS = compile_patterns()


SKELETON = "skeleton"


def is_reporting_only(key, form):
    """True if this form never decides membership for this headgroup.

    That is decided by which dict the form lives in, not by its name: a
    `sub_` form in `forms` adds population (sub_arylidene), while one in
    `extra_forms` only labels molecules already counted (sub_indane_enol).
    """
    return form in HEADGROUPS[key].extra_forms


def matching_forms(mol, key, patterns=None):
    """Which forms of headgroup `key` molecule `mol` contains (list of form names)."""
    patterns = PATTERNS if patterns is None else patterns
    return [form for form, pattern in patterns[key].items() if mol.HasSubstructMatch(pattern)]


def has_headgroup(mol, key, patterns=None):
    """True if `mol` contains headgroup `key` in any of its membership forms.

    extra_forms are reporting only, so they never make this True on their own.
    """
    patterns = PATTERNS if patterns is None else patterns
    return any(mol.HasSubstructMatch(patterns[key][form]) for form in HEADGROUPS[key].forms)


def find_headgroups(mol, patterns=None):
    """Every headgroup key present in `mol`, in HEADGROUPS order."""
    return [key for key in HEADGROUPS if has_headgroup(mol, key, patterns)]


def count_matches(mol, key, form, patterns=None):
    """How many times one form matches (number of distinct substructure matches)."""
    patterns = PATTERNS if patterns is None else patterns
    return len(mol.GetSubstructMatches(patterns[key][form], uniquify=True))


def count_headgroup(mol, key, patterns=None):
    """How many separate copies of headgroup `key` the molecule carries.

    Counts distinct ATOM SETS across the membership forms, so one tetrazole
    ring that matches both the 1H and the 2H form still counts once, while a
    diacid counts twice. extra_forms are reporting only and never counted.

    This is what tells a real swap from a mislabel: in a genuine acid ->
    tetrazole swap A has one more acid than B and B one more tetrazole than
    A. If both molecules carry the same number of tetrazoles, the tetrazole
    was not swapped in - it was there all along.
    """
    patterns = PATTERNS if patterns is None else patterns
    atom_sets = set()
    for form in HEADGROUPS[key].forms:
        for match in mol.GetSubstructMatches(patterns[key][form], uniquify=True):
            atom_sets.add(frozenset(match))
    return len(atom_sets)
