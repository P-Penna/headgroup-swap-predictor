"""Helpers for step 1.5: joining matched pairs to the measured activities.

WHAT STEP 1.5 IS FOR
--------------------
Up to now a pair is two molecules and a chemistry label. It has no number
attached. This step gives it one: for every protein target both molecules were
measured against, Delta p = pchembl(B) - pchembl(A). That is the label the
whole project predicts.

THE ONE RULE THAT MATTERS MOST HERE
-------------------------------------------------
Aggregate to the MEDIAN per (parent, target, standard_type) BEFORE forming
pairs, never after. If molecule A has 3 IC50 records against a target and
molecule B has 2, the cross product is 6 rows - six "training examples" for
what is really one comparison. Those six rows are near-identical, so a random
split scatters them across train and test and the model scores well by
remembering the duplicate. Collapsing each molecule to one value first makes it
one row, which is what it is.

WHY STANDARD TYPES MUST NEVER MIX INSIDE A PAIR
-----------------------------------------------
IC50, Ki and EC50 all land on the same pchembl scale, but they do not mean the
same thing. Measured on this dataset (session of 2026-10-03): for the 27,427
(parent, target) combinations carrying both an exact IC50 and an exact Ki, Ki
is 0.40 log units more potent on the median, IQR +0.02 to +0.90. Subtracting a
Ki from an IC50 would therefore inject a systematic ~0.4 log shift into the
label - comparable to the 0.57 noise floor - that has nothing to do with the
headgroup swap. So the join matches on (target, standard_type) together, and
`standard_type` is carried on every row.

MATCH LEVEL: HOW COMPARABLE THE TWO MEASUREMENTS ARE
----------------------------------------------------
Not all deltas are equally trustworthy, and the difference is large enough to
matter against a 0.57 noise floor:

  assay   both molecules appear in the same assay (same assay_chembl_id).
          The best case: one experiment, one protocol, one day. Systematic
          error largely cancels in the difference, which is why rule 3 says a
          legitimate model can score below 0.57 on same-assay pairs without
          leaking.
  doc     different assays, same paper. Usually one lab and one protocol.
  target  different papers entirely. This is the case the 0.57 cross-document
          noise floor was measured on, and the worst case for the label.

Kept as a column rather than used to filter, so a later step can weight or
restrict by it and report what that costs.
"""

import statistics

MEDIAN = "median"          # the name the SQLite aggregate is registered under
DISAGREEMENT_LOGS = 2.0    # per decision: flag, never drop, above this spread

# How a measurement's censoring is labelled. A pchembl UPPER bound is a record
# like "IC50 > 1000 nM": the compound is at best this potent, so the pchembl is
# capped above. These are named in words rather than carried as a flag, and the
# same vocabulary is used throughout.
CENSOR_LABELS = ("exact", "upper", "lower", "mixed")


class Median:
    """SQLite aggregate: the median of a group's values.

    SQLite has no median of its own, so it is registered per connection with
    `register_median`. Values accumulate per group and the group is small (the
    records for one parent, target and standard type), so this holds a handful
    of floats at a time, not the table.
    """

    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(float(value))

    def finalize(self):
        if not self.values:
            return None
        return float(statistics.median(self.values))


def register_median(connection):
    """Make median() available to SQL on this connection."""
    connection.create_aggregate(MEDIAN, 1, Median)
    return connection


def split_ids(text):
    """SQLite group_concat output back into a set of id strings.

    group_concat joins with commas and gives NULL for an empty group, so both
    the None and the empty-string cases have to come back as an empty set
    rather than {""} - otherwise two measurements with no assay recorded would
    "share" the empty id and be called a same-assay pair.
    """
    if not text:
        return set()
    return {piece for piece in str(text).split(",") if piece and piece != "None"}


def match_level(assays_a, assays_b, docs_a, docs_b):
    """How comparable two measurements are: 'assay', 'doc' or 'target'.

    Each argument is group_concat text or a set. Checked in order of quality:
    sharing an assay is better evidence than sharing a paper, which is better
    than sharing only the protein.
    """
    assays_a = assays_a if isinstance(assays_a, set) else split_ids(assays_a)
    assays_b = assays_b if isinstance(assays_b, set) else split_ids(assays_b)
    if assays_a & assays_b:
        return "assay"
    docs_a = docs_a if isinstance(docs_a, set) else split_ids(docs_a)
    docs_b = docs_b if isinstance(docs_b, set) else split_ids(docs_b)
    if docs_a & docs_b:
        return "doc"
    return "target"


def censor_label(n_exact, n_upper, n_lower):
    """One word for how a molecule's median value is censored.

    'mixed' is its own label rather than being folded into one of the others
    because the median of some exact and some bounded records is neither an
    exact value nor a clean bound, and a later step that wants to drop or
    down-weight censored labels needs to be able to see that case.
    """
    n_exact, n_upper, n_lower = int(n_exact or 0), int(n_upper or 0), int(n_lower or 0)
    kinds = [n_exact > 0, n_upper > 0, n_lower > 0]
    if sum(kinds) > 1:
        return "mixed"
    if n_upper:
        return "upper"
    if n_lower:
        return "lower"
    return "exact"


def documents_disagree(spread_a, spread_b, limit=DISAGREEMENT_LOGS):
    """True if either molecule's own documents disagree by more than `limit`.

    The spread is max - min over that molecule's per-document median pchembl
    for this target and type. A unit error - nM reported as uM, or a value
    copied from the wrong column - shows up as a 3-log gap between two papers
    on the same compound, and the median then lands between two incompatible
    numbers. That manufactures an activity cliff out of a typo, which hurts
    more than ordinary scatter because the model is being asked to explain a
    difference that is not real. Flagged, not dropped: the pair may still be
    fine, and the count is reported so the decision can be made on numbers.
    """
    for spread in (spread_a, spread_b):
        if spread is not None and float(spread) > limit:
            return True
    return False


def pair_year(year_a, year_b):
    """The year the pair became knowable: the later of the two first-reports.

    Each molecule's year is the earliest year it was measured on this target,
    so the pair itself cannot predate the later of the two. The time split
    trains on <= a cutoff and tests after it, and using the earlier year would
    put a pair in the training era before one of its two molecules existed.
    """
    years = [year for year in (year_a, year_b) if year is not None]
    if not years:
        return None
    return int(max(years))
