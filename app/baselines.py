"""The three baselines every model must beat.

  zero                              predict delta = 0 always.
  transformation_median             the training median delta for that
                                    directed headgroup swap, global median
                                    otherwise.
  transformation_environment_median keyed on (directed headgroup swap,
                                    radius-2 environment), falling back to the
                                    swap median, then global.

"Transformation" here means the HEADGROUP PAIR, e.g. ('carboxylic_acid',
'tetrazole'), not mmpdb's transformation string. The string carries whatever
scaffold the cut left in the variable fragment, so on the real data the 66,900
bioisostere pairs spread over 46,622 strings - a median over 1.43 pairs, which
is a noisy lookup, not a baseline. Over headgroup pairs it is 184 directed
classes. The names keep "transformation" because they are the rule-2 names.

These are not strawmen. Most headgroup swaps do little, so `zero` is hard to
beat on RMSE, and a model that cannot beat it has learned nothing. The other
two encode what a chemist would guess from a lookup table: "tetrazoles usually
cost you about this much", then "...and about this much on an aryl".

DIRECTION MATTERS, SO SWAPS ARE NOT MERGED HERE
-----------------------------------------------
delta is antisymmetric: if A>>B is +0.8 then B>>A is -0.8. So the median must
be keyed on the DIRECTED headgroup pair (`splits.directed_swap`). Merging the
two directions (which `splits.swap_class` deliberately does) would average
+0.8 with -0.8 and predict roughly zero for both.

This is the one place where splitting and predicting must use different keys,
and getting it backwards would quietly destroy both baselines while still
producing plausible-looking numbers:
  splitting    merges directions, so a transformation cannot be in train and
               test at once.
  predicting   keeps them separate, so the sign survives.

BASELINE 2 IS UNDEFINED ON A TRANSFORMATION SPLIT
-------------------------------------------------
A transformation split holds out whole transformations, so by construction NO
test transformation was seen in training and every row falls back to the
global median. The baseline does not fail - it silently becomes "predict the
global median", which is a different and much weaker model. `coverage()`
reports the fallback rate for exactly this reason, and
`describe_collapse()` turns a 100% fallback rate into a sentence, so the
number is never read as a transformation-median score. The same applies to
baseline 3, which collapses two levels instead of one.
"""

from collections import Counter

import numpy as np
import pandas as pd

from app.splits import directed_swap

HEADGROUP_A = "headgroup_a"
HEADGROUP_B = "headgroup_b"
ENVIRONMENT = "env_radius2_smarts"
DELTA = "delta_p"

# Names for the lookup level a prediction came from, most specific first.
LEVELS = ["transformation_environment", "transformation", "global"]


# Columns that identify one COMPARISON - two molecules and the series they sit
# in - as opposed to one row, which is a comparison on one target with one
# measurement type. Used for training weights.
COMPARISON_COLUMNS = ("parent_a_id", "parent_b_id", "constant_smiles")


def comparison_ids(pairs):
    """An integer id per row naming the COMPARISON it belongs to.

    Rows sharing an id are the same two molecules in the same series, measured
    on different targets or by different assay types. They are not independent
    observations, which matters twice: for training weights and
    for the standard error of any comparison between models.

    Falls back to one id per row when the identifying columns are absent, which
    makes every row its own cluster - the behaviour of code that never knew
    about comparisons.
    """
    missing = [name for name in COMPARISON_COLUMNS if name not in pairs.columns]
    if missing:
        return np.arange(len(pairs))
    identity = [(tuple(sorted((str(a), str(b)))), str(constant))
                for a, b, constant in zip(pairs[COMPARISON_COLUMNS[0]],
                                          pairs[COMPARISON_COLUMNS[1]],
                                          pairs[COMPARISON_COLUMNS[2]])]
    seen = {}
    ids = np.empty(len(identity), dtype=np.int64)
    for index, key in enumerate(identity):
        ids[index] = seen.setdefault(key, len(seen))
    return ids


def comparison_weights(pairs):
    """Weight each training row by 1 / rows describing the same comparison.

    THE PROBLEM THIS FIXES: one compound pair measured against
    five targets contributes five rows whose features are IDENTICAL - the
    headgroups, the environment and the swap class do not depend on the target
    - so it pulls every median and every regression five times as hard as a
    pair measured once. Those five rows are not five independent observations
    of the same quantity either: they are the same two molecules on five
    different proteins.

    So each comparison gets total weight 1, shared between its rows. The key is
    the UNORDERED molecule pair plus the constant, which makes a pair and its
    mirror one comparison - they carry the same information with opposite sign,
    and an antisymmetric model needs both, so both are kept at half weight
    each. Relative weighting between compound pairs is what matters and this
    equalises it.

    Returns all-ones when the identifying columns are absent, so synthetic
    tables and unit tests behave exactly as before.
    """
    missing = [name for name in COMPARISON_COLUMNS if name not in pairs.columns]
    if missing:
        return np.ones(len(pairs), dtype=float)
    ids = comparison_ids(pairs)
    counts = Counter(ids.tolist())
    return np.asarray([1.0 / counts[key] for key in ids], dtype=float)


def weighted_median(values, weights):
    """The weighted median, agreeing with np.median when the weights are equal.

    Written here rather than reusing `src.noise_floor.weighted_quantile`
    because that one picks a data point at the crossing, so on an even number
    of equally weighted values it returns the lower of the two middle ones
    where np.median averages them. That convention is correct for the
    published noise-floor figures and must not change, but adopting it here
    would silently move EVERY median baseline the moment weighting was turned
    on - a change of model, disguised as a change of weighting. So this
    averages at an exact crossing, which reproduces np.median for equal
    weights and is pinned by a test against it.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(values):
        return 0.0
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    total = float(weights.sum())
    if total <= 0:
        return float(np.median(values))
    cumulative = np.cumsum(weights)
    half = total / 2.0
    index = int(np.searchsorted(cumulative, half))
    index = min(index, len(values) - 1)
    # An exact crossing means the weight splits evenly between this point and
    # the next, so neither is "the" median and the midpoint is taken.
    if index + 1 < len(values) and np.isclose(cumulative[index], half):
        return float((values[index] + values[index + 1]) / 2.0)
    return float(values[index])


def swap_keys(pairs):
    """The directed headgroup pair for every row, as a list of tuples."""
    return [directed_swap(a, b) for a, b in zip(pairs[HEADGROUP_A], pairs[HEADGROUP_B])]


class Baseline:
    """Shared shape: fit on a training frame, predict on a test frame."""

    name = "baseline"
    required_columns = []

    def require(self, pairs, extra=()):
        needed = list(self.required_columns) + list(extra)
        missing = [column for column in needed if column not in pairs.columns]
        if missing:
            raise ValueError(f"{self.name} needs columns {missing}; the table has "
                             f"{sorted(pairs.columns)}")

    def fit(self, train):
        raise NotImplementedError

    def predict(self, test):
        raise NotImplementedError

    def coverage(self, test):
        """How many test rows were answered at each lookup level."""
        return {"global": len(test)}

    def describe_collapse(self, test):
        """A sentence when this baseline has degenerated into a weaker one.

        Returns None when it has not, so callers can print it only if present.
        """
        counts = self.coverage(test)
        total = sum(counts.values())
        if not total:
            return None
        specific = total - counts.get("global", 0)
        if specific == 0:
            return (f"{self.name}: every one of {total:,} test pairs fell back to the "
                    f"GLOBAL median, so this is not a {self.name} score - it is the "
                    f"global median. Expected on a transformation split, where no "
                    f"test transformation appears in training.")
        fraction = 1 - specific / total
        if fraction > 0.5:
            return (f"{self.name}: {fraction:.0%} of {total:,} test pairs fell back to "
                    f"the global median, so most of this score is not coming from the "
                    f"lookup it is named after.")
        return None


class ZeroBaseline(Baseline):
    """Predict no change at all.

    The one baseline that cannot overfit, and the one that exposes a dataset
    where nothing much happens: if most swaps are small, this scores well on
    RMSE while being useless, which is why sign accuracy and MCC are reported
    beside it. It has no direction and no ordering, so sign_accuracy is 0 and
    spearman_rho is NaN by construction, not by accident.
    """

    name = "zero"

    def fit(self, train):
        return self

    def predict(self, test):
        return np.zeros(len(test), dtype=float)

    def coverage(self, test):
        """No lookup levels: this baseline does not consult the training data."""
        return {}

    def describe_collapse(self, test):
        """Never collapses, because there is nothing to fall back FROM.

        The base class treats "everything answered at the global level" as a
        collapse, which is right for the median baselines and wrong here - the
        first version of this class inherited that and reported the zero
        baseline as having degenerated into the global median on every run.
        """
        return None


class TrainingMeanBaseline(Baseline):
    """Predict one constant: the mean Delta p of the training rows.

    The point of it is to answer "are the clever baselines even beating a
    single number?". A model that cannot beat one constant has learned
    nothing, and RMSE alone will not tell you, because RMSE against a constant
    IS the spread of the labels.

    EXPECT THIS TO EQUAL THE ZERO BASELINE ON THIS DATASET, and that is a
    result rather than a bug. `mmpdb index --symmetric` stores both directions
    of every pair, every splitter keeps the two directions in the same fold, so a training fold contains +d for every -d and its mean is
    exactly 0. The zero baseline is therefore not an arbitrary choice of
    constant - it is the best constant, and this baseline is what demonstrates
    that rather than assuming it.

    It consults the training data at the global level only, which for this
    baseline is the whole design, so the collapse note is suppressed like the
    zero baseline's.
    """

    name = "training_mean"
    required_columns = [DELTA]

    def fit(self, train):
        self.require(train)
        values = pd.to_numeric(train[DELTA], errors="coerce")
        weights = comparison_weights(train)
        keep = values.notna().to_numpy()
        if not keep.any():
            self.value = 0.0
            return self
        # Weighted, so a pair measured on five targets does not move the mean
        # five times as far.
        self.value = float(np.average(values.to_numpy()[keep],
                                      weights=weights[keep]))
        return self

    def predict(self, test):
        return np.full(len(test), self.value, dtype=float)

    def coverage(self, test):
        """One constant, so there is no lookup level to report."""
        return {}

    def describe_collapse(self, test):
        """Cannot collapse: a single constant is what it claims to be."""
        return None


class TransformationMedianBaseline(Baseline):
    """Median training delta for that directed headgroup swap.

    Median rather than mean: ChEMBL deltas have a heavy tail (10% of
    cross-document differences exceed 2 log units), and one
    mis-keyed measurement would drag a mean a long way.
    """

    name = "transformation_median"
    required_columns = [HEADGROUP_A, HEADGROUP_B, DELTA]

    def __init__(self):
        self.median_by_swap = {}
        self.global_median = 0.0

    def fit(self, train):
        self.require(train)
        deltas = pd.to_numeric(train[DELTA], errors="coerce")
        weights = pd.Series(comparison_weights(train), index=train.index)
        usable = train.assign(**{DELTA: deltas}).dropna(subset=[DELTA])
        weights = weights.loc[usable.index]
        self.global_median = weighted_median(usable[DELTA], weights) \
            if len(usable) else 0.0
        deltas_by_swap = {}
        for swap, delta, weight in zip(swap_keys(usable), usable[DELTA], weights):
            deltas_by_swap.setdefault(swap, ([], []))
            deltas_by_swap[swap][0].append(delta)
            deltas_by_swap[swap][1].append(weight)
        self.median_by_swap = {swap: weighted_median(values, row_weights)
                               for swap, (values, row_weights)
                               in deltas_by_swap.items()}
        return self

    def predict(self, test):
        self.require(test, extra=())
        predictions = []
        for swap in swap_keys(test):
            predictions.append(self.median_by_swap.get(swap, self.global_median))
        return np.asarray(predictions, dtype=float)

    def coverage(self, test):
        counts = {"transformation": 0, "global": 0}
        for swap in swap_keys(test):
            key = "transformation" if swap in self.median_by_swap else "global"
            counts[key] += 1
        return counts


class TransformationEnvironmentMedianBaseline(Baseline):
    """Median training delta for (directed headgroup swap, radius-2 environment).

    The local environment is the point: an acid to tetrazole swap on an
    electron-poor aryl is not the same as on an alkyl chain. This is the
    strongest thing available without a model, so it is the one to beat.
    Falls back to the transformation median, then the global median.
    """

    name = "transformation_environment_median"
    required_columns = [HEADGROUP_A, HEADGROUP_B, ENVIRONMENT, DELTA]

    def __init__(self, min_count=1):
        self.min_count = min_count
        self.median_by_pair = {}
        self.transformation_baseline = TransformationMedianBaseline()

    def fit(self, train):
        self.require(train)
        self.transformation_baseline.fit(train)
        deltas = pd.to_numeric(train[DELTA], errors="coerce")
        weights = pd.Series(comparison_weights(train), index=train.index)
        usable = train.assign(**{DELTA: deltas}).dropna(subset=[DELTA])
        weights = weights.loc[usable.index]
        deltas_by_pair = {}
        for swap, environment, delta, weight in zip(
                swap_keys(usable), usable[ENVIRONMENT], usable[DELTA], weights):
            deltas_by_pair.setdefault((swap, environment), ([], []))
            deltas_by_pair[(swap, environment)][0].append(delta)
            deltas_by_pair[(swap, environment)][1].append(weight)
        self.median_by_pair = {key: weighted_median(values, row_weights)
                               for key, (values, row_weights)
                               in deltas_by_pair.items()
                               if len(values) >= self.min_count}
        return self

    def predict(self, test):
        self.require(test)
        fallback = self.transformation_baseline.predict(test)
        predictions = []
        for position, (swap, environment) in enumerate(
                zip(swap_keys(test), test[ENVIRONMENT])):
            key = (swap, environment)
            if key in self.median_by_pair:
                predictions.append(self.median_by_pair[key])
            else:
                predictions.append(fallback[position])
        return np.asarray(predictions, dtype=float)

    def coverage(self, test):
        counts = {level: 0 for level in LEVELS}
        known = self.transformation_baseline.median_by_swap
        for swap, environment in zip(swap_keys(test), test[ENVIRONMENT]):
            if (swap, environment) in self.median_by_pair:
                counts["transformation_environment"] += 1
            elif swap in known:
                counts["transformation"] += 1
            else:
                counts["global"] += 1
        return counts


class HeadgroupPropertyBaseline(Baseline):
    """Ridge regression on the DIFFERENCE of the two headgroups' properties.

    The first baseline here that does not look the swap up in the training
    data, so the first that still has a prediction when the swap class is
    unseen. Features are properties(B) - properties(A): pKa, heavy atoms,
    H-bond donors, H-bond acceptors, TPSA (see app/headgroup_properties.py).

    FITTED WITHOUT AN INTERCEPT, deliberately. On difference features that
    makes the model antisymmetric - swap the two headgroups and the prediction
    negates - which is the symmetry Delta p actually has and the symmetry the
    training data carries, since both directions of every pair are present. An intercept would be fitted to about zero anyway; pinning
    it removes the chance of a small spurious offset and makes the antisymmetry
    exact rather than approximate.

    Ridge rather than plain least squares because pKa, heavy atoms and TPSA are
    correlated across these seventeen groups (a sulfonic acid is small, polar
    and very acidic at once), and an unregularised fit on five correlated
    features over a handful of distinct swap classes gives unstable, unreadable
    coefficients.

    WHAT IT MEANS IF THIS WORKS: bioisosterism says a swap is tolerated when
    the replacement matches the original on what the protein sees. A useful
    score here is evidence for that, measured on this dataset, and a reason to
    build a physics-feature channel. A useless score is a reason not to.
    """

    name = "headgroup_property_ridge"
    required_columns = [HEADGROUP_A, HEADGROUP_B, DELTA]
    alpha = 1.0

    def _features(self, pairs):
        """Feature rows and a mask of which pairs could be described.

        swap_keys returns the directed (headgroup_a, headgroup_b) tuple per
        row, or None when neither side has a headgroup, so both cases are
        handled rather than unpacked blindly.
        """
        from app.headgroup_properties import difference_vector
        rows, usable = [], []
        for key in swap_keys(pairs):
            vector = difference_vector(key[0], key[1]) if key else None
            usable.append(vector is not None)
            rows.append(vector)
        return rows, np.asarray(usable, dtype=bool)

    def fit(self, train):
        from sklearn.linear_model import Ridge
        self.require(train)
        rows, usable = self._features(train)
        labels = pd.to_numeric(train[DELTA], errors="coerce").to_numpy()
        weights = comparison_weights(train)
        finite = usable & np.isfinite(labels)
        self.model = None
        self.n_fitted = int(finite.sum())
        if self.n_fitted >= 2:
            matrix = np.asarray([row for row, keep in zip(rows, finite) if keep],
                                dtype=float)
            self.model = Ridge(alpha=self.alpha, fit_intercept=False)
            self.model.fit(matrix, labels[finite], sample_weight=weights[finite])
        return self

    def predict(self, test):
        rows, usable = self._features(test)
        predictions = np.zeros(len(test), dtype=float)
        if self.model is None or not usable.any():
            return predictions
        matrix = np.asarray([row for row, keep in zip(rows, usable) if keep],
                            dtype=float)
        predictions[usable] = self.model.predict(matrix)
        return predictions

    def coverage(self, test):
        """Rows the properties could describe, against rows that fell back.

        A fallback here means a headgroup with no properties at all - the "none"
        side of a deletion pair, or a group added to app/headgroups.py without
        an entry in app/headgroup_properties.py. It does NOT mean an unseen
        swap class, which this baseline handles by construction.
        """
        _, usable = self._features(test)
        return {"headgroup_properties": int(usable.sum()),
                "global": int((~usable).sum())}

    def coefficients(self):
        """{property: weight}, for reading what the model thinks matters."""
        from app.headgroup_properties import PROPERTY_NAMES
        if self.model is None:
            return {}
        return {name: float(weight)
                for name, weight in zip(PROPERTY_NAMES, self.model.coef_)}


class EnvironmentPropertyBaseline(Baseline):
    """Headgroup-property ridge fitted SEPARATELY PER LOCAL ENVIRONMENT.

    The baseline the variance decomposition pointed at. Keyed on
    the headgroup pair alone, only 17.0% of Delta p variance is reachable,
    because the property features are identical for every pair of a given swap
    class. Crossing them with `env_radius2_smarts` - the atoms next to the cut,
    which baseline 3 already answers 85-94% of test pairs with - raises the
    reachable share to 39.0% (measured on cells of at least 5 comparisons), so
    the environment is the cheapest available source of WITHIN-class variation.

    One ridge per environment with at least `min_count` training rows, falling
    back to a single global ridge over all rows, then to zero. The fallback
    chain is the same shape as baseline 3's and is reported the same way, so a
    run can see how much of the test set the specific level actually answered.

    Still antisymmetric: every ridge is fitted without an intercept on
    difference features, and a pair's mirror carries the SAME environment -
    checked on the real data, 57,184 of 57,184 bioisostere rows (100.00%) - so
    the two directions land in the same model and negate exactly.

    Per-environment fits rather than one model with interaction terms because
    the environment is a high-cardinality string (1,412 swap x environment
    cells on the real data); one-hot crossing it with five properties would be
    thousands of mostly empty columns, while a per-environment fit degrades
    gracefully and stays readable.
    """

    name = "environment_property_ridge"
    required_columns = [HEADGROUP_A, HEADGROUP_B, ENVIRONMENT, DELTA]
    alpha = 1.0

    def __init__(self, min_count=20):
        # 20 rows for 5 free parameters. Below that the fit is noise, and the
        # global ridge is the better answer.
        self.min_count = min_count
        self.models = {}
        self.fallback = HeadgroupPropertyBaseline()

    def fit(self, train):
        from sklearn.linear_model import Ridge
        self.require(train)
        self.fallback.fit(train)
        rows, usable = self.fallback._features(train)
        labels = pd.to_numeric(train[DELTA], errors="coerce").to_numpy()
        weights = comparison_weights(train)
        finite = usable & np.isfinite(labels)

        grouped = {}
        for index, environment in enumerate(train[ENVIRONMENT]):
            if finite[index]:
                grouped.setdefault(environment, []).append(index)

        self.models = {}
        for environment, indices in grouped.items():
            if len(indices) < self.min_count:
                continue
            matrix = np.asarray([rows[index] for index in indices], dtype=float)
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(matrix, labels[indices], sample_weight=weights[indices])
            self.models[environment] = model
        return self

    def _level(self, environment, vector):
        if vector is None:
            return "global", None
        if environment in self.models:
            return "transformation_environment", self.models[environment]
        if self.fallback.model is not None:
            return "transformation", self.fallback.model
        return "global", None

    def predict(self, test):
        self.require(test)
        rows, _usable = self.fallback._features(test)
        predictions = np.zeros(len(test), dtype=float)
        for index, environment in enumerate(test[ENVIRONMENT]):
            _level, model = self._level(environment, rows[index])
            if model is not None:
                predictions[index] = float(
                    model.predict(np.asarray([rows[index]], dtype=float))[0])
        return predictions

    def coverage(self, test):
        rows, _usable = self.fallback._features(test)
        counts = {level: 0 for level in LEVELS}
        for index, environment in enumerate(test[ENVIRONMENT]):
            level, _model = self._level(environment, rows[index])
            counts[level] += 1
        return counts

    def environments_fitted(self):
        """How many environments got their own model."""
        return len(self.models)


class ShrunkCellMeanBaseline(Baseline):
    """Swap x <cell> cell means pulled toward their swap-class mean.

    `cell_column` names the second key. Two concrete versions exist:
    `ShrunkEnvironmentMeanBaseline` (the local environment) and
    `ShrunkTargetMeanBaseline` (the protein target). The machinery
    is identical, which is the point - it makes the two conditioning variables
    comparable on exactly the same footing.

    THE BASELINE THE VARIANCE ANALYSIS ARGUED FOR. The environment carries real
    signal -
    the unbiased between-cell variance component rises from 17.7% of Delta p
    variance for the swap class alone to 37.2% once the environment is added,
    an RMSE ceiling of 0.916 against the zero baseline's 1.158. But estimating
    each cell on its own fails, because the median cell holds 3 comparisons and
    the standard error of its mean (0.530) is comparable to the whole
    between-cell spread (0.705): `transformation_environment_median` and
    `environment_property_ridge` are both WORSE than their pooled counterparts.
    Partial pooling is the textbook answer to exactly that shape of problem.

    For each cell c inside swap class g:

        prediction = mu_g + lambda_c * (ybar_c - mu_g)
        lambda_c   = n_c / (n_c + alpha)
        alpha      = strength * sigma_within^2 / tau^2

    lambda_c runs from 0 for a cell with no data of its own to 1 for a cell
    with plenty, so a thinly measured cell is barely trusted and a well
    measured one is trusted almost completely. alpha is the empirical-Bayes
    value - how many observations a cell needs before its own mean outweighs
    the class mean - estimated from the TRAINING data only: sigma_within^2 is
    the pooled within-cell variance and tau^2 the between-cell variance
    component by the ANOVA moment estimator, both computed on training rows.

    `strength` multiplies that estimate so it can be swept, and the two
    endpoints are the baselines it sits between: strength -> 0 gives the raw
    cell means (partitioning, what baseline 3 does) and strength -> infinity
    gives the class mean for every cell (pooling, what baseline 2 does).
    strength = 1 is the empirical-Bayes setting.

    MEANS, not medians, because the shrinkage formula is a normal-model result
    and there is no comparably simple one for medians. That makes this baseline
    more exposed to the heavy tail than baselines 2 and 3 (9.3% of bioisostere
    comparisons move more than 2 log units), which is a real cost and visible
    in the sweep as the strength -> 0 end.

    Antisymmetric: the cell for A -> B and the cell for B -> A hold exactly
    negated values and the same n, and a pair's mirror always carries the same
    environment (checked: 57,184 of 57,184 real rows), so both the class mean
    and the cell mean negate, and lambda is identical.
    """

    name = "shrunk_cell_mean"
    cell_column = ENVIRONMENT

    def __init__(self, strength=1.0, cell_column=None):
        self.strength = strength
        if cell_column is not None:
            self.cell_column = cell_column
        self.required_columns = [HEADGROUP_A, HEADGROUP_B, self.cell_column, DELTA]
        self.cell = {}
        self.group = {}
        self.global_mean = 0.0
        self.alpha = float("nan")

    def fit(self, train):
        self.require(train)
        deltas = pd.to_numeric(train[DELTA], errors="coerce").to_numpy()
        weights = comparison_weights(train)
        finite = np.isfinite(deltas)

        swaps = swap_keys(train)
        environments = list(train[self.cell_column])
        cells, groups = {}, {}
        for index in range(len(train)):
            swap = swaps[index]
            if not finite[index] or swap is None:
                continue
            cell_key = (swap, environments[index])
            for store, key in ((cells, cell_key), (groups, swap)):
                if key not in store:
                    store[key] = ([], [])
                values, row_weights = store[key]
                values.append(deltas[index])
                row_weights.append(weights[index])

        def weighted_mean(values, row_weights):
            values = np.asarray(values, dtype=float)
            row_weights = np.asarray(row_weights, dtype=float)
            if row_weights.sum() <= 0:
                return float(values.mean())
            return float(np.average(values, weights=row_weights))

        self.global_mean = weighted_mean(deltas[finite], weights[finite]) \
            if finite.any() else 0.0
        self.group = {key: weighted_mean(*value) for key, value in groups.items()}
        cell_means = {key: weighted_mean(*value) for key, value in cells.items()}
        cell_counts = {key: len(value[0]) for key, value in cells.items()}

        # Variance components, from training rows only. Plain row counts are
        # used here: the weights decide what the MEANS are, these two numbers
        # are nuisance parameters for how far to trust them.
        within_sq, within_df = 0.0, 0
        for key, (values, _row_weights) in cells.items():
            array = np.asarray(values, dtype=float)
            if len(array) > 1:
                within_sq += float(((array - array.mean()) ** 2).sum())
                within_df += len(array) - 1
        sigma_within = within_sq / within_df if within_df > 0 else 0.0

        total_rows = sum(cell_counts.values())
        n_cells = len(cell_counts)
        between_sq = 0.0
        for key, mean in cell_means.items():
            between_sq += cell_counts[key] * (mean - self.group[key[0]]) ** 2
        n_groups = len(self.group)
        between_df = max(n_cells - n_groups, 1)
        mean_square_between = between_sq / between_df
        sizes = np.asarray(list(cell_counts.values()), dtype=float)
        n_effective = ((total_rows - (sizes ** 2).sum() / total_rows) / between_df
                       if total_rows else 1.0)
        tau = max((mean_square_between - sigma_within) / max(n_effective, 1e-9), 0.0)

        # The sweep's two endpoints are honoured exactly, whatever the data
        # says, because they are what the sweep means by "partition" and
        # "pool". The two DEGENERATE cases below them pull in OPPOSITE
        # directions, and the first version of this ran them together, which
        # made a noiseless training set pool completely instead of trusting
        # its cells.
        if self.strength == 0:
            self.alpha = 0.0                 # asked for no shrinkage
        elif not np.isfinite(self.strength):
            self.alpha = float("inf")        # asked for total pooling
        elif tau <= 0:
            self.alpha = float("inf")        # cells carry no signal: pool
        elif sigma_within <= 0:
            self.alpha = 0.0                 # cells are exact: trust them
        else:
            self.alpha = self.strength * sigma_within / tau

        self.cell = {}
        for key, mean in cell_means.items():
            count = cell_counts[key]
            if not np.isfinite(self.alpha):
                shrunk = self.group[key[0]]
            elif self.alpha <= 0:
                shrunk = mean
            else:
                weight = count / (count + self.alpha)
                shrunk = self.group[key[0]] + weight * (mean - self.group[key[0]])
            self.cell[key] = float(shrunk)
        return self

    def _lookup(self, swap, environment):
        if swap is not None and (swap, environment) in self.cell:
            return self.cell[(swap, environment)], "transformation_environment"
        if swap in self.group:
            return self.group[swap], "transformation"
        return self.global_mean, "global"

    def predict(self, test):
        self.require(test)
        swaps = swap_keys(test)
        return np.asarray([self._lookup(swap, cell)[0]
                           for swap, cell in zip(swaps, test[self.cell_column])],
                          dtype=float)

    def coverage(self, test):
        counts = {level: 0 for level in LEVELS}
        swaps = swap_keys(test)
        for swap, cell in zip(swaps, test[self.cell_column]):
            counts[self._lookup(swap, cell)[1]] += 1
        return counts

    def shrinkage_report(self):
        """alpha, and how hard the typical cell is actually pulled."""
        if not self.cell:
            return {"alpha": self.alpha, "cells": 0}
        return {"alpha": self.alpha, "cells": len(self.cell),
                "groups": len(self.group)}


class ShrunkEnvironmentMeanBaseline(ShrunkCellMeanBaseline):
    """Shrunk cell means keyed on the local environment.

    The result was negative: the optimum is at near-total pooling and the
    environment's whole marginal contribution is 0.4 RMSE points, because a
    series has a median of 1 environment so the cell partly encodes series
    identity and does not transfer across a series split.
    """

    name = "shrunk_environment_mean"
    cell_column = ENVIRONMENT


class ShrunkTargetMeanBaseline(ShrunkCellMeanBaseline):
    """Shrunk cell means keyed on the PROTEIN TARGET.

    The candidate the within-class variance points at. A headgroup swap's
    effect is partly a property of the protein, not of the molecule: acid ->
    hydroxamic acid has a within-class standard deviation of 1.47 log units
    because a hydroxamic acid chelates zinc, so the swap is worth a lot on a
    metalloenzyme and little elsewhere. Unlike the environment, the target is
    not a function of the series, so there is no reason for it to be absorbed
    by a series split.
    """

    name = "shrunk_target_mean"
    cell_column = "target_chembl_id"


class NearestSwapClassMedianBaseline(Baseline):
    """The training median of the most property-similar swap class.

    The cheap companion to the ridge model, and a different way out of the same
    hole: when a swap class is unseen, fall back not to the global median but
    to the median of the nearest swap class in property-difference space.
    "Nearest" is Euclidean distance on the five difference features, each
    scaled by its spread across the training classes so pKa (range ~10) does
    not drown TPSA (range ~100).

    The difference vector is SIGNED, so acid -> tetrazole and tetrazole -> acid
    are far apart and the sign of the borrowed median is right. Exact matches
    reduce to the transformation median, which is the point: it degrades to
    baseline 2 where baseline 2 works, and keeps working where it does not.
    """

    name = "nearest_swap_class_median"
    required_columns = [HEADGROUP_A, HEADGROUP_B, DELTA]

    def fit(self, train):
        from app.headgroup_properties import difference_vector
        self.require(train)
        labels = pd.to_numeric(train[DELTA], errors="coerce")
        weights = comparison_weights(train)
        sums = {}
        for key, label, weight in zip(swap_keys(train), labels, weights):
            if key is None or not np.isfinite(label):
                continue
            sums.setdefault(key, ([], []))
            sums[key][0].append(float(label))
            sums[key][1].append(float(weight))
        self.medians = {key: weighted_median(values, row_weights)
                        for key, (values, row_weights) in sums.items()}
        self.vectors = {}
        for key in self.medians:
            vector = difference_vector(key[0], key[1])
            if vector is not None:
                self.vectors[key] = np.asarray(vector, dtype=float)
        finite = labels.notna().to_numpy()
        self.global_median = weighted_median(labels.to_numpy()[finite],
                                             weights[finite]) \
            if finite.any() else 0.0
        if self.vectors:
            stacked = np.vstack(list(self.vectors.values()))
            spread = stacked.std(axis=0)
            # A feature with no variation carries no information; scaling it by
            # 1 rather than 0 avoids dividing by zero and leaves it harmless.
            self.scale = np.where(spread > 0, spread, 1.0)
        else:
            self.scale = None
        return self

    def _nearest(self, key):
        from app.headgroup_properties import difference_vector
        if key in self.medians:
            return self.medians[key], "transformation"
        if not self.vectors or self.scale is None:
            return self.global_median, "global"
        vector = difference_vector(key[0], key[1]) if key else None
        if vector is None:
            return self.global_median, "global"
        target = np.asarray(vector, dtype=float) / self.scale
        best, best_distance = None, None
        for other, other_vector in self.vectors.items():
            distance = float(np.linalg.norm(other_vector / self.scale - target))
            if best_distance is None or distance < best_distance:
                best, best_distance = other, distance
        return self.medians[best], "nearest_swap_class"

    def predict(self, test):
        return np.asarray([self._nearest(key)[0] for key in swap_keys(test)],
                          dtype=float)

    def coverage(self, test):
        counts = {"transformation": 0, "nearest_swap_class": 0, "global": 0}
        for key in swap_keys(test):
            counts[self._nearest(key)[1]] += 1
        return counts


def all_baselines():
    """Fresh instances, in the order they should appear in a results table."""
    return [ZeroBaseline(), TrainingMeanBaseline(),
            TransformationMedianBaseline(),
            TransformationEnvironmentMedianBaseline(),
            HeadgroupPropertyBaseline(),
            EnvironmentPropertyBaseline(),
            ShrunkEnvironmentMeanBaseline(),
            ShrunkTargetMeanBaseline(),
            NearestSwapClassMedianBaseline()]


def run_baselines(train, test, baselines=None):
    """Fit each baseline on train and predict on test.

    Returns (predictions, notes): predictions is {name: array}, notes is a list
    of collapse warnings - empty when every baseline is doing what its name says.
    """
    baselines = all_baselines() if baselines is None else baselines
    predictions = {}
    notes = []
    for baseline in baselines:
        missing = [column for column in baseline.required_columns
                   if column not in train.columns or column not in test.columns]
        if missing:
            # Not silent (rule 5): a baseline that cannot run on this table is
            # named, with the columns it wanted. `shrunk_target_mean` needs
            # target_chembl_id, which only exists after the step-1.5 join, so
            # this is how it behaves on the real pair table.
            notes.append(f"{baseline.name}: SKIPPED, the table has no "
                         f"{missing} - this baseline needs columns that the "
                         f"step-1.5 join adds.")
            continue
        baseline.fit(train)
        predictions[baseline.name] = baseline.predict(test)
        message = baseline.describe_collapse(test)
        if message:
            notes.append(message)
    return predictions, notes
