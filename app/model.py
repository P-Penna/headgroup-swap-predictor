"""The Siamese delta model. Architecture only - no training loop.

WHAT IT PREDICTS
----------------
One item is (molecule A, molecule B, protein T), and the model predicts
Delta p = pIC50(B) - pIC50(A) plus a three-class call (worse / same / better).
Inputs come from app/pair_dataset.py's `collate_pairs`.

ONE SHARED ENCODER
------------------
A single chemprop D-MPNN (`BondMessagePassing`, hidden size 300) is created
once in __init__ and called on both molecules. Not two encoders with tied
weights - the same object, which a test asserts by identity. The molecules
differ only in a headgroup, so an asymmetric pair of encoders could learn
"side A is usually the acid" and we would never see it in the loss.

WHY ANTISYMMETRY HAS TO BE BUILT IN
-----------------------------------
Delta p is antisymmetric by definition: swapping A and B negates it. The data
already says so - pairs.csv holds both directions of all 57,184 bioisostere
rows - but training on both only encourages the symmetry, it
does not guarantee it, and an untrained model violates it.

Enforcing it exactly constrains the architecture hard. Write z(A, B) for the
fusion vector and look at how each block behaves when A and B swap:

    h_A - h_B        NEGATES          (antisymmetric - usable directly)
    h_A * h_B        unchanged        (symmetric)
    h_A, h_B         exchange         (neither)
    swap_emb         becomes the reverse swap's embedding
    family, target   unchanged        (symmetric)

A plain bias-free linear head on a concatenation is antisymmetric only if it
ignores every symmetric block, which would delete the family and target
conditioning - the only conditioning that measurably works - and a ReLU trunk is illegal too, because ReLU is not odd, so
MLP(-z) != -MLP(z).

THE CONSTRUCTION USED: AN ANTISYMMETRIZED TRUNK
-----------------------------------------------
Evaluate the whole trunk both ways and subtract:

    g      = trunk(z(A, B))            z = [h_A, h_B, h_A - h_B, h_A * h_B,
    g_rev  = trunk(z(B, A))                 swap, family, target]
    delta  = head(g) - head(g_rev)

Swapping A and B exchanges g and g_rev, so `delta` negates exactly - for any
trunk, any activation, trained or not, to float precision. Three consequences
worth stating:

  - every feature stays usable, including the symmetric ones and all three
    conditioning embeddings, because the subtraction does the work rather than
    a restriction on the features;
  - f(A, A) = 0 identically, since z(A, A) = z(A, A) makes the two terms equal.
    That is the identity-pair label satisfied by construction, not learned;
  - it costs a SECOND TRUNK PASS per item. The encoder already runs twice
    (once per molecule) and is by far the expensive part: measured on CPU, a
    forward pass is 2.18 ms of which the encoder is 1.87 ms and BOTH trunk
    passes together are 0.25 ms (12%), so the extra pass costs about 6% of a
    forward pass. 794,028 parameters in total, 227,700 of them the encoder.

The mirrored pass uses the reverse swap's own embedding
(`reverse_swap_idx` from the loader), not the negation of the forward one,
because the two directions are separate vocabulary entries and nothing ties
them.

WHAT GOES INTO THE FUSION, AND WHY NOT h_A AND h_B THEMSELVES
-------------------------------------------------------------
The default (`fusion="difference_product"`) feeds the trunk only
`h_A - h_B` and `h_A * h_B`, not the two molecule representations separately.
Two reasons, one of them measured:

  - the model predicts a DIFFERENCE, and the signal that has survived every
    honest split is in the swap and the protein, not in either molecule's
    absolute representation;
  - keeping h_A and h_B separately lets the trunk memorise molecule identity,
    which is the route that inflated target identity from +10.08 to +5.69
    points when the folds were made molecule-disjoint.
    That route was measured directly: on the
    loose constant split 56.5% of test comparisons shared a molecule with a
    training row in the same (swap class x protein) cell.

It also halves the fusion width (680 against 1,280 on the real vocabularies)
and moves the parameter balance toward the encoder, which is the part that
should be doing the chemistry: the trunk falls from 69% of parameters to 60%,
and the encoder rises from 28% to 36%.

`fusion="full"` keeps the old concatenation. Both are kept behind the flag and
compared on identical folds; if the full concatenation wins by a
real margin it stays. Nothing here assumes the default is better - it is the
one with the better prior and the smaller leakage surface.

THE CLASSIFICATION HEAD
-----------------------
Swapping A and B must swap P(worse) and P(better) and leave P(same) alone.
Built from two channels off the same pair of trunk passes:

    odd  = c(g) - c(g_rev)      negates under swap
    even = s(g) + s(g_rev)      unchanged under swap
    logits = [odd, even, -odd]  -> worse, same, better

Softmax is monotone and shift-invariant, so swapping A and B exchanges the
worse and better logits exactly and leaves same untouched: the probabilities
permute. "same" sits in the middle and is driven only by the even channel,
which is what "this swap changes little in either direction" should mean.

DROPOUT HAD TO BE PAIRED, AND THE ENCODER'S TURNED OFF
------------------------------------------------------
Ordinary dropout samples an independent mask per element, so the trunk would
use DIFFERENT masks for its two passes and the subtraction would no longer
cancel: antisymmetry would hold in eval mode and quietly fail in train mode,
which is the worse of the two failure modes because training is where it
matters. Two fixes, both needed:

  - `PairedDropout` samples ONE mask and applies it to both passes, so the
    trunk stays exactly antisymmetric while training;
  - the ENCODER runs with dropout 0 (chemprop's own default). Molecule A and
    molecule B are two separate encoder calls, so any encoder dropout would
    give h_A != h_B for an identity pair (A, A) and break f(A, A) = 0 in train
    mode. Dropout 0.1 therefore lives in the trunk, as specified, and not in
    the message passing.

Both are covered by tests that run in train mode, not just eval.

INDEX 0 IS TRAINABLE
--------------------
Each embedding reserves index 0 for a value absent from training (an unseen
protein, family or swap). It is an ordinary trainable row, NOT
a zero vector and NOT padding_idx: target dropout blanks the
protein on a fraction of training rows precisely so this row gets gradients
and the model degrades gracefully on a new protein. A test checks the
gradient arrives.
"""

import torch
from torch import nn

from chemprop.nn import BondMessagePassing, MeanAggregation

HIDDEN = 300            # chemprop's default D-MPNN width, and ours
SWAP_DIM = 32
FAMILY_DIM = 16
TARGET_DIM = 32
PAIR_CLASS_DIM = 8      # only when trained on several pair classes
TRUNK_HIDDEN = 300
DROPOUT = 0.1
N_CLASSES = 3           # worse / same / better, in that order

# What goes into the fusion vector, besides the three conditioning embeddings.
#   "difference_product"  h_A - h_B and h_A * h_B only. THE DEFAULT.
#   "full"                those plus h_A and h_B separately.
# Kept as a flag to be compared on real folds rather than settled by argument;
# see FUSION in the module docstring.
FUSION_TERMS = ("difference_product", "full")
WORSE, SAME, BETTER = 0, 1, 2


class PairedDropout(nn.Module):
    """Dropout that uses ONE mask for both halves of a paired batch.

    The trunk is fed `cat([z(A, B), z(B, A)])`, so rows i and i + n are the two
    orderings of the same item and MUST be masked identically - otherwise the
    two trunk passes differ by noise and the antisymmetric subtraction stops
    being exact (see the module docstring). Standard nn.Dropout cannot do this:
    it samples per element.

    Inverted dropout, as nn.Dropout does it: survivors are scaled by
    1 / (1 - p) so the expected sum is unchanged and eval mode needs no rescale.
    """

    def __init__(self, p=DROPOUT):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if x.shape[0] % 2 != 0:
            raise ValueError(f"PairedDropout needs an even batch (forward rows "
                             f"then reverse rows); got {x.shape[0]}")
        half = x.shape[0] // 2
        keep = torch.rand(half, *x.shape[1:], device=x.device, dtype=x.dtype) >= self.p
        mask = keep.to(x.dtype) / (1.0 - self.p)
        return x * torch.cat([mask, mask], dim=0)


class SiameseDelta(nn.Module):
    """Exactly antisymmetric Delta p regression plus a three-class head.

    n_swaps / n_families / n_targets are vocabulary sizes INCLUDING index 0,
    i.e. `len(vocab) + 1` for a vocabulary built by
    app/pair_dataset.build_vocabs (which numbers its values from 1).
    """

    def __init__(self, n_swaps, n_families, n_targets, hidden=HIDDEN,
                 trunk_hidden=TRUNK_HIDDEN, dropout=DROPOUT, depth=3,
                 fusion="difference_product", n_pair_classes=0):
        super().__init__()
        if fusion not in FUSION_TERMS:
            raise ValueError(f"fusion must be one of {FUSION_TERMS}; got {fusion!r}")
        self.fusion = fusion
        for name, size in (("n_swaps", n_swaps), ("n_families", n_families),
                           ("n_targets", n_targets)):
            if size < 2:
                raise ValueError(f"{name} must be at least 2 (index 0 is the "
                                 f"unseen value, plus one real value); got {size}")

        # ONE encoder object, called on both molecules. The test asserts the
        # identity, but keeping a single attribute is what makes it true.
        # dropout stays at chemprop's default 0: see the module docstring.
        self.encoder = BondMessagePassing(d_h=hidden, depth=depth)
        self.aggregate = MeanAggregation()

        # No padding_idx anywhere: index 0 must learn (see the module docstring).
        self.swap_embedding = nn.Embedding(n_swaps, SWAP_DIM)
        self.family_embedding = nn.Embedding(n_families, FAMILY_DIM)
        self.target_embedding = nn.Embedding(n_targets, TARGET_DIM)
        # Which kind of pair (bioisostere, same_headgroup, to_non_acid,
        # deletion), so the model need not average them. 0 (the default) means
        # no such input: a bioisostere-only model is built exactly as before,
        # and every earlier checkpoint still loads. Symmetric, like family and
        # target: the class of (A, B) is the class of (B, A).
        self.n_pair_classes = n_pair_classes
        self.pair_class_embedding = (nn.Embedding(n_pair_classes, PAIR_CLASS_DIM)
                                     if n_pair_classes else None)

        molecule_terms = 4 if fusion == "full" else 2
        fusion_dim = molecule_terms * hidden + SWAP_DIM + FAMILY_DIM + TARGET_DIM
        if n_pair_classes:
            fusion_dim += PAIR_CLASS_DIM
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, trunk_hidden), nn.ReLU(), PairedDropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(), PairedDropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(), PairedDropout(dropout))

        # Bias-free. A bias would in fact be HARMLESS - it cancels in
        # head(g) - head(g_rev), checked by mutation: adding one leaves
        # antisymmetry exact - but a parameter that provably cannot affect the
        # output is better not created than explained.
        self.delta_head = nn.Linear(trunk_hidden, 1, bias=False)
        self.direction_head = nn.Linear(trunk_hidden, 1, bias=False)  # odd channel
        self.same_head = nn.Linear(trunk_hidden, 1, bias=False)       # even channel

    # --- the pieces, kept separate so tests can reach them ------------------

    def embed_molecules(self, graph_a, graph_b):
        """(h_A, h_B) from the one shared encoder."""
        return (self.aggregate(self.encoder(graph_a), graph_a.batch),
                self.aggregate(self.encoder(graph_b), graph_b.batch))

    def fuse(self, h_first, h_second, swap_idx, family_idx, target_idx,
             pair_class_idx=None):
        """The fusion vector z for one ordering of the molecules.

        The difference negates when the molecules swap and the product does
        not; under `fusion="full"` the two separate representations exchange.
        The trunk subtraction handles all three, so the choice here is about
        what the model may LEARN FROM, not about antisymmetry.
        """
        terms = []
        if self.fusion == "full":
            terms += [h_first, h_second]
        terms += [h_first - h_second, h_first * h_second,
                  self.swap_embedding(swap_idx),
                  self.family_embedding(family_idx),
                  self.target_embedding(target_idx)]
        if self.pair_class_embedding is not None:
            terms.append(self.pair_class_embedding(pair_class_idx))
        return torch.cat(terms, dim=1)

    def trunk_both_ways(self, h_a, h_b, swap_idx, reverse_swap_idx, family_idx,
                        target_idx, pair_class_idx=None):
        """(g, g_rev): the trunk on (A, B) and on (B, A).

        The two orderings go through as ONE batch, forward rows first, because
        `PairedDropout` reads that layout to give both passes the same mask.
        Splitting this into two trunk calls would re-break train-mode
        antisymmetry.
        """
        forward = self.fuse(h_a, h_b, swap_idx, family_idx, target_idx, pair_class_idx)
        reverse = self.fuse(h_b, h_a, reverse_swap_idx, family_idx, target_idx,
                            pair_class_idx)
        stacked = self.trunk(torch.cat([forward, reverse], dim=0))
        return stacked[:len(forward)], stacked[len(forward):]

    def forward(self, batch):
        """{delta, logits}: delta (n,) and logits (n, 3) in worse/same/better order.

        `batch` is what app/pair_dataset.collate_pairs returns.
        """
        h_a, h_b = self.embed_molecules(batch["graph_a"], batch["graph_b"])
        g, g_rev = self.trunk_both_ways(
            h_a, h_b, batch["swap_idx"], batch["reverse_swap_idx"],
            batch["family_idx"], batch["target_idx"], batch.get("pair_class_idx"))

        delta = (self.delta_head(g) - self.delta_head(g_rev)).squeeze(1)
        odd = (self.direction_head(g) - self.direction_head(g_rev)).squeeze(1)
        even = (self.same_head(g) + self.same_head(g_rev)).squeeze(1)
        # worse, same, better. Negating `odd` is what makes a swap exchange the
        # outer two classes while `even` holds the middle one fixed.
        logits = torch.stack([odd, even, -odd], dim=1)
        return {"delta": delta, "logits": logits}

    # --- facts about the model, for the log and the tests -------------------

    def parameter_counts(self):
        """{part: trainable parameters}, so a change in size is visible."""
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in (("encoder", self.encoder),
                                       ("swap_embedding", self.swap_embedding),
                                       ("family_embedding", self.family_embedding),
                                       ("target_embedding", self.target_embedding),
                                       ("pair_class_embedding",
                                        self.pair_class_embedding or nn.Identity()),
                                       ("trunk", self.trunk),
                                       ("delta_head", self.delta_head),
                                       ("direction_head", self.direction_head),
                                       ("same_head", self.same_head))}
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
