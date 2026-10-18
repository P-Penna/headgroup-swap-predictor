"""Tests for app/model.py: the architectural guarantees, on an UNTRAINED model.

Every claim the architecture makes is checked here before any training exists,
because each one is a property of the construction rather than of a fitted
model - if it needs training to hold, it is not a guarantee:

  - the encoder used for A is the SAME OBJECT as the one used for B;
  - f(A, B) = -f(A, B) swapped, to float precision, on both heads;
  - f(A, A) = 0 exactly;
  - that both of those survive TRAIN mode, where dropout is live;
  - index 0 of every embedding receives gradient.

The molecules are real - chemprop parses SMILES - and small, so the whole file
runs in a couple of seconds.
"""

import pytest
import torch

from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.utils import make_mol

from app.model import (
    BETTER, FUSION_TERMS, SAME, WORSE, PairedDropout, SiameseDelta,
)
from app.pair_dataset import UNKNOWN

ACID = "OC(=O)c1ccccc1"
TETRAZOLE = "c1ccccc1-c1nnn[nH]1"
ACYLSULFONAMIDE = "CS(=O)(=O)NC(=O)c1ccccc1"
MOLECULES = [ACID, TETRAZOLE, ACYLSULFONAMIDE, "OC(=O)c1ccncc1"]

N_SWAPS, N_FAMILIES, N_TARGETS = 8, 5, 6
FEATURIZER = SimpleMoleculeMolGraphFeaturizer()


def graphs(smiles_list):
    return BatchMolGraph([FEATURIZER(make_mol(smiles)) for smiles in smiles_list])


def batch(a_smiles, b_smiles, swap, reverse_swap, family, target):
    """One batch in collate_pairs' shape, built by hand so a test can swap it."""
    return {"graph_a": graphs(a_smiles), "graph_b": graphs(b_smiles),
            "swap_idx": torch.tensor(swap), "reverse_swap_idx": torch.tensor(reverse_swap),
            "family_idx": torch.tensor(family), "target_idx": torch.tensor(target)}


def swapped(data):
    """The same comparisons written backwards: A<->B and swap<->reverse swap."""
    return {"graph_a": data["graph_b"], "graph_b": data["graph_a"],
            "swap_idx": data["reverse_swap_idx"],
            "reverse_swap_idx": data["swap_idx"],
            "family_idx": data["family_idx"], "target_idx": data["target_idx"]}


@pytest.fixture(params=FUSION_TERMS)
def model(request):
    """Every guarantee is checked for BOTH fusion variants, since the flag
    exists to be compared on real folds and a guarantee that holds for only one
    of them is not a guarantee."""
    torch.manual_seed(42)
    return SiameseDelta(N_SWAPS, N_FAMILIES, N_TARGETS,
                        fusion=request.param).eval()


@pytest.fixture
def data():
    return batch([ACID, TETRAZOLE, ACID], [TETRAZOLE, ACYLSULFONAMIDE, ACYLSULFONAMIDE],
                 swap=[1, 2, 3], reverse_swap=[4, 5, 6], family=[1, 2, 0],
                 target=[1, 0, 2])


# --- the shared encoder -----------------------------------------------------

def test_one_encoder_object_serves_both_molecules(model):
    """Not two encoders with tied weights - the same object."""
    seen = []
    original = model.encoder.forward

    def record(*args, **kwargs):
        seen.append(id(model.encoder))
        return original(*args, **kwargs)

    model.encoder.forward = record
    model(batch([ACID], [TETRAZOLE], [1], [2], [1], [1]))
    assert len(seen) == 2, "the encoder should be called once per molecule"
    assert seen[0] == seen[1] == id(model.encoder)


def test_encoder_has_no_dropout(model):
    """Encoder dropout would make h_A != h_B for an identity pair in train mode."""
    dropouts = [module.p for module in model.encoder.modules()
                if isinstance(module, torch.nn.Dropout)]
    assert all(p == 0.0 for p in dropouts), dropouts


# --- exact antisymmetry -----------------------------------------------------

def test_delta_is_exactly_antisymmetric_untrained(model, data):
    forward = model(data)["delta"]
    reverse = model(swapped(data))["delta"]
    assert torch.allclose(forward, -reverse, atol=0.0, rtol=0.0), \
        (forward, reverse)


def test_logits_swap_worse_and_better_and_fix_same(model, data):
    forward = model(data)["logits"]
    reverse = model(swapped(data))["logits"]
    assert torch.allclose(forward[:, WORSE], reverse[:, BETTER], atol=0.0, rtol=0.0)
    assert torch.allclose(forward[:, BETTER], reverse[:, WORSE], atol=0.0, rtol=0.0)
    assert torch.allclose(forward[:, SAME], reverse[:, SAME], atol=0.0, rtol=0.0)


def test_class_probabilities_permute_under_swap(model, data):
    """The guarantee as a user would state it, in probabilities not logits."""
    forward = model(data)["logits"].softmax(dim=1)
    reverse = model(swapped(data))["logits"].softmax(dim=1)
    assert torch.allclose(forward[:, WORSE], reverse[:, BETTER])
    assert torch.allclose(forward[:, SAME], reverse[:, SAME])


def test_antisymmetry_holds_in_train_mode(model, data):
    """Dropout is live here; PairedDropout is what keeps this exact."""
    model.train()
    torch.manual_seed(7)
    forward = model(data)["delta"]
    torch.manual_seed(7)
    reverse = model(swapped(data))["delta"]
    assert torch.allclose(forward, -reverse, atol=0.0, rtol=0.0)


def test_identity_pair_predicts_exactly_zero(model):
    """f(A, A) = 0 by construction, in both modes, for identity pairs - and the
    'same' class must win by symmetry of the outer two."""
    identity = batch(MOLECULES, MOLECULES, swap=[1, 1, 2, 2], reverse_swap=[1, 1, 2, 2],
                     family=[1, 2, 0, 1], target=[1, 0, 2, 2])
    for mode in ("eval", "train"):
        getattr(model, mode)()
        out = model(identity)
        assert torch.equal(out["delta"], torch.zeros(len(MOLECULES)))
        assert torch.equal(out["logits"][:, WORSE], out["logits"][:, BETTER])


def test_identity_pair_needs_the_same_swap_index_both_ways(model):
    """A real (M, M) item has swap == reverse_swap (acid->acid both ways). If a
    caller passes different indices it is not an identity pair and zero is not
    promised: this pins that the zero comes from the construction, not luck."""
    mismatched = batch([ACID], [ACID], swap=[1], reverse_swap=[2], family=[1],
                       target=[1])
    assert model(mismatched)["delta"].abs().item() > 0.0


# --- the paired dropout -----------------------------------------------------

def test_paired_dropout_masks_both_halves_identically():
    layer = PairedDropout(0.5).train()
    x = torch.ones(8, 16)
    out = layer(x)
    assert torch.equal(out[:4], out[4:])
    assert (out == 0.0).any(), "p=0.5 should drop something in 64 values"


def test_paired_dropout_is_identity_in_eval():
    layer = PairedDropout(0.5).eval()
    x = torch.randn(4, 3)
    assert torch.equal(layer(x), x)


def test_paired_dropout_refuses_an_odd_batch():
    with pytest.raises(ValueError, match="even batch"):
        PairedDropout(0.5).train()(torch.ones(3, 2))


# --- shapes, sizes and gradients --------------------------------------------

def test_output_shapes(model, data):
    out = model(data)
    assert out["delta"].shape == (3,)
    assert out["logits"].shape == (3, 3)


def test_parameter_counts(model):
    counts = model.parameter_counts()
    assert counts["encoder"] == 227_700
    assert counts["swap_embedding"] == N_SWAPS * 32
    assert counts["family_embedding"] == N_FAMILIES * 16
    assert counts["target_embedding"] == N_TARGETS * 32
    # Embeddings are 32 + 16 + 32 = 80 wide; molecule terms are 300 each.
    width = (4 if model.fusion == "full" else 2) * 300 + 80
    assert counts["trunk"] == (width * 300 + 300) + 2 * (300 * 300 + 300)
    for head in ("delta_head", "direction_head", "same_head"):
        assert counts[head] == 300, "heads are bias-free"
    assert counts["total"] == sum(value for name, value in counts.items()
                                  if name != "total")


def test_the_default_fusion_drops_the_separate_molecule_terms():
    """The default is the narrower, lower-leakage one (decision on 4.2)."""
    assert SiameseDelta(N_SWAPS, N_FAMILIES, N_TARGETS).fusion == "difference_product"


def test_dropping_the_separate_terms_shrinks_the_trunk():
    narrow = SiameseDelta(N_SWAPS, N_FAMILIES, N_TARGETS).parameter_counts()
    full = SiameseDelta(N_SWAPS, N_FAMILIES, N_TARGETS,
                        fusion="full").parameter_counts()
    assert narrow["trunk"] < full["trunk"]
    assert narrow["encoder"] == full["encoder"]
    # The encoder's share of the model rises, which is the point.
    assert (narrow["encoder"] / narrow["total"]
            > full["encoder"] / full["total"])


def test_both_fusions_see_the_same_conditioning(model):
    """Neither variant may drop a conditioning embedding."""
    out = model(batch([ACID], [TETRAZOLE], [1], [2], [1], [1]))
    out["delta"].sum().backward()
    for name in ("swap_embedding", "family_embedding", "target_embedding"):
        grad = getattr(model, name).weight.grad
        assert grad is not None and grad.abs().sum().item() > 0.0, name


def test_an_unknown_fusion_is_refused():
    with pytest.raises(ValueError, match="fusion must be one of"):
        SiameseDelta(N_SWAPS, N_FAMILIES, N_TARGETS, fusion="concat")


def test_index_zero_embeddings_receive_gradient(model):
    """Target dropout trains index 0, so it must not be padding."""
    unseen = batch([ACID], [TETRAZOLE], swap=[UNKNOWN], reverse_swap=[UNKNOWN],
                   family=[UNKNOWN], target=[UNKNOWN])
    model.train()
    model(unseen)["delta"].sum().backward()
    for name in ("swap_embedding", "family_embedding", "target_embedding"):
        grad = getattr(model, name).weight.grad
        assert grad is not None, name
        assert grad[UNKNOWN].abs().sum().item() > 0.0, f"{name} index 0 got no gradient"


def test_a_real_index_gets_gradient_too(model):
    present = batch([ACID], [TETRAZOLE], swap=[1], reverse_swap=[2], family=[1],
                    target=[1])
    model.train()
    model(present)["delta"].sum().backward()
    assert model.target_embedding.weight.grad[1].abs().sum().item() > 0.0
    assert model.target_embedding.weight.grad[UNKNOWN].abs().sum().item() == 0.0


def test_vocabulary_sizes_must_leave_room_for_index_zero():
    with pytest.raises(ValueError, match="n_targets"):
        SiameseDelta(N_SWAPS, N_FAMILIES, 1)
