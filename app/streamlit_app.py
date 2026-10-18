"""Streamlit interface for the headgroup swap potency model.

Run locally:            streamlit run app/streamlit_app.py
Streamlit Community
Cloud:                  point the app at this file; requirements.txt is at the
                        repository root where the platform expects it.

This is the same model, the same checks and the same numbers as the Gradio
interface in app/app.py: both are thin presentation layers over app/predict.py,
so neither can drift from the other or from the published result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from rdkit import Chem  # noqa: E402

# rdkit.Chem.Draw links against X11 libraries (libXrender and friends). They are
# listed in packages.txt so a hosted container has them, but a host that ignores
# that file would raise ImportError here and take the whole app down over a
# picture. The structures are a convenience; the prediction is the point.
try:
    from rdkit.Chem import Draw  # noqa: E402
    DRAWING_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the host's system libraries
    Draw = None
    DRAWING_AVAILABLE = False

from app.predict import (  # noqa: E402
    BLEND_WEIGHT, FOLD, HEADLINE_FOLD_ERROR, HEADLINE_MCC, HEADLINE_RMSE,
    NOISE_FLOOR, NOISE_FLOOR_FOLD, predict, target_choices,
)

st.set_page_config(page_title="Headgroup swap potency model",
                   page_icon="🧪", layout="wide")

# Real comparisons from the dataset, differing only at the headgroup. Most were
# in training for the loaded models, so agreement here is not evidence.
EXAMPLES = {
    "carboxylic acid → acyl cyanamide": (
        "Cc1c(-c2ccc(-c3ccccc3F)cc2)nc2ccc(F)cc2c1C(=O)O",
        "Cc1c(-c2ccc(-c3ccccc3F)cc2)nc2ccc(F)cc2c1C(=O)NC#N",
        "CHEMBL1966"),
    "carboxylic acid → sulfonic acid": (
        "O=C(O)c1cc(NC(=O)c2cc(C(F)(F)F)cnc2N2CCC(F)(F)CC2)ccn1",
        "O=C(Nc1ccnc(S(=O)(=O)O)c1)c1cc(C(F)(F)F)cnc1N1CCC(F)(F)CC1",
        "CHEMBL5451"),
    "tetrazole → carboxylic acid": (
        "O=C(CCCc1nnn[nH]1)NC1CCN(C(=O)OCc2cc(Cl)cc(Cl)c2)CC1",
        "O=C(O)CCCCC(=O)NC1CCN(C(=O)OCc2cc(Cl)cc(Cl)c2)CC1",
        "CHEMBL3691"),
}


@st.cache_resource(show_spinner="Loading the five models and fitting the baseline…")
def _targets():
    """Loaded once per container, not once per interaction."""
    return target_choices(FOLD)


def _structure_image(smiles_a, smiles_b):
    if not DRAWING_AVAILABLE:
        return None
    mols, labels = [], []
    for label, smiles in (("A", smiles_a), ("B", smiles_b)):
        mol = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
        if mol is not None:
            mols.append(mol)
            labels.append(label)
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, legends=labels, molsPerRow=2,
                                subImgSize=(340, 250))


def main():
    st.title("Headgroup swap potency model")
    st.markdown(
        f"Predicts how much a drug's potency changes when one acidic headgroup is "
        f"swapped for another — a carboxylic acid for a tetrazole, say — on a given "
        f"protein target.\n\n"
        f"**The typical error is {HEADLINE_FOLD_ERROR}-fold in potency** "
        f"(RMSE {HEADLINE_RMSE} log units, three-class MCC +{HEADLINE_MCC:.3f}), "
        f"measured on 16,116 held-out comparisons whose molecules the model never saw. "
        f"That is the number to judge every prediction below by: useful for ranking a "
        f"handful of candidate swaps, far too coarse to tell you a compound is worth "
        f"making. Two labs measuring *the same* pair disagree by about {NOISE_FLOOR} "
        f"log units ({NOISE_FLOOR_FOLD}×), so no model built on this data can do much "
        f"better.")

    targets = _targets()
    labels = [label for label, _ in targets]
    by_label = dict(targets)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Input")
        choice = st.selectbox("Start from an example (optional)",
                              ["— type my own —"] + list(EXAMPLES))
        preset = EXAMPLES.get(choice, ("", "", None))

        smiles_a = st.text_input(
            "Molecule A (SMILES)", value=preset[0],
            help="The parent, usually the one bearing the carboxylic acid.")
        smiles_b = st.text_input(
            "Molecule B (SMILES)", value=preset[1],
            help="Identical to A except for the headgroup.")

        index = 0
        if preset[2] is not None:
            for position, (_, target_id) in enumerate(targets):
                if target_id == preset[2]:
                    index = position
                    break
        target_label = st.selectbox(
            f"Protein target — {len(targets):,} the model trained on",
            labels, index=index,
            help="Type to search. Sorted by how many training pairs each target has.")

        st.caption(
            "**Why the list is finite.** Much of what the model knows is *which "
            "protein* the swap is made on: acid → hydroxamic acid is worth a great "
            "deal on a zinc metalloenzyme and little elsewhere. A protein that was "
            "not in training has no learned representation — it falls back to a "
            "reserved \"unknown\" slot carrying no information about that target — so "
            "the model has no skill there and the list leaves it out rather than "
            "offering a confident number. Free-text target entry is deliberately not "
            "available.")

        go = st.button("Predict Δp", type="primary", width='stretch')

    with right:
        st.subheader("Result")
        if not go:
            st.info("Enter two molecules and pick a target, then press **Predict Δp**.")
            return

        result = predict(smiles_a, smiles_b, by_label[target_label])
        image = _structure_image(smiles_a, smiles_b)

        if not result["ok"]:
            st.error(f"**No prediction.** {result['message']}")
            if image is not None:
                st.image(image, caption="What the model read")
            return

        delta = result["delta_p"]
        direction = "more potent" if delta > 0 else "less potent"
        st.metric(label=f"Predicted Δp — molecule B is {result['fold_change']:.1f}× "
                        f"{direction}",
                  value=f"{delta:+.2f} log units",
                  delta=result["call"], delta_color="off")

        if abs(delta) < HEADLINE_RMSE:
            st.warning(
                f"This prediction ({delta:+.2f}) is **smaller than the model's own "
                f"typical error** ({HEADLINE_RMSE}), so it should not be read as a "
                f"real change.")

        st.caption(f"Swap read from the structures: **{result['swap']}** · "
                   f"target `{result['target_id']}`")

        st.markdown("#### How the five seeds disagree")
        st.markdown(
            f"The same architecture trained five times with different initialisations "
            f"gives **{result['seed_low']:+.2f} to {result['seed_high']:+.2f}** "
            f"(spread {result['seed_spread']:.2f}, standard deviation "
            f"{result['seed_sd']:.2f}). This is the model's disagreement about *this* "
            f"pair and is **not** included in the {HEADLINE_FOLD_ERROR}-fold figure "
            f"above.")
        if result["seed_spread"] > HEADLINE_RMSE:
            st.warning(
                "The five models span more than the error bar, so they do not agree on "
                "an answer here and the mean should not be read as one. That usually "
                "means the pair is unlike anything in training.")

        rows = [{"prediction": f"seed {seed}", "Δp (log units)": round(value, 3)}
                for seed, value in result["per_seed"].items()]
        rows += [
            {"prediction": "mean of the five (ensemble)",
             "Δp (log units)": round(result["ensemble"], 3)},
            {"prediction": "shrunk_target_mean baseline",
             "Δp (log units)": round(result["baseline"], 3)},
            {"prediction": "blend — the prediction", "Δp (log units)": round(delta, 3)},
        ]
        st.dataframe(rows, hide_index=True, width='stretch')

        st.markdown(
            f"**The number is a blend, by design.** "
            f"`{BLEND_WEIGHT} × ensemble + {1 - BLEND_WEIGHT} × baseline` = "
            f"`{BLEND_WEIGHT} × {result['ensemble']:+.3f} + "
            f"{1 - BLEND_WEIGHT} × {result['baseline']:+.3f}` = **{delta:+.3f}**. "
            f"The baseline is the average measured Δp for this swap on this protein, "
            f"shrunk toward the swap's overall average. The weight 0.5 was fixed on "
            f"validation data before any test fold was scored.")

        for note in result.get("notes", []):
            st.warning(note)

        if image is not None:
            st.image(image, caption="What the model read")

        st.caption(
            f"Predicted by the five seeds of fold {result['fold']}, which were trained "
            f"on four fifths of the data. Molecules in fold {result['fold']} are "
            f"genuinely unseen; the rest were in training, so a prediction for a "
            f"molecule already in ChEMBL will tend to look better than the error bar "
            f"suggests. The interface checks that exactly one headgroup differs; it "
            f"does **not** check that the rest of the two molecules is identical — see "
            f"`docs/limits.md`.")


if __name__ == "__main__":
    main()
