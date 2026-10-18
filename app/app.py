"""Gradio interface for the headgroup swap potency model.

Run locally:      python app/app.py
Hugging Face:     this file is the Space's app_file; nothing needs changing.

Everything the interface claims is computed in app/predict.py, which loads the
five published seeds of one fold and blends them with the shrunk_target_mean
baseline exactly as the published headline does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from app.predict import (  # noqa: E402
    BLEND_WEIGHT, FOLD, HEADLINE_FOLD_ERROR, HEADLINE_MCC, HEADLINE_RMSE,
    MODEL_SEEDS, NOISE_FLOOR, NOISE_FLOOR_FOLD, predict, target_choices,
)

TITLE = "Headgroup swap potency model"

INTRO = f"""
Predicts how much a drug's potency changes when one acidic headgroup is swapped for
another — a carboxylic acid for a tetrazole, say — on a given protein target.

**The typical error is {HEADLINE_FOLD_ERROR}-fold in potency**
(RMSE {HEADLINE_RMSE} log units, three-class MCC +{HEADLINE_MCC:.3f}, measured on
16,116 held-out comparisons whose molecules the model never saw). That is the number to
judge every prediction below by: it is useful for ranking a handful of candidate swaps,
and far too coarse to tell you a compound is worth making.

For reference, two labs measuring *the same* pair disagree by about {NOISE_FLOOR}
log units ({NOISE_FLOOR_FOLD}-fold), so no model built on this data can do much better.
"""

WHY_FINITE = """
**Why the list is finite.** Much of what the model knows is *which protein* the swap is
made on: acid → hydroxamic acid is worth a great deal on a zinc metalloenzyme and little
elsewhere. A protein that was not in training has no learned representation — it falls
back to a reserved "unknown" slot that carries no information about that target — so the
model has no skill there and the list leaves it out rather than offering a confident
number. Free-text target entry is deliberately not available.
"""

# Real comparisons from data/pairs_bioisostere.csv, chosen to differ only at the
# headgroup. Most of them were in training for the loaded models, so the model
# agreeing with the measured value on these is not evidence of anything.
EXAMPLES = [
    ["Cc1c(-c2ccc(-c3ccccc3F)cc2)nc2ccc(F)cc2c1C(=O)O",
     "Cc1c(-c2ccc(-c3ccccc3F)cc2)nc2ccc(F)cc2c1C(=O)NC#N",
     "CHEMBL1966"],      # carboxylic acid -> acyl cyanamide
    ["O=C(O)c1cc(NC(=O)c2cc(C(F)(F)F)cnc2N2CCC(F)(F)CC2)ccn1",
     "O=C(Nc1ccnc(S(=O)(=O)O)c1)c1cc(C(F)(F)F)cnc1N1CCC(F)(F)CC1",
     "CHEMBL5451"],      # carboxylic acid -> sulfonic acid
    ["O=C(CCCc1nnn[nH]1)NC1CCN(C(=O)OCc2cc(Cl)cc(Cl)c2)CC1",
     "O=C(O)CCCCC(=O)NC1CCN(C(=O)OCc2cc(Cl)cc(Cl)c2)CC1",
     "CHEMBL3691"],      # tetrazole -> carboxylic acid
]


def _structures(smiles_a, smiles_b):
    mols, labels = [], []
    for label, smiles in (("A", smiles_a), ("B", smiles_b)):
        mol = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
        if mol is not None:
            mols.append(mol)
            labels.append(label)
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, legends=labels, molsPerRow=2,
                                subImgSize=(320, 240))


def run(smiles_a, smiles_b, target_id):
    """Returns (verdict markdown, per-seed rows, structure image)."""
    image = _structures(smiles_a, smiles_b)
    result = predict(smiles_a, smiles_b, target_id)

    if not result["ok"]:
        return f"### No prediction\n\n{result['message']}", [], image

    delta = result["delta_p"]
    direction = "more potent" if delta > 0 else "less potent"
    lines = [
        f"## Δp = {delta:+.2f} log units",
        f"### {result['call']} — molecule B is {result['fold_change']:.1f}× "
        f"{direction} than A, predicted",
        "",
        f"**Swap read from the structures:** {result['swap']}  ",
        f"**Target:** `{result['target_id']}`",
        "",
        f"> Judge this against the model's typical error of "
        f"**{HEADLINE_FOLD_ERROR}-fold** ({HEADLINE_RMSE} log units). A Δp of "
        f"{delta:+.2f} is {'well inside' if abs(delta) < HEADLINE_RMSE else 'larger than'}"
        f" that error, so it {'does not' if abs(delta) < HEADLINE_RMSE else 'may'} "
        f"indicate a real change.",
        "",
        "### How the five seeds disagree",
        f"The same architecture trained five times with different initialisations gives "
        f"**{result['seed_low']:+.2f} to {result['seed_high']:+.2f}** "
        f"(spread {result['seed_spread']:.2f}, standard deviation {result['seed_sd']:.2f}). "
        f"This spread is the model's own disagreement about *this* pair, and it is not "
        f"included in the {HEADLINE_FOLD_ERROR}-fold figure above. A spread wider than "
        f"the error bar means the five models do not agree on an answer, and the mean "
        f"should not be read as one.",
        "",
        f"### The number is a blend, by design",
        f"`{BLEND_WEIGHT} × ensemble + {1 - BLEND_WEIGHT} × baseline` = "
        f"`{BLEND_WEIGHT} × {result['ensemble']:+.3f} + "
        f"{1 - BLEND_WEIGHT} × {result['baseline']:+.3f}` = **{delta:+.3f}**. "
        f"The baseline is the average measured Δp for this swap on this protein, shrunk "
        f"toward the swap's overall average. The weight 0.5 was fixed on validation data "
        f"before any test fold was scored, not tuned to flatter the result.",
    ]
    for note in result.get("notes", []):
        lines += ["", f"⚠️ {note}"]
    lines += [
        "",
        f"*Predicted by the five seeds of fold {result['fold']}. Those models were "
        f"trained on four fifths of the data; molecules in fold {result['fold']} are "
        f"genuinely unseen, and the rest were in training, so a prediction for a "
        f"molecule already in ChEMBL will tend to look better than the error bar "
        f"suggests.*",
        "",
        "*The interface checks that exactly one headgroup differs. It does **not** "
        "check that the rest of the two molecules is identical — see the limits in "
        "`docs/limits.md`. If they differ elsewhere, this number means nothing.*",
    ]

    rows = [[f"seed {seed}", round(value, 3)]
            for seed, value in result["per_seed"].items()]
    rows.append(["mean of the five (ensemble)", round(result["ensemble"], 3)])
    rows.append(["shrunk_target_mean baseline", round(result["baseline"], 3)])
    rows.append(["blend — the prediction", round(delta, 3)])
    return "\n".join(lines), rows, image


def build():
    choices = target_choices(FOLD)
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=3):
                smiles_a = gr.Textbox(
                    label="Molecule A (SMILES)", placeholder="OC(=O)CCc1ccc(Cl)cc1",
                    info="The parent, usually the one bearing the carboxylic acid.")
                smiles_b = gr.Textbox(
                    label="Molecule B (SMILES)", placeholder="Clc1ccc(CCc2nnn[nH]2)cc1",
                    info="Identical to A except for the headgroup.")
                target = gr.Dropdown(
                    label=f"Protein target — {len(choices):,} the model trained on",
                    choices=choices, value=choices[0][1] if choices else None,
                    filterable=True,
                    info="Type to search. Sorted by how many training pairs each has.")
                gr.Markdown(WHY_FINITE)
                go = gr.Button("Predict Δp", variant="primary")
            with gr.Column(scale=4):
                verdict = gr.Markdown("Enter two molecules and pick a target.")
                table = gr.Dataframe(headers=["prediction", "Δp (log units)"],
                                     label="Every number behind the answer",
                                     interactive=False, wrap=True)
                picture = gr.Image(label="What the model read", type="pil")
        gr.Examples(
            examples=EXAMPLES, inputs=[smiles_a, smiles_b, target],
            label="Real comparisons from the dataset — click one, then Predict "
                  "(most were in training for the loaded models, so agreement "
                  "with the measured value would not be evidence of accuracy)")
        gr.Markdown(
            f"Model: five seeds (`{', '.join(str(s) for s in MODEL_SEEDS)}`) of a "
            f"D-MPNN delta model, trained on all pair classes for 3 epochs, blended "
            f"50/50 with `shrunk_target_mean`. Data derived from ChEMBL 37 "
            f"(CC BY-SA 3.0); code MIT. Full limits in `docs/limits.md`."
        )
        go.click(run, inputs=[smiles_a, smiles_b, target],
                 outputs=[verdict, table, picture])
    return demo


if __name__ == "__main__":
    build().launch()
