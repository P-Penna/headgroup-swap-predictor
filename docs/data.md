# Data

## What is in this repository

| file | size | what it is |
|---|---|---|
| `data/pairs_bioisostere.csv` | 22 MB | 57,184 rows — 20,210 distinct bioisostere comparisons, each stored in both directions (A→B and B→A) |
| `data/target_properties.csv` | 1 MB | ChEMBL protein family (levels 1–3), organism and metal-binding flag for every target |

Both are derived from **ChEMBL 37** and carry **CC BY-SA 3.0** — see `LICENSE`.

Each molecule's potency is the median of its measurements on that target and assay type,
and Δp is only taken within one assay type. Values that were really placeholders rather
than measurements have been censored out.

`pairs_bioisostere.csv` is everything the interface needs: it fits the
`shrunk_target_mean` baseline, builds the target dropdown and reproduces the fold split.

### Columns you will actually use

`parent_a_id`, `parent_b_id` — ChEMBL molecule ids. `smiles_a`, `smiles_b` — the two
structures. `headgroup_a`, `headgroup_b` — which acidic group each carries, by the keys in
`app/headgroups.py`. `target_chembl_id`, `family_l2` — the protein and its family.
`delta_p` — the label. `censored_a`, `censored_b` — whether each potency was an exact
measurement or a bound. `constant_smiles` — the part that did not change, which is what
the fold split is computed on. `pair_class` — always `bioisostere` in this file.

## The full pairs.csv

The published model was trained on **all** pair classes, not just bioisosteres: headgroup
changes that are not bioisosteric, deletions, and same-headgroup pairs as well. That file
is **1,291,071 rows, 620 MB uncompressed and about 41 MB gzipped**, which is too large to
keep in git, so it lives on Zenodo.

> **DOI:** _to be filled in once the record is published — see "Depositing it" below._
>
> Download `pairs.csv.gz` into `data/`. Nothing needs decompressing; pandas reads
> `.csv.gz` directly, so the default `--all-pairs data/pairs.csv.gz` works as-is.

With it you can retrain the published model:

```
python app/train.py --split constant --fold 0 --no-val --class-weight 0 \
    --pair-classes all --epochs 3 --fit-every 2000 --seed 42 --model-seed 1
```

Without it you can still retrain the bioisostere-only variant, run the interface, run the
tests and redraw the figures — everything except reproducing the headline model from
scratch.

`pairs.csv` has the same columns plus a populated `pair_class`
(`bioisostere`, `same_headgroup`, `to_non_acid`, `deletion`, and `not_a_swap` for the
2,090 rows that failed the swap check and are never trained on).

## Depositing it on Zenodo

If you are reading this because you maintain the repository, here is the whole procedure.

1. **Compress first.** `gzip -k pairs.csv` (wherever your copy lives) turns 620 MB into about 41 MB.
   Zenodo accepts up to 50 GB per record, so this is only about making the download
   civil.
2. **Sign in** at <https://zenodo.org> with GitHub or ORCID. No application is needed.
3. **New upload** → drag in `pairs.csv.gz`.
4. **Upload type:** Dataset.
5. **Title:** something that will still make sense alone, e.g. *Matched molecular pairs
   of acidic headgroup swaps from ChEMBL 37, with potency differences*.
6. **Creators:** you, with your ORCID if you have one.
7. **Description:** what one row is (two molecules, a target, and Δp), how many rows,
   which ChEMBL release, and that pairs are stored in both directions. Link back to the
   GitHub repository.
8. **Licence:** **Creative Commons Attribution-ShareAlike 3.0** — not your choice to make,
   it is inherited from ChEMBL.
9. **Related identifiers:** add the GitHub repository URL as *is supplement to*, and the
   ChEMBL DOI `10.1093/nar/gkad1004` as *is derived from*.
10. **Publish.** You get a DOI immediately, and it is permanent — a published Zenodo
    record cannot be deleted, and new versions get their own DOI under a shared concept
    DOI. Check the file before publishing.
11. **Come back here** and replace the DOI placeholder above, and the one in `README.md`.

A reserved DOI is available before publishing (under "Basic information") if you want the
README to be correct in the same commit that announces the dataset.

## Should the checkpoints go to Zenodo too?

**No — keep them in git, and optionally mirror them.**

The 25 checkpoints are 2.7 MB each, 65 MB in total. They stay in the repository because
the interface cannot run without them: a Hugging Face Space is a git clone, so a model
fetched from Zenodo at startup would mean a download on every cold boot and a Space that
breaks when Zenodo is slow. 65 MB is also comfortably within what git handles well — no
Git LFS needed, and well under GitHub's 100 MB per-file limit.

The one reason to *also* put them on Zenodo is citability: a DOI for the exact weights
behind the published numbers, which a git tag does not give you. If you want that, add
them as a second file in the same record (`models.tar.gz`, about 60 MB) and say in the
description that they are a mirror of the repository's `models/`. That is a nice-to-have,
not a requirement.

What would change this answer is the weights getting much bigger — a larger model, or
checkpoints for many more folds. Past a few hundred megabytes, git stops being the right
home and a download-on-first-run step becomes worth the fragility.
