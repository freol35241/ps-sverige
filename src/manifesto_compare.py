"""Comparative analysis across the 2022 manifestos.

Three outputs:
  1. Value frequency heatmap — which parties invoke which values, normalised
     per 1000 words.
  2. Element-text similarity matrices — using SBERT embeddings of all
     diagnosis / end_state / mechanism sentences per party. Are parties more
     similar in what they identify as problems, what they want, or how they
     propose to act?
  3. Stated vs revealed gap — for each party, compare manifesto reasoning
     vector against their Riksdag reservation reasoning vector.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# Display order: left-to-right ideological-ish.
PARTIES = ["v", "s", "mp", "c", "l", "kd", "m", "sd"]
PARTY_UPPER = {p: p.upper() for p in PARTIES}
PARTY_COLOR = {
    "v": "#AF0000", "s": "#E8112D", "mp": "#83CF39", "c": "#009933",
    "l": "#006AB3", "kd": "#211F70", "m": "#52BDEC", "sd": "#DDDD00",
}

BG = "#FAFAFA"
INK = "#222222"


# ----------------------------------------------------------------------------
# Value frequency analysis
# ----------------------------------------------------------------------------

def value_heatmap() -> None:
    profile = pd.read_parquet(IN / "manifesto_party_profile.parquet")
    profile = profile.set_index("party").loc[PARTIES]

    value_cols = [c for c in profile.columns if c.endswith("_per_kw")]
    value_names = [c[2:].replace("_per_kw", "") for c in value_cols]

    matrix = profile[value_cols].to_numpy()

    # Z-score per value (column) so each value is shown as deviation from
    # the cross-party mean for that value. Highlights distinctiveness.
    col_mean = matrix.mean(axis=0)
    col_std = matrix.std(axis=0)
    col_std[col_std == 0] = 1
    z = (matrix - col_mean) / col_std

    # Sort values by max absolute deviation across parties so the most
    # distinctive ones appear first.
    distinct = np.abs(z).max(axis=0)
    order = np.argsort(distinct)[::-1]
    z_sorted = z[:, order]
    names_sorted = [value_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    im = ax.imshow(z_sorted, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(PARTIES)))
    ax.set_yticklabels([p.upper() for p in PARTIES], fontsize=11, fontweight="bold")

    # Cell values
    for i in range(z_sorted.shape[0]):
        for j in range(z_sorted.shape[1]):
            v = z_sorted[i, j]
            color = "white" if abs(v) > 1.5 else "#333"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Which values each party invokes most\n"
                 "Z-scored frequency per 1000 words of manifesto. "
                 "Red = invokes this value more than average; blue = less.",
                 loc="left", fontsize=12, fontweight="bold",
                 color=INK, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                        label="standard deviations from cross-party mean")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    out = FIG / "web" / "22_manifesto_values.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    site_out = ROOT / "site" / "public" / "figures" / out.name
    import shutil
    shutil.copyfile(out, site_out)
    print(f"  wrote {out.name}")


# ----------------------------------------------------------------------------
# Element similarity via SBERT
# ----------------------------------------------------------------------------

def element_similarity() -> None:
    """For each element type, embed all sentences with that label per party
    and compute pairwise cosine similarity. Reveals which parties share
    diagnoses vs which share aspirations vs which share mechanisms."""
    from sentence_transformers import SentenceTransformer

    print("  loading SBERT ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

    ann = pd.read_parquet(IN / "manifesto_sentences.parquet")

    # Build per-(party, element) mean vector.
    element_vectors = {}  # (party, element) -> 768-dim vector
    n_cells = {}
    for (party, element), g in ann.groupby(["party", "element"]):
        if element == "other":
            continue
        sentences = g["sentence"].tolist()
        if len(sentences) < 3:
            continue
        embs = model.encode(sentences, batch_size=16, show_progress_bar=False,
                             normalize_embeddings=True)
        element_vectors[(party, element)] = embs.mean(axis=0)
        n_cells[(party, element)] = len(sentences)
    print(f"  built {len(element_vectors)} (party, element) vectors")

    # Three similarity matrices, one per element.
    for element in ["diagnosis", "end_state", "mechanism"]:
        present = [p for p in PARTIES if (p, element) in element_vectors]
        n = len(present)
        sim = np.zeros((n, n))
        for i, a in enumerate(present):
            va = element_vectors[(a, element)]
            for j, b in enumerate(present):
                vb = element_vectors[(b, element)]
                sim[i, j] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

        fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        im = ax.imshow(sim, cmap="RdBu_r", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(n)); ax.set_xticklabels([p.upper() for p in present])
        ax.set_yticks(range(n)); ax.set_yticklabels([p.upper() for p in present])
        for i in range(n):
            for j in range(n):
                color = "white" if sim[i, j] > 0.85 else "#333"
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, color=color)
        title_map = {
            "diagnosis": "Diagnosis: which parties identify similar problems?",
            "end_state": "End state: which parties want similar futures?",
            "mechanism": "Mechanism: which parties propose similar means?",
        }
        ax.set_title(title_map[element],
                     loc="left", fontsize=12, fontweight="bold", pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      label="cosine similarity (manifesto sentence embeddings)")
        fig.tight_layout()

        out = FIG / "web" / f"23_manifesto_sim_{element}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        import shutil
        shutil.copyfile(out, ROOT / "site" / "public" / "figures" / out.name)
        print(f"  wrote {out.name}")


def main() -> None:
    print("== value frequency heatmap ==")
    value_heatmap()
    print("\n== element similarity matrices ==")
    element_similarity()


if __name__ == "__main__":
    main()
