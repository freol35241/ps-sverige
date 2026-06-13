"""Defector index: how behaviourally displaced is each MP from their own party?

In a tightly disciplined parliament most MPs sit on top of their party centroid.
For each MP we compute (in the full PCA space used for the fit):

  d_own   = distance from MP to centroid of own party
  d_near  = distance from MP to centroid of nearest *other* party
  z_own   = d_own measured in units of the party's within-party scatter
  ratio   = d_own / d_near
            < 1 → closer to own party (normal)
            ~ 1 → equidistant
            > 1 → behaviourally inside another party

We also identify, per MP, which other party they sit closest to.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import PARTY_COLOR

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PC_COLS = ["PC1", "PC2", "PC3", "PC4"]


def main() -> None:
    emb = pd.read_parquet(IN / "mp_embedding.parquet")
    fit = emb[emb["in_fit"]].copy()

    # Party centroids in the full 4D space.
    centroids = fit.groupby("party_modal")[PC_COLS].mean()
    # Within-party scatter (used to z-score d_own).
    scatter = (fit.groupby("party_modal")[PC_COLS]
                  .apply(lambda g: np.linalg.norm(g.to_numpy() - g.to_numpy().mean(axis=0), axis=1).mean()))

    def per_mp(row):
        x = row[PC_COLS].to_numpy(dtype=float)
        own = row["party_modal"]
        d_own = float(np.linalg.norm(x - centroids.loc[own].to_numpy()))
        # Distance to every other party.
        others = centroids.drop(index=own)
        d_each = np.linalg.norm(others.to_numpy() - x, axis=1)
        i_near = int(np.argmin(d_each))
        d_near = float(d_each[i_near])
        nearest_other = others.index[i_near]
        return pd.Series({
            "d_own": d_own,
            "d_near_other": d_near,
            "nearest_other": nearest_other,
            "ratio_own_over_near": d_own / d_near if d_near else np.nan,
            "z_own": d_own / scatter[own] if scatter[own] else np.nan,
        })

    extra = fit.apply(per_mp, axis=1)
    out = pd.concat([fit.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    out = out.sort_values("ratio_own_over_near", ascending=False)

    out.to_parquet(OUT / "defectors.parquet", index=False)

    print("=== Top behavioural defectors (highest d_own / d_near_other) ===\n")
    cols = ["namn", "party_modal", "party_latest", "nearest_other",
            "ratio_own_over_near", "z_own", "n_present_votes"]
    top = out.head(15)[cols]
    print(top.to_string(index=False))

    print("\n=== Who sits closest to whom (off-diagonal counts) ===")
    crosstab = pd.crosstab(out["party_modal"], out["nearest_other"])
    print(crosstab.to_string())

    # Plot the headline list.
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    head = out.head(12).iloc[::-1]
    y = np.arange(len(head))
    ax.barh(y, head["ratio_own_over_near"],
            color=[PARTY_COLOR.get(p, "#888") for p in head["party_modal"]],
            edgecolor="white")
    labels = [f"{n}  ({p}→{q})"
              for n, p, q in zip(head["namn"], head["party_modal"], head["nearest_other"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#444", lw=0.8, linestyle="--")
    ax.set_xlabel("d(own party) / d(nearest other party) — log scale")
    ax.set_title("Top behavioural defectors — closer to another party's centroid\n"
                 "than their own (current mandate, full PCA space)",
                 fontsize=12, loc="left")
    ax.text(1.03, len(head) - 0.5, "  equidistant", fontsize=9, color="#444")
    # Annotate the ratio value on each bar.
    for yi, val in zip(y, head["ratio_own_over_near"].to_numpy()):
        ax.text(val * 1.07, yi, f"{val:.1f}×", va="center", fontsize=8, color="#333")
    fig.tight_layout()
    p = FIG / "03_defectors.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
