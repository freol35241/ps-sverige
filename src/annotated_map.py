"""Single composite figure: behavioural map + annotated defectors.

Picks the top behavioural defectors and draws an arrow from each MP to the
centroid of their nearest other party. The narrative figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from src.figures import PARTY_COLOR, PARTY_ORDER

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# MPs to call out explicitly. Chosen for narrative weight, not just top ratio:
#   - Avci: Liberal party leader who sits on KD centroid
#   - Widding & El-Haj: politisk vilde whose votes track their former party
#   - Cederfelt: only M MP behaviourally inside L
HIGHLIGHT = {"Gulan Avci", "Elsa Widding", "Jamal El-Haj", "Margareta Cederfelt"}


def main() -> None:
    emb = pd.read_parquet(IN / "mp_embedding.parquet")
    defectors = pd.read_parquet(IN / "defectors.parquet")
    fit = emb[emb["in_fit"]].copy()

    centroids = fit.groupby("party_modal")[["PC1", "PC2"]].mean()

    fig, ax = plt.subplots(figsize=(13, 9), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for party in PARTY_ORDER:
        pts = fit[fit["party_modal"] == party][["PC1", "PC2"]].to_numpy()
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1],
                    color=PARTY_COLOR[party], alpha=0.07,
                    edgecolor=PARTY_COLOR[party], linewidth=0.4)
        except Exception:
            pass

    for party in PARTY_ORDER:
        sub = fit[fit["party_modal"] == party]
        if sub.empty:
            continue
        ax.scatter(sub["PC1"], sub["PC2"], s=28,
                   c=PARTY_COLOR[party],
                   edgecolor="white", linewidth=0.5,
                   alpha=0.85, zorder=3)

    # Party centroid badges.
    for party, row in centroids.iterrows():
        ax.annotate(party, (row["PC1"], row["PC2"]),
                    fontsize=13, fontweight="bold",
                    ha="center", va="center", color="white",
                    bbox=dict(boxstyle="circle,pad=0.32",
                              fc=PARTY_COLOR.get(party, "#444"),
                              ec="white", lw=1.5),
                    zorder=5)

    # Highlighted defectors: arrow from MP position to nearest other party.
    # Label placements set manually to avoid overlap in the dense Tidö cluster.
    LABEL_POS = {
        "Gulan Avci":         (33, -6),
        "Margareta Cederfelt":(33, -10),
        "Elsa Widding":       (33, -14),
        "Jamal El-Haj":       (-35, -2),
    }
    highlighted = defectors[defectors["namn"].isin(HIGHLIGHT)]
    for _, m in highlighted.iterrows():
        x, y = m["PC1"], m["PC2"]
        tx, ty = centroids.loc[m["nearest_other"], ["PC1", "PC2"]]
        ax.annotate("",
                    xy=(tx, ty), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#222",
                                    lw=1.2, alpha=0.7,
                                    connectionstyle="arc3,rad=0.15"),
                    zorder=4)
        ax.scatter([x], [y], s=120, marker="o",
                   facecolor=PARTY_COLOR.get(m["party_modal"], "#888"),
                   edgecolor="#222", linewidth=1.4, zorder=6)
        label = f"{m['namn']}\n({m['party_modal']} → behaves like {m['nearest_other']})"
        lx, ly = LABEL_POS[m["namn"]]
        ha = "left" if lx > x else "right"
        ax.annotate(label, (x, y), xytext=(lx, ly),
                    fontsize=9, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.5,
                                    connectionstyle="arc3,rad=0.1"),
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="#222", lw=0.6, alpha=0.95),
                    zorder=7)

    ax.axhline(0, color="#BBB", lw=0.5, zorder=1)
    ax.axvline(0, color="#BBB", lw=0.5, zorder=1)
    ax.set_xlabel("PC1 — government / opposition", fontsize=11)
    ax.set_ylabel("PC2 — within-bloc structure", fontsize=11)
    ax.set_title("Where the boundary-walkers sit\n"
                 "MPs whose votes look more like another party than their own",
                 fontsize=13, loc="left", pad=12)

    ax.annotate("Tidö bloc →", xy=(0.97, 0.02), xycoords="axes fraction",
                ha="right", fontsize=10, color="#666", style="italic")
    ax.annotate("← opposition", xy=(0.03, 0.02), xycoords="axes fraction",
                ha="left", fontsize=10, color="#666", style="italic")

    fig.tight_layout()
    out = FIG / "04_annotated_map.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
