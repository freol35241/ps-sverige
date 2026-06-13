"""Generate headline figures from the MP embedding."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# Official party hex codes (close approximations).
PARTY_COLOR = {
    "S": "#E8112D",     # Socialdemokraterna
    "M": "#52BDEC",     # Moderaterna
    "SD": "#DDDD00",    # Sverigedemokraterna
    "V": "#AF0000",     # Vänsterpartiet
    "C": "#009933",     # Centerpartiet
    "KD": "#211F70",    # Kristdemokraterna
    "MP": "#83CF39",    # Miljöpartiet
    "L": "#006AB3",     # Liberalerna
    "-": "#888888",     # politisk vilde / independent
}

# Display order in the legend (roughly left-bloc → right-bloc).
PARTY_ORDER = ["V", "S", "MP", "C", "L", "KD", "M", "SD", "-"]

# Tidö government bloc (M+KD+L cabinet, SD support party).
TIDOE = {"M", "KD", "L", "SD"}


def load():
    pca = np.load(IN / "pca.npz", allow_pickle=True)
    var = pca["var_ratio"]
    emb = pd.read_parquet(IN / "mp_embedding.parquet")
    return emb, var


def behavioural_map(emb: pd.DataFrame, var: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(12, 9), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    fit = emb[emb["in_fit"]].copy()

    # Shade the two blocs as faint background regions using party centroids.
    for party in PARTY_ORDER:
        pts = fit[fit["party_modal"] == party][["PC1", "PC2"]].to_numpy()
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1],
                    color=PARTY_COLOR[party], alpha=0.07,
                    edgecolor=PARTY_COLOR[party], linewidth=0.5)
        except Exception:
            pass

    # MP scatter, colored by modal party.
    for party in PARTY_ORDER:
        sub = fit[fit["party_modal"] == party]
        if sub.empty:
            continue
        ax.scatter(sub["PC1"], sub["PC2"],
                   s=40,
                   c=PARTY_COLOR[party],
                   edgecolor="white", linewidth=0.6,
                   alpha=0.95,
                   label=f"{party} (n={len(sub)})", zorder=3)

    # Party centroid labels.
    cents = fit.groupby("party_modal")[["PC1", "PC2"]].mean()
    for party, row in cents.iterrows():
        ax.annotate(party, (row["PC1"], row["PC2"]),
                    fontsize=14, fontweight="bold",
                    ha="center", va="center",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.35",
                              fc=PARTY_COLOR.get(party, "#444"),
                              ec="white", lw=1.5),
                    zorder=5)

    ax.axhline(0, color="#BBB", lw=0.5, zorder=1)
    ax.axvline(0, color="#BBB", lw=0.5, zorder=1)
    ax.set_xlabel(f"PC1 — government / opposition  ({var[0]:.1%} var)", fontsize=11)
    ax.set_ylabel(f"PC2 — within-bloc structure  ({var[1]:.1%} var)", fontsize=11)
    ax.set_title("Behavioural map of the Riksdag (2022–2026 mandate)\n"
                 "Each dot is an MP, positioned by how they voted on ~2,500 chamber votes",
                 fontsize=13, loc="left", pad=12)

    # Subtle bloc annotations.
    ax.annotate("Tidö bloc →", xy=(0.97, 0.02), xycoords="axes fraction",
                ha="right", fontsize=10, color="#666", style="italic")
    ax.annotate("← opposition", xy=(0.03, 0.02), xycoords="axes fraction",
                ha="left", fontsize=10, color="#666", style="italic")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2,
              title="Modal party (n MPs in fit)")

    fig.tight_layout()
    out = FIG / "01_behavioural_map.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def variance_scree(var: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    x = np.arange(1, len(var) + 1)
    ax.bar(x, var * 100, color="#2A6F97", edgecolor="white")
    for xi, vi in zip(x, var):
        ax.text(xi, vi * 100 + 0.5, f"{vi:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree — how much vote variance each axis captures", loc="left")
    ax.set_ylim(0, max(var) * 110)
    fig.tight_layout()
    out = FIG / "02_scree.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    emb, var = load()
    p1 = behavioural_map(emb, var)
    p2 = variance_scree(var)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
