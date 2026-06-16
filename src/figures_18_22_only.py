"""Render 2018-22-only versions of figures that were previously
side-by-side comparisons, so the /2018-2022 page can be self-contained.

Output:
  60_chamber_map_18_22.png  — 2018-22 PCA, party centroids, members
  61_defectors_18_22.png    — top 12 2018-22 behavioural defectors
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "web"
PUB = ROOT / "site" / "public" / "figures"

PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
PARTY_ORDER = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
INK = "#222222"
BG = "#FAFAFA"


def fig_chamber_map():
    emb = pd.read_parquet(IN / "mp_embedding_18_22.parquet")
    fit = emb[emb["in_fit"]]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for party in PARTY_ORDER:
        sub = fit[fit["party_modal"] == party]
        if sub.empty:
            continue
        ax.scatter(sub["PC1"], sub["PC2"], s=22,
                   color=PARTY_COLOR[party], alpha=0.65,
                   edgecolor="white", linewidth=0.5)
        cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
        ax.scatter([cx], [cy], s=520, color=PARTY_COLOR[party],
                   edgecolor="white", linewidth=2, zorder=10)
        ax.text(cx, cy, party, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=11)
    ax.axhline(0, color="#aaa", lw=0.5, alpha=0.5)
    ax.axvline(0, color="#aaa", lw=0.5, alpha=0.5)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("The 2018-22 chamber map\n"
                 "PCA on the 444-by-3,203 vote matrix. "
                 "Each dot is a Member of Parliament; circles mark party centroids.",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "60_chamber_map_18_22.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, PUB / out.name)
    print(f"  wrote {out.name}")


def fig_defectors():
    df = pd.read_parquet(IN / "defectors_18_22.parquet")
    df = df[df["party_modal"].isin(PARTY_ORDER)].head(12)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    head = df.iloc[::-1]
    y = np.arange(len(head))
    ax.barh(y, head["ratio_own_over_near"],
            color=[PARTY_COLOR.get(p, "#888") for p in head["party_modal"]],
            edgecolor="white")
    labels = [f"{n[:32]:<32}  ({p}->{q})"
              for n, p, q in zip(head["namn"], head["party_modal"],
                                 head["nearest_other"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontfamily="monospace")
    ax.set_xscale("log")
    ax.axvline(1.0, color="#444", lw=0.8, linestyle="--")
    ax.set_xlabel("d(own party) / d(nearest other party), log scale",
                  fontsize=11)
    ax.set_title("Top 12 behavioural defectors, 2018-22 mandate\n"
                 "Each row: MP, own party (modal), nearest other party. "
                 "Ratio > 1 means the MP voted closer to another party than to their own.",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    for yi, val in zip(y, head["ratio_own_over_near"].to_numpy()):
        ax.text(val * 1.07, yi, f"{val:.1f}x", va="center",
                fontsize=8.5, color="#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "61_defectors_18_22.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, PUB / out.name)
    print(f"  wrote {out.name}")


def main() -> None:
    print("== rendering 2018-22-only figures ==")
    fig_chamber_map()
    fig_defectors()


if __name__ == "__main__":
    main()
