"""The what-vs-why gap analysis.

For each pair of parties, compute two similarity matrices:
  - position similarity = correlation of their stance scores across topics
  - reasoning similarity = mean cosine sim of their per-topic reasoning vectors

If the two matrices look the same: parties that vote together also justify
together. If they diverge: rhetoric and voting come apart — the more
interesting case.

Also: per-topic gap. For each topic, find pairs where position similarity is
high but reasoning sim is low (or vice versa).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import PARTY_COLOR

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def main() -> None:
    grid = pd.read_parquet(IN / "party_topic.parquet")
    meta = pd.read_parquet(IN / "topic_meta.parquet")
    rv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    V = rv["V"]  # (n_parties, n_topics, 768)
    N = rv["N"]  # (n_parties, n_topics)

    pivot = grid.pivot(index="parti", columns="topic_id", values="score").loc[PARTIES]

    # --- WHAT: 1 − mean absolute stance difference (in [0, 1], higher = closer).
    # Pearson on raw scores is misleading because the cabinet has near-zero
    # variance across topics (always Ja with their own proposals).
    X = pivot.to_numpy()  # (n_parties, n_topics) in [-1, +1]
    what = np.zeros((len(PARTIES), len(PARTIES)), dtype=np.float64)
    for i in range(len(PARTIES)):
        for j in range(len(PARTIES)):
            mad = np.nanmean(np.abs(X[i] - X[j]))
            what[i, j] = 1 - mad / 2.0  # mad in [0,2] → similarity in [0,1]
    what_df = pd.DataFrame(what, index=PARTIES, columns=PARTIES)

    # --- WHY: party-party reasoning cosine, with embeddings centered per topic
    # so the chamber-Swedish baseline doesn't dominate.
    V_c = V.copy()
    for t in range(V.shape[1]):
        present = N[:, t] > 0
        if present.sum() >= 2:
            V_c[present, t] -= V_c[present, t].mean(axis=0, keepdims=True)

    why = np.zeros((len(PARTIES), len(PARTIES)), dtype=np.float64)
    for i in range(len(PARTIES)):
        for j in range(len(PARTIES)):
            if i == j:
                why[i, j] = 1.0
                continue
            sims = []
            for t in range(V.shape[1]):
                if N[i, t] == 0 or N[j, t] == 0:
                    continue
                vi, vj = V_c[i, t], V_c[j, t]
                ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
                if ni == 0 or nj == 0:
                    continue
                sims.append(float(vi @ vj / (ni * nj)))
            why[i, j] = np.mean(sims) if sims else np.nan
    why_df = pd.DataFrame(why, index=PARTIES, columns=PARTIES)

    print("WHAT (vote correlation):")
    print(what_df.round(2).to_string())
    print("\nWHY (mean reasoning cosine):")
    print(why_df.round(3).to_string())

    # The gap: rank order by WHAT vs WHY differs?
    # Take off-diagonal pairs, compute (WHAT − WHY normalised).
    pairs = []
    for i in range(len(PARTIES)):
        for j in range(i + 1, len(PARTIES)):
            pairs.append((PARTIES[i], PARTIES[j],
                          what[i, j], why[i, j]))
    pairs_df = pd.DataFrame(pairs, columns=["a", "b", "what_sim", "why_sim"])
    pairs_df["what_rank"] = pairs_df["what_sim"].rank()
    pairs_df["why_rank"] = pairs_df["why_sim"].rank()
    pairs_df["gap_rank"] = pairs_df["what_rank"] - pairs_df["why_rank"]
    print("\nPairs with biggest WHAT-vs-WHY rank gap (positive = much closer in votes than in rhetoric):")
    print(pairs_df.sort_values("gap_rank", ascending=False).head(8).to_string(index=False))
    print("\nPairs where rhetoric is closer than votes (negative gap):")
    print(pairs_df.sort_values("gap_rank").head(5).to_string(index=False))

    # === Figure A: side-by-side matrices ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    for ax, mat, title, vmin, vmax in [
        (axes[0], what, "WHAT — 1 − mean stance distance", 0.4, 1.0),
        (axes[1], why, "WHY — debate-speech cosine (per-topic centered)", -0.4, 1.0),
    ]:
        ax.set_facecolor("#FAFAFA")
        im = ax.imshow(mat, cmap="RdBu_r" if vmin < 0 else "viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(PARTIES))); ax.set_xticklabels(PARTIES)
        ax.set_yticks(range(len(PARTIES))); ax.set_yticklabels(PARTIES)
        for i in range(len(PARTIES)):
            for j in range(len(PARTIES)):
                v = mat[i, j]
                c = "white" if (vmin < 0 and abs(v) > 0.55) or (vmin >= 0 and v > (vmin + vmax) / 2 + 0.03) else "#333"
                if vmin < 0:
                    txt = f"{v:+.2f}".replace("+", "")
                else:
                    txt = f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=c)
        ax.set_title(title, loc="left", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("What parties do vs. what they say — two views of the same parliament",
                 fontsize=13, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "14_what_vs_why.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # === Figure B: pair scatter ===
    fig, ax = plt.subplots(figsize=(11, 8), dpi=140)
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")

    # Trend line: same-rank diagonal in normalised coordinates.
    xs = pairs_df["what_sim"].to_numpy()
    ys = pairs_df["why_sim"].to_numpy()
    # Fit a simple line.
    a, b = np.polyfit(xs, ys, 1)
    xline = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
    ax.plot(xline, a * xline + b, "--", color="#888", lw=1, label="trend")

    for _, r in pairs_df.iterrows():
        ax.scatter(r["what_sim"], r["why_sim"], s=120, alpha=0.75,
                   edgecolor="#333", linewidth=0.6, c="#3A6EA5")
        # Highlight pairs with the biggest deviation
        residual = r["why_sim"] - (a * r["what_sim"] + b)
        c = "#1B9E77" if residual > 0 else "#D95F02"
        weight = "bold" if abs(residual) > 0.05 else "normal"
        ax.annotate(f"{r['a']}–{r['b']}",
                    (r["what_sim"], r["why_sim"]),
                    fontsize=10, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points",
                    fontweight=weight, color=c)

    ax.set_xlabel("WHAT — vote-stance similarity (1 − mean stance distance)",
                  fontsize=10)
    ax.set_ylabel("WHY — reasoning cosine (per-topic centered)",
                  fontsize=10)
    ax.set_title("What parties do vs how they argue\n"
                 "Above the line (green): rhetoric closer than votes — "
                 "shared values, different policies.\n"
                 "Below the line (orange): votes closer than rhetoric — "
                 "marriage of convenience.",
                 fontsize=11, loc="left", pad=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "15_pair_what_vs_why.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
