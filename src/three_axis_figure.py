"""WHAT × WHY × HOW gap figure — the three-axis compass headline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def what_sim_matrix() -> pd.DataFrame:
    grid = pd.read_parquet(IN / "party_topic.parquet")
    pivot = grid.pivot(index="parti", columns="topic_id",
                       values="score").loc[PARTIES]
    X = pivot.to_numpy()
    M = np.zeros((len(PARTIES), len(PARTIES)))
    for i in range(len(PARTIES)):
        for j in range(len(PARTIES)):
            M[i, j] = 1 - np.nanmean(np.abs(X[i] - X[j])) / 2.0
    return pd.DataFrame(M, index=PARTIES, columns=PARTIES)


def why_sim_matrix() -> pd.DataFrame:
    rv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    V, N = rv["V"], rv["N"]
    V_c = V.copy()
    for t in range(V.shape[1]):
        present = N[:, t] > 0
        if present.sum() >= 2:
            V_c[present, t] -= V_c[present, t].mean(axis=0, keepdims=True)
    M = np.zeros((len(PARTIES), len(PARTIES)))
    for i in range(len(PARTIES)):
        for j in range(len(PARTIES)):
            if i == j:
                M[i, j] = 1.0
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
            M[i, j] = np.mean(sims) if sims else np.nan
    return pd.DataFrame(M, index=PARTIES, columns=PARTIES)


def how_pair_dict() -> dict[tuple[str, str], float]:
    pair = pd.read_parquet(IN / "how_pair_sim.parquet")
    out = {}
    for _, r in pair.iterrows():
        a, b = r["a"], r["b"]
        out[(a, b)] = float(r["how_sim"])
        out[(b, a)] = float(r["how_sim"])
    return out


def main() -> None:
    what = what_sim_matrix()
    why = why_sim_matrix()
    how_pair = how_pair_dict()

    # Pair-level table for the figure
    rows = []
    for i, a in enumerate(PARTIES):
        for j in range(i + 1, len(PARTIES)):
            b = PARTIES[j]
            rows.append({
                "a": a, "b": b, "pair": f"{a}–{b}",
                "what_sim": float(what.loc[a, b]),
                "why_sim": float(why.loc[a, b]),
                "how_sim": float(how_pair.get((a, b), np.nan)),
            })
    pairs = pd.DataFrame(rows)
    # Z-rank each axis to put on comparable scale.
    for col in ["what_sim", "why_sim", "how_sim"]:
        z = pairs[col]
        pairs[col + "_n"] = (z - z.min()) / (z.max() - z.min())
    # Mean across the three: an integrated "are these parties really aligned" score.
    pairs["mean_n"] = pairs[["what_sim_n", "why_sim_n", "how_sim_n"]].mean(axis=1)
    pairs["spread_n"] = pairs[["what_sim_n", "why_sim_n", "how_sim_n"]].std(axis=1)
    pairs = pairs.sort_values("mean_n", ascending=False).reset_index(drop=True)

    print("\nPairs ranked by mean alignment across all 3 axes:")
    print(pairs[["pair", "what_sim", "why_sim", "how_sim",
                 "mean_n", "spread_n"]].round(3).to_string(index=False))

    # === Figure: parallel-coords + a side bar of "consistency" ===
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 8.5), dpi=140,
                                     gridspec_kw={"width_ratios": [4, 1]})
    fig.patch.set_facecolor("#FAFAFA")
    axL.set_facecolor("#FAFAFA"); axR.set_facecolor("#FAFAFA")

    # Highlight: tag pairs as
    #   "all-three aligned" (top quartile mean_n, low spread)
    #   "one-axis only"     (high spread)
    #   "all-three apart"   (bottom quartile mean_n)
    def label(row):
        if row["mean_n"] >= 0.66 and row["spread_n"] <= 0.20:
            return "aligned"
        if row["spread_n"] >= 0.30:
            return "fragmented"
        if row["mean_n"] <= 0.33 and row["spread_n"] <= 0.20:
            return "apart"
        return "middle"
    pairs["category"] = pairs.apply(label, axis=1)

    cat_color = {"aligned": "#1B9E77",
                 "fragmented": "#D95F02",
                 "apart": "#7570B3",
                 "middle": "#999999"}
    cat_alpha = {"aligned": 1.0, "fragmented": 1.0, "apart": 0.85, "middle": 0.4}
    cat_lw = {"aligned": 2.0, "fragmented": 2.0, "apart": 1.6, "middle": 1.0}

    # Parallel coords: x ∈ {WHAT, WHY, HOW} mapped to 0,1,2; y = normalised similarity.
    x = np.array([0, 1, 2])
    axL.set_xticks(x)
    axL.set_xticklabels(["WHAT\nvote stance", "WHY\nspeech reasoning",
                          "HOW\nco-authorship + lexicon"], fontsize=11)
    axL.set_ylabel("Pair similarity (rank-normalised)", fontsize=10)
    axL.set_ylim(-0.04, 1.06)
    axL.grid(axis="y", alpha=0.3)

    # Plot each pair as a line connecting its three normalised values.
    for _, row in pairs.iterrows():
        y = [row["what_sim_n"], row["why_sim_n"], row["how_sim_n"]]
        cat = row["category"]
        axL.plot(x, y, "-o", color=cat_color[cat], alpha=cat_alpha[cat],
                 lw=cat_lw[cat], markersize=5, zorder=3 if cat != "middle" else 1)
        if cat in ("aligned", "fragmented", "apart"):
            axL.annotate(row["pair"], (x[2], y[2]),
                         xytext=(6, 0), textcoords="offset points",
                         fontsize=9, va="center",
                         color=cat_color[cat], fontweight="bold")
        else:
            # Faint labels on the right for middle pairs that stand out
            pass

    axL.set_title("What parties do, what they say, and how they propose to do it\n"
                  "One line per party pair, across the three axes "
                  "(rank-normalised for comparability)",
                  fontsize=12, loc="left", pad=10)

    # Legend
    handles = [plt.Line2D([0], [0], color=c, lw=2, marker="o", label=lbl)
               for lbl, c in [("aligned on all three", cat_color["aligned"]),
                              ("fragmented (high axis-disagreement)", cat_color["fragmented"]),
                              ("apart on all three", cat_color["apart"]),
                              ("middle", cat_color["middle"])]]
    axL.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    # Right panel: top fragmented pairs ranked by spread
    frag = pairs.sort_values("spread_n", ascending=False).head(10)
    y_pos = np.arange(len(frag))
    axR.barh(y_pos, frag["spread_n"], color=[cat_color.get(c, "#888") for c in frag["category"]])
    for yi, row in zip(y_pos, frag.itertuples()):
        # Note which axis they're highest on
        scores = {"WHAT": row.what_sim_n, "WHY": row.why_sim_n, "HOW": row.how_sim_n}
        peak = max(scores, key=scores.get)
        valley = min(scores, key=scores.get)
        axR.text(row.spread_n + 0.005, yi,
                 f"{peak}↑ {valley}↓",
                 va="center", fontsize=8, color="#444")
    axR.set_yticks(y_pos)
    axR.set_yticklabels(frag["pair"], fontsize=9)
    axR.invert_yaxis()
    axR.set_xlabel("Spread across axes")
    axR.set_title("Most fragmented pairs\n"
                  "Where the three axes disagree the most",
                  fontsize=10, loc="left", pad=8)
    axR.set_xlim(0, frag["spread_n"].max() * 1.35)
    axR.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = FIG / "16_three_axis.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Save the table for the diary
    pairs.to_parquet(IN / "three_axis_pairs.parquet", index=False)


if __name__ == "__main__":
    main()
