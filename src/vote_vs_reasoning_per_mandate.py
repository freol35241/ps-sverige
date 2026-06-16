"""Per-mandate vote-vs-reasoning scatter plots.

For each mandate independently:
  vote-stance similarity: 1 - mean |stance_a - stance_b|/2 across the 28
                          unified topics
  reasoning similarity: mean topic-centred cosine between (party, topic)
                       reservation vectors

Output:
  34_vote_vs_reasoning.png        — 2022-26 (re-render with unified IDs)
  58_vote_vs_reasoning_18_22.png  — 2018-22
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

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
INK = "#222222"
BG = "#FAFAFA"
GRID = "#DDD"


def per_mandate(tag: str, vec_file: str, named_pairs: dict, title: str,
                out_name: str, fallback_speech_file: str | None = None):
    stats = pd.read_parquet(IN / f"per_topic_stats_{tag}.parquet")
    pivot = stats.pivot(index="party", columns="topic_id_unified",
                        values="mean_stance").reindex(PARTIES)
    X = pivot.to_numpy()
    n = len(PARTIES)
    vote = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = np.abs(X[i] - X[j])
            vote[i, j] = 1 - np.nanmean(diff) / 2.0

    rv = np.load(IN / vec_file, allow_pickle=True)
    V, N = rv["V"].copy(), rv["N"].copy()
    if fallback_speech_file is not None:
        sp = np.load(IN / fallback_speech_file, allow_pickle=True)
        sp_V, sp_N = sp["V"], sp["N"]
        # Use reservation if available, otherwise speech-based vector
        for i in range(V.shape[0]):
            for t in range(V.shape[1]):
                if N[i, t] == 0 and sp_N[i, t] > 0:
                    V[i, t] = sp_V[i, t]
                    N[i, t] = sp_N[i, t]
    V_c = V.copy()
    for t in range(V.shape[1]):
        present = N[:, t] > 0
        if present.sum() >= 2:
            V_c[present, t] -= V_c[present, t].mean(axis=0, keepdims=True)
    reasoning = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                reasoning[i, j] = 1.0
                continue
            sims = []
            for t in range(V.shape[1]):
                if N[i, t] == 0 or N[j, t] == 0:
                    continue
                vi, vj = V_c[i, t], V_c[j, t]
                ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
                if ni and nj:
                    sims.append(float(vi @ vj / (ni * nj)))
            reasoning[i, j] = np.mean(sims) if sims else np.nan

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"a": PARTIES[i], "b": PARTIES[j],
                          "pair": f"{PARTIES[i]}-{PARTIES[j]}",
                          "vote": vote[i, j], "reasoning": reasoning[i, j]})
    df = pd.DataFrame(pairs).dropna(subset=["vote", "reasoning"])
    if df.empty:
        print(f"  {tag}: no valid pairs")
        return None
    xs, ys = df["vote"].to_numpy(), df["reasoning"].to_numpy()
    a_fit, b_fit = np.polyfit(xs, ys, 1)

    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    xline = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
    ax.plot(xline, a_fit * xline + b_fit, "--", color="#BBB", lw=1.4,
            label="trend (vote predicts reasoning)", zorder=1)

    for _, r in df.iterrows():
        key = (r["a"], r["b"]) if (r["a"], r["b"]) in named_pairs else \
              ((r["b"], r["a"]) if (r["b"], r["a"]) in named_pairs else None)
        if key is not None:
            colour = "#1B9E77" if named_pairs[key][1] == "above" else "#D95F02"
            ax.scatter(r["vote"], r["reasoning"], s=320, alpha=0.95,
                       edgecolor=colour, linewidth=2.6, c="white", zorder=5)
            ax.annotate(r["pair"], (r["vote"], r["reasoning"]),
                        fontsize=12, fontweight="bold", color=colour,
                        xytext=(12, 10), textcoords="offset points", zorder=6)
        else:
            ax.scatter(r["vote"], r["reasoning"], s=120, alpha=0.55,
                       edgecolor="#555", linewidth=0.5, c="#9CB7D4", zorder=3)
            ax.annotate(r["pair"], (r["vote"], r["reasoning"]),
                        fontsize=9, color="#777",
                        xytext=(6, 4), textcoords="offset points", zorder=4)

    ax.text(0.99, 0.04,
            "above trend: closer in reasoning than in votes\n"
            "below trend: closer in votes than in reasoning",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="#444",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#DDD", lw=0.8))
    ax.set_xlabel("Vote-stance similarity", fontsize=11)
    ax.set_ylabel("Reasoning similarity (topic-centred cosine)", fontsize=11)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold",
                 color=INK, pad=12)
    ax.grid(alpha=0.3, color=GRID)
    fig.tight_layout()
    p = FIG / out_name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(p, PUB / out_name)
    print(f"  wrote {out_name}")

    # Print the pairs that are most/least above-trend
    df["residual"] = df["reasoning"] - (a_fit * df["vote"] + b_fit)
    print(f"  {tag} above-trend pairs:")
    print(df.sort_values("residual", ascending=False).head(5)[
        ["pair", "vote", "reasoning", "residual"]].round(3).to_string(index=False))
    print(f"  {tag} below-trend pairs:")
    print(df.sort_values("residual").head(5)[
        ["pair", "vote", "reasoning", "residual"]].round(3).to_string(index=False))
    return df


NAMED_22 = {
    ("S", "C"):  ("vote alike, argue apart", "below"),
    ("M", "SD"): ("vote alike, argue apart", "below"),
    ("V", "MP"): ("vote and reason alike", "above"),
    ("L", "SD"): ("vote alike, argue apart", "below"),
    ("KD", "SD"): ("vote alike, argue apart", "below"),
}
NAMED_18 = {
    ("M", "KD"):  ("Alliance partners", "above"),
    ("M", "SD"):  ("not yet a bloc", "below"),
    ("V", "MP"):  ("red-green pair", "above"),
}


def main() -> None:
    print("== 2022-26 (with speech fallback) ==")
    per_mandate(
        "22_26", "reservation_vectors_22_26_unified.npz", NAMED_22,
        "Vote vs reasoning, 2022-26 mandate (unified topic IDs)\n"
        "Reservation vectors for opposition parties; speech-based fallback "
        "for cabinet cells with no reservation coverage.",
        "34_vote_vs_reasoning.png",
        fallback_speech_file="reasoning_vectors_22_26_unified.npz")
    print("\n== 2018-22 ==")
    per_mandate(
        "18_22", "reservation_vectors_18_22.npz", NAMED_18,
        "Vote vs reasoning, 2018-22 mandate (unified topic IDs)\n"
        "Reservation-only metric. In this mandate every party has ≥26 "
        "reservation cells covered, so no speech fallback is needed.",
        "58_vote_vs_reasoning_18_22.png")


if __name__ == "__main__":
    main()
