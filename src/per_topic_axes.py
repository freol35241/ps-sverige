"""Per-topic WHAT/WHY/HOW analysis.

The aggregated three-axis analysis answered "do parties align across all
topics on average?" This module answers the more granular question:
"on which topics does the alignment break down — and along which axis?"

For each (topic, party-pair) we compute the three similarities. Then we ask:
- Which topics fragment the most (high spread across axes within the topic)?
- Which topics give the cleanest, most informative pair distinctions?
- Where on each topic does a pair stand on each axis?

The downstream question: which topics should appear in a voter-facing tool?
Topics where party positions are most differentiated across axes carry the
most information per voter question.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def what_per_topic() -> dict[int, np.ndarray]:
    """topic_id → (n_parties, n_parties) stance-distance similarity matrix."""
    grid = pd.read_parquet(IN / "party_topic.parquet")
    pivot = grid.pivot(index="parti", columns="topic_id",
                       values="score").loc[PARTIES]
    out = {}
    for t in pivot.columns:
        col = pivot[t].to_numpy()
        M = np.zeros((len(PARTIES), len(PARTIES)))
        for i in range(len(PARTIES)):
            for j in range(len(PARTIES)):
                if np.isnan(col[i]) or np.isnan(col[j]):
                    M[i, j] = np.nan
                else:
                    M[i, j] = 1 - abs(col[i] - col[j]) / 2.0
        out[int(t)] = M
    return out


def why_per_topic() -> dict[int, np.ndarray]:
    rv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    V, N = rv["V"], rv["N"]
    out = {}
    for t in range(V.shape[1]):
        present = N[:, t] > 0
        if present.sum() < 2:
            out[t] = np.full((len(PARTIES), len(PARTIES)), np.nan)
            continue
        # Centre per topic.
        Vt = V[:, t, :].copy()
        Vt[present] -= Vt[present].mean(axis=0, keepdims=True)
        M = np.full((len(PARTIES), len(PARTIES)), np.nan)
        for i in range(len(PARTIES)):
            for j in range(len(PARTIES)):
                if N[i, t] == 0 or N[j, t] == 0:
                    continue
                if i == j:
                    M[i, j] = 1.0
                    continue
                vi, vj = Vt[i], Vt[j]
                ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
                if ni == 0 or nj == 0:
                    continue
                M[i, j] = float(vi @ vj / (ni * nj))
        out[t] = M
    return out


def how_per_topic() -> dict[int, np.ndarray]:
    """Per-topic HOW = co-authorship similarity (within topic) + lexicon distance.
    Co-authorship: pairs of parties signing the same reservations on votes in this topic.
    Lexicon: per-(party, topic) score differences.
    """
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates("votering_id"))
    topics = pd.read_parquet(IN / "topics.parquet")
    df = topics.merge(tex[["votering_id", "reservation_partier"]],
                       on="votering_id")

    PARTY_SET = set(PARTIES)
    coauth = {t: (Counter(), Counter()) for t in range(28)}
    for _, r in df.iterrows():
        s = r["reservation_partier"]
        if not isinstance(s, str) or not s:
            continue
        for grp in s.split(";"):
            party_set = sorted(p.strip() for p in grp.split(",")
                               if p.strip() in PARTY_SET)
            for p in party_set:
                coauth[int(r["topic_id"])][0][p] += 1
            for a, b in combinations(party_set, 2):
                coauth[int(r["topic_id"])][1][(a, b)] += 1
                coauth[int(r["topic_id"])][1][(b, a)] += 1

    lex_cell = pd.read_parquet(IN / "how_lexicon.parquet")
    dims = ["market_vs_regulation", "prevention_vs_punishment",
            "state_vs_local", "universal_vs_targeted"]
    lex_lookup = {(r["party"], int(r["topic_id"])):
                  {d: r[d] for d in dims}
                  for _, r in lex_cell.iterrows()}

    out = {}
    for t in range(28):
        n_sign, n_co = coauth[t]
        M = np.full((len(PARTIES), len(PARTIES)), np.nan)
        for i, a in enumerate(PARTIES):
            for j, b in enumerate(PARTIES):
                if i == j:
                    M[i, j] = 1.0
                    continue
                # Co-authorship cosine for this topic.
                denom = (n_sign.get(a, 0) * n_sign.get(b, 0)) ** 0.5
                co_sim = (n_co.get((a, b), 0) / denom) if denom else 0.0
                # Lexicon distance for this topic.
                la = lex_lookup.get((a, t))
                lb = lex_lookup.get((b, t))
                if la and lb:
                    diffs = []
                    for d in dims:
                        va, vb = la[d], lb[d]
                        if not (np.isnan(va) or np.isnan(vb)):
                            diffs.append(abs(va - vb))
                    lex_sim = 1 - np.mean(diffs) / 2.0 if diffs else np.nan
                else:
                    lex_sim = np.nan
                # Combine: average where available.
                vals = [v for v in [co_sim, lex_sim] if not np.isnan(v)]
                M[i, j] = float(np.mean(vals)) if vals else np.nan
        out[t] = M
    return out


def main() -> None:
    print("Building per-topic similarity matrices ...")
    what = what_per_topic()
    why = why_per_topic()
    how = how_per_topic()

    # Build long table: per (topic, pair) — three sims and a fragmentation score.
    meta = pd.read_parquet(IN / "topic_meta.parquet").set_index("topic_id")
    rows = []
    for t in range(28):
        for i, a in enumerate(PARTIES):
            for j in range(i + 1, len(PARTIES)):
                b = PARTIES[j]
                w = what[t][i, j]; y = why[t][i, j]; h = how[t][i, j]
                if any(np.isnan([w, y, h])):
                    continue
                rows.append({"topic_id": t, "a": a, "b": b,
                             "pair": f"{a}–{b}",
                             "what": w, "why": y, "how": h})
    df = pd.DataFrame(rows)
    # Rank-normalise per axis globally (across all topic-pair cells).
    for c in ["what", "why", "how"]:
        df[f"{c}_n"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    df["spread"] = df[["what_n", "why_n", "how_n"]].std(axis=1)
    df["mean"] = df[["what_n", "why_n", "how_n"]].mean(axis=1)

    # Per-topic stats: how informative is each topic?
    topic_stats = (df.groupby("topic_id")
                     .agg(mean_spread=("spread", "mean"),
                          mean_align=("mean", "mean"),
                          mean_distinct_what=("what_n", "std"),
                          n_pairs=("pair", "count"))
                     .reset_index())
    topic_stats["label"] = topic_stats["topic_id"].map(
        lambda t: meta.loc[t, "label_terms"].split(",")[0:3])
    topic_stats["label"] = topic_stats["label"].str.join(", ")
    topic_stats["primary_organ"] = topic_stats["topic_id"].map(
        lambda t: meta.loc[t, "primary_organ"])

    print("\n=== Most informative topics ===")
    print("(high mean_distinct_what = parties take strongly different positions)\n")
    info = topic_stats.sort_values("mean_distinct_what", ascending=False).head(10)
    for _, r in info.iterrows():
        print(f"  [{r['topic_id']:>2}] {r['primary_organ']:>4}  "
              f"WHAT-spread={r['mean_distinct_what']:.3f}  "
              f"axes-spread={r['mean_spread']:.3f}  | "
              f"{r['label'][:60]}")

    print("\n=== Most fragmented topics ===")
    print("(high mean_spread = three axes disagree the most on this topic)\n")
    frag = topic_stats.sort_values("mean_spread", ascending=False).head(10)
    for _, r in frag.iterrows():
        print(f"  [{r['topic_id']:>2}] {r['primary_organ']:>4}  "
              f"axes-spread={r['mean_spread']:.3f}  "
              f"WHAT-spread={r['mean_distinct_what']:.3f}  | "
              f"{r['label'][:60]}")

    print("\n=== Pair-topic cells with the most extreme axis disagreement ===")
    print("(per pair, on which topic do the three axes disagree the most?)\n")
    for pair, sub in df.groupby("pair"):
        idx = sub["spread"].idxmax()
        r = sub.loc[idx]
        peak = max(["what", "why", "how"], key=lambda c: r[f"{c}_n"])
        valley = min(["what", "why", "how"], key=lambda c: r[f"{c}_n"])
        label = meta.loc[int(r['topic_id']), 'label_terms'].split(",")[0]
        if r["spread"] > 0.30:
            print(f"  {pair:>6}  spread={r['spread']:.3f}  "
                  f"{peak.upper()}↑ {valley.upper()}↓  | "
                  f"topic {r['topic_id']:>2}: {label[:50]}")

    df.to_parquet(IN / "per_topic_axes.parquet", index=False)
    topic_stats.to_parquet(IN / "per_topic_stats.parquet", index=False)

    # === Figure: per-topic "fragmentation × informativeness" map ===
    fig, ax = plt.subplots(figsize=(13, 9), dpi=140)
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    sizes = 80 + topic_stats["n_pairs"] * 1.5
    sc = ax.scatter(topic_stats["mean_distinct_what"],
                    topic_stats["mean_spread"],
                    s=sizes, c=topic_stats["mean_align"],
                    cmap="RdYlGn", vmin=0.35, vmax=0.75,
                    edgecolor="#333", linewidth=0.6, alpha=0.92)
    for _, r in topic_stats.iterrows():
        ax.annotate(f"[{int(r['topic_id'])}] {r['label'].split(',')[0][:25]}",
                    (r['mean_distinct_what'], r['mean_spread']),
                    fontsize=8, xytext=(6, 4), textcoords="offset points",
                    color="#333", alpha=0.85)
    ax.set_xlabel("WHAT differentiation — how much parties differ on this topic", fontsize=10)
    ax.set_ylabel("Three-axis fragmentation — how much WHAT, WHY, HOW disagree", fontsize=10)
    ax.set_title("Which topics carry the most information for a voter tool?\n"
                 "Upper-right = informative AND axes fragment "
                 "(good question candidates). Colour = mean pair alignment.",
                 fontsize=12, loc="left", pad=10)
    fig.colorbar(sc, ax=ax, label="mean pair alignment", fraction=0.04)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "17_per_topic_info.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
