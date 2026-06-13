"""Per-riksmöte HOW evolution.

Two signals over time:
(A) Co-authorship pattern shifts per year — who signs with whom.
(B) Lexicon scores per (party, riksmöte) — did SD's punishment vocabulary
    grow? Did the cabinet's state-orientation drift?

Speeches and reservations both carry a `dok_rm` / `rm` field — we can slice
the lexicon analysis by riksmöte without re-embedding.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.how_axis import LEXICON, _build_pattern
from src.figures import PARTY_COLOR

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
RIKSMOTEN = ["2022/23", "2023/24", "2024/25", "2025/26"]
PARTY_SET = set(PARTIES)


def coauthorship_per_year() -> pd.DataFrame:
    res = pd.read_parquet(IN / "reservations.parquet")
    rows = []
    for rm in RIKSMOTEN:
        sub = res[res["rm"] == rm]
        n_sign = Counter()
        n_co = Counter()
        for s in sub["partier"]:
            party_set = sorted(p for p in str(s).split(";") if p in PARTY_SET)
            for p in party_set:
                n_sign[p] += 1
            for a, b in combinations(party_set, 2):
                n_co[(a, b)] += 1
                n_co[(b, a)] += 1
        for i, a in enumerate(PARTIES):
            for j, b in enumerate(PARTIES):
                if i >= j:
                    continue
                denom = (n_sign.get(a, 0) * n_sign.get(b, 0)) ** 0.5
                sim = (n_co.get((a, b), 0) / denom) if denom else 0.0
                rows.append({"rm": rm, "a": a, "b": b,
                              "pair": f"{a}-{b}", "coauthor_sim": sim,
                              "n_co": n_co.get((a, b), 0)})
    return pd.DataFrame(rows)


def lexicon_per_year() -> pd.DataFrame:
    """Score each speech, attribute per riksmöte."""
    speeches = pd.read_parquet(IN / "speeches.parquet")
    speeches = speeches[speeches["parti"].isin(PARTIES)]
    speeches = speeches[speeches["replik"].fillna("N").str.upper() != "Y"]
    speeches = speeches[(speeches["n_words"] >= 50) &
                         (speeches["n_words"] <= 2000)]
    print(f"speeches to score for temporal lexicon: {len(speeches):,}")

    patterns = {k: (_build_pattern(L), _build_pattern(R))
                for k, (L, R) in LEXICON.items()}
    counts = []
    for _, sp in speeches.iterrows():
        text = (sp["text"] or "").lower()
        row = {"party": sp["parti"], "rm": sp["dok_rm"]}
        for dim, (lp, rp) in patterns.items():
            row[f"{dim}_l"] = len(lp.findall(text))
            row[f"{dim}_r"] = len(rp.findall(text))
        counts.append(row)
    df = pd.DataFrame(counts)

    dims = list(LEXICON.keys())
    out_rows = []
    for (party, rm), g in df.groupby(["party", "rm"]):
        row = {"party": party, "rm": rm, "n_speeches": len(g)}
        for d in dims:
            l_sum = g[f"{d}_l"].sum()
            r_sum = g[f"{d}_r"].sum()
            row[f"{d}_n"] = int(l_sum + r_sum)
            row[d] = ((l_sum - r_sum) / (l_sum + r_sum)
                      if (l_sum + r_sum) > 0 else np.nan)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main() -> None:
    print("(A) co-authorship per riksmöte ...")
    coauth = coauthorship_per_year()

    print("\n(B) lexicon per riksmöte ...")
    lex = lexicon_per_year()
    dims = list(LEXICON.keys())

    # === Co-authorship time trends ===
    pivot_co = coauth.pivot(index="pair", columns="rm", values="coauthor_sim")
    pivot_co = pivot_co[RIKSMOTEN]
    print("\nPairwise co-authorship similarity per riksmöte:")
    print(pivot_co.round(2).to_string())

    # Identify pairs whose co-authorship rose or fell most over the mandate
    delta = pivot_co.iloc[:, -1] - pivot_co.iloc[:, 0]
    print("\nLargest rises in co-authorship (2022/23 → 2025/26):")
    print(delta.sort_values(ascending=False).head(8).round(3).to_string())
    print("\nLargest drops:")
    print(delta.sort_values().head(8).round(3).to_string())

    # === Lexicon time trends ===
    print("\nLexicon trends — per party, change in dimension score 2022/23 → 2025/26:")
    for dim in dims:
        pivot_l = lex.pivot(index="party", columns="rm", values=dim).reindex(PARTIES)
        pivot_l = pivot_l[RIKSMOTEN]
        deltas = pivot_l[RIKSMOTEN[-1]] - pivot_l[RIKSMOTEN[0]]
        print(f"\n  {dim}:")
        print(pivot_l.round(2).to_string())
        print("  delta (2025/26 − 2022/23):")
        print(deltas.sort_values().round(2).to_string())

    coauth.to_parquet(IN / "temporal_coauthor.parquet", index=False)
    lex.to_parquet(IN / "temporal_lexicon.parquet", index=False)

    # === Figure: 2×2 grid, one panel per lexicon dimension ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=140, sharex=True)
    fig.patch.set_facecolor("#FAFAFA")
    x = np.arange(len(RIKSMOTEN))
    titles = {
        "market_vs_regulation":   "Market ↔ Regulation",
        "prevention_vs_punishment": "Prevention ↔ Punishment",
        "state_vs_local":          "State ↔ Local",
        "universal_vs_targeted":   "Universal ↔ Targeted",
    }
    for ax, dim in zip(axes.flat, dims):
        ax.set_facecolor("#FAFAFA")
        for p in PARTIES:
            sub = lex[lex["party"] == p].set_index("rm").reindex(RIKSMOTEN)
            ax.plot(x, sub[dim], "-o",
                    color=PARTY_COLOR.get(p, "#888"), lw=2,
                    markersize=8, label=p)
        ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN, fontsize=9)
        ax.set_title(titles[dim], fontsize=12, loc="left", pad=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="#888", lw=0.5, zorder=1)
    axes.flat[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                          fontsize=10, frameon=False)
    fig.suptitle("Temporal HOW — theory-of-change vocabulary by riksmöte",
                 fontsize=13, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    out1 = FIG / "19_temporal_how_lexicon.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out1}")

    # === Figure: co-authorship trajectory for selected pairs ===
    interesting_pairs = ["MP-V", "S-V", "S-MP", "C-MP", "C-V", "C-S",
                          "M-SD", "L-SD", "KD-SD", "L-M", "KD-M", "L-KD"]
    fig, ax = plt.subplots(figsize=(13, 7), dpi=140)
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    for pair in interesting_pairs:
        sub = coauth[coauth["pair"] == pair].set_index("rm").reindex(RIKSMOTEN)
        if sub["coauthor_sim"].notna().sum() == 0:
            continue
        a, b = pair.split("-")
        c = (PARTY_COLOR.get(a, "#888"), PARTY_COLOR.get(b, "#888"))
        avg = tuple(np.mean([
            int(PARTY_COLOR.get(p, "#888888").lstrip("#")[i:i+2], 16) / 255.0
            for p in [a, b]]) for i in (0, 2, 4))
        ax.plot(x, sub["coauthor_sim"], "-o", lw=2, markersize=7, color=avg)
        ax.annotate(pair, (x[-1], sub["coauthor_sim"].iat[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, va="center", color=avg)
    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN)
    ax.set_ylabel("Reservation co-authorship similarity")
    ax.set_title("Who signs reservations with whom, over the mandate",
                 fontsize=12, loc="left", pad=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = FIG / "20_temporal_coauthorship.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
