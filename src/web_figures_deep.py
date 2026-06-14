"""Deep-dive web figures for the article restructure.

Adds:
  28 — centroid trajectories (all 8 parties' drift across the mandate)
  29 — temporal HOW lexicon (4 panels)
  30 — temporal coauthorship (selected pairs)
  31 — defectors (top behavioural defectors by MP)
  32 — speech vs reservation register cosine per party
  33 — per-topic vote stance heatmap

All output English, no em-dashes. Written to figures/web/ and mirrored to
site/public/figures/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import PARTY_COLOR, PARTY_ORDER

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
WEB = ROOT / "figures" / "web"
SITE = ROOT / "site" / "public" / "figures"

RIKSMOTEN = ["2022/23", "2023/24", "2024/25", "2025/26"]
BG = "#FAFAFA"
INK = "#222222"
MUTED = "#888888"
GRID = "#DDDDDD"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.titlepad": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": BG,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def _save(fig: plt.Figure, name: str) -> Path:
    WEB.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    out = WEB / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, SITE / name)
    return out


# ---------------------------------------------------------------------------
# 28. Centroid trajectories
# ---------------------------------------------------------------------------
def fig_centroid_trajectories() -> Path:
    long = pd.read_parquet(IN / "embedding_yearly.parquet")
    fit = long[long["in_fit"]]
    cents = (fit.groupby(["party_modal", "rm", "rm_idx"])[["PC1", "PC2"]]
                .mean().reset_index()
                .sort_values(["party_modal", "rm_idx"]))

    fig, ax = plt.subplots(figsize=(12, 8))

    for party in PARTY_ORDER:
        sub = cents[cents["party_modal"] == party]
        if len(sub) < 2 or party == "-":
            continue
        c = PARTY_COLOR.get(party, "#888")
        ax.plot(sub["PC1"], sub["PC2"], "-", color=c, lw=1.5, alpha=0.55, zorder=2)
        for _, r in sub.iterrows():
            size = 35 + 80 * (r["rm_idx"] / 3)
            ax.scatter(r["PC1"], r["PC2"], s=size, c=c,
                       edgecolor="white", linewidth=0.9, zorder=3)
        last = sub.iloc[-2:]
        ax.annotate("",
                    xy=(last.iloc[1]["PC1"], last.iloc[1]["PC2"]),
                    xytext=(last.iloc[0]["PC1"], last.iloc[0]["PC2"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.2, alpha=0.75),
                    zorder=2)
        end = sub.iloc[-1]
        ax.annotate(party, (end["PC1"], end["PC2"]),
                    fontsize=12, fontweight="bold", ha="center", va="center",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.36", fc=c,
                              ec="white", lw=1.4),
                    zorder=5)

    ax.axhline(0, color=GRID, lw=0.6, zorder=1)
    ax.axvline(0, color=GRID, lw=0.6, zorder=1)
    ax.set_xlabel("PC1 (government vs opposition)")
    ax.set_ylabel("PC2 (within-bloc structure)")
    ax.set_title("Where each party drifted, 2022/23 to 2025/26\n"
                 "Dot size grows with parliamentary year. Procrustes-aligned.",
                 loc="left", fontsize=13, pad=14)

    for i, rm in enumerate(RIKSMOTEN):
        ax.scatter([], [], s=35 + 80 * (i / 3), c="#888",
                   edgecolor="white", linewidth=0.8, label=rm)
    ax.legend(loc="lower right", fontsize=10, title="Year", framealpha=0.92)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "28_centroid_trajectories.png")


# ---------------------------------------------------------------------------
# 29. Temporal HOW lexicon
# ---------------------------------------------------------------------------
def fig_temporal_how_lexicon() -> Path:
    lex = pd.read_parquet(IN / "temporal_lexicon.parquet")
    dims = ["market_vs_regulation", "prevention_vs_punishment",
            "state_vs_local", "universal_vs_targeted"]
    titles = {
        "market_vs_regulation":
            "Market vs regulation  (higher = more market vocabulary)",
        "prevention_vs_punishment":
            "Prevention vs punishment  (higher = more prevention)",
        "state_vs_local":
            "Central state vs local  (higher = more central)",
        "universal_vs_targeted":
            "Universal vs targeted  (higher = more universal welfare)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    x = np.arange(len(RIKSMOTEN))

    for ax, dim in zip(axes.flat, dims):
        for p in PARTIES:
            sub = (lex[lex["party"] == p]
                   .set_index("rm").reindex(RIKSMOTEN))
            ax.plot(x, sub[dim], "-o",
                    color=PARTY_COLOR.get(p, "#888"), lw=2,
                    markersize=7, alpha=0.9)
            ax.annotate(p, (x[-1], sub[dim].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=10, fontweight="bold",
                        color=PARTY_COLOR.get(p, "#444"),
                        va="center", zorder=6)
        ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN, fontsize=10)
        ax.axhline(0, color=GRID, lw=0.6, zorder=1)
        ax.set_title(titles[dim], loc="left", fontsize=11.5, pad=10)
        ax.grid(alpha=0.25, color=GRID)
        ax.tick_params(length=0)

    fig.suptitle("Mechanism vocabulary, per party, across the mandate",
                 x=0.02, ha="left", y=1.00, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "29_temporal_how_lexicon.png")


# ---------------------------------------------------------------------------
# 30. Temporal coauthorship
# ---------------------------------------------------------------------------
def fig_temporal_coauthorship() -> Path:
    co = pd.read_parquet(IN / "temporal_coauthor.parquet")

    OPPOSITION_PAIRS = ["V-MP", "V-S", "S-MP", "C-V", "C-S", "C-MP"]
    SD_PAIRS = ["M-SD", "KD-SD", "L-SD"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(RIKSMOTEN))

    def _draw(ax, pairs, title):
        ax.axhline(0, color=GRID, lw=0.6)
        for pair in pairs:
            sub = (co[co["pair"] == pair].set_index("rm").reindex(RIKSMOTEN))
            if sub["coauthor_sim"].notna().sum() == 0:
                continue
            a, b = pair.split("-")
            ca = PARTY_COLOR.get(a, "#888")
            cb = PARTY_COLOR.get(b, "#888")
            rgb = tuple(np.mean([
                int(c.lstrip("#")[i:i+2], 16) / 255.0
                for c in (ca, cb)]) for i in (0, 2, 4))
            ax.plot(x, sub["coauthor_sim"], "-o", lw=2.2, markersize=7,
                    color=rgb, alpha=0.95)
            ax.annotate(pair, (x[-1], sub["coauthor_sim"].iloc[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=11, color=rgb, fontweight="bold",
                        va="center")
        ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN, fontsize=10)
        ax.set_title(title, loc="left", fontsize=13, pad=10)
        ax.set_ylim(-0.05, 0.55)
        ax.grid(alpha=0.25, color=GRID)
        ax.tick_params(length=0)

    _draw(axes[0], OPPOSITION_PAIRS, "Opposition pairs: active co-authorship")
    _draw(axes[1], SD_PAIRS, "SD with cabinet parties: structural zero")

    axes[0].set_ylabel("Reservation co-authorship similarity")

    fig.suptitle("Who signs reservations with whom, across the mandate",
                 x=0.02, ha="left", y=1.00, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "30_temporal_coauthorship.png")


# ---------------------------------------------------------------------------
# 31. Defectors
# ---------------------------------------------------------------------------
def fig_defectors() -> Path:
    d = pd.read_parquet(IN / "defectors.parquet")
    d = d.sort_values("ratio_own_over_near", ascending=False)
    head = d.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    y = np.arange(len(head))
    ax.barh(y, head["ratio_own_over_near"],
            color=[PARTY_COLOR.get(p, "#888") for p in head["party_modal"]],
            edgecolor="white", linewidth=0.6)
    labels = [f"{n}  ({p} closer to {q})"
              for n, p, q in zip(head["namn"], head["party_modal"], head["nearest_other"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#444", lw=0.8, linestyle="--")
    ax.text(1.04, len(head) - 0.5, "equidistant", fontsize=10, color="#444",
            style="italic")
    ax.set_xlabel("distance to own party / distance to nearest other party (log scale)")
    ax.set_title("Top behavioural defectors, full mandate\n"
                 "Each bar: an MP whose vote pattern places them closer to another party's centroid than to their own.",
                 loc="left", fontsize=13, pad=14)

    for yi, val in zip(y, head["ratio_own_over_near"].to_numpy()):
        ax.text(val * 1.06, yi, f"{val:.1f}x", va="center", fontsize=9, color="#333")
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "31_defectors.png")


# ---------------------------------------------------------------------------
# 32. Speech vs reservation register
# ---------------------------------------------------------------------------
def fig_speech_vs_reservation() -> Path:
    """For each party compute cosine between their mean speech reasoning
    vector and their mean reservation reasoning vector, restricted to topics
    covered by both. High = the same party uses the same register in chamber
    speeches as in written reservations. Low = the two registers diverge.
    Cabinet parties have very few reservations (n=8 to 11 topic cells) so
    their numbers are noisy and are marked as such."""
    rv = np.load(IN / "reservation_vectors.npz", allow_pickle=True)
    sv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)

    parties_in_file = list(rv["parties"])
    rows = []
    for i, p in enumerate(parties_in_file):
        if p not in PARTIES:
            continue
        res_present = rv["N"][i] > 0
        spk_present = sv["N"][i] > 0
        both = res_present & spk_present
        if not both.any():
            continue
        res_mean = rv["V"][i, both].mean(axis=0)
        spk_mean = sv["V"][i, both].mean(axis=0)
        cos = float(res_mean @ spk_mean /
                    (np.linalg.norm(res_mean) * np.linalg.norm(spk_mean)))
        rows.append({"party": p, "cos": cos,
                     "n_overlap": int(both.sum()),
                     "n_res": int(rv["N"][i].sum()),
                     "n_spk": int(sv["N"][i].sum())})
    df = pd.DataFrame(rows).set_index("party").loc[
        [p for p in PARTIES if p in [r["party"] for r in rows]]]
    print(df)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(df))
    THIN = df["n_res"] < 50
    colors = [PARTY_COLOR[p] for p in df.index]
    bars = ax.bar(x, df["cos"],
                  color=colors,
                  edgecolor="white", linewidth=1.2)
    for bar, t in zip(bars, THIN):
        if t:
            bar.set_alpha(0.30)
            bar.set_hatch("//")
    for i, (p, row) in enumerate(df.iterrows()):
        ax.text(i, row["cos"] + 0.015, f"{row['cos']:.2f}",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color="#888" if THIN.iloc[i] else "#222")
    ax.set_xticks(x); ax.set_xticklabels(df.index, fontsize=13, fontweight="bold")
    ax.set_ylabel("Cosine, speech reasoning vs reservation reasoning")
    ax.set_ylim(0, 1.0)
    ax.set_title("Same party, two registers: chamber speech vs written reservation\n"
                 "Cabinet bars (L, KD, M) are hatched and faded: too few reservations to compare reliably.",
                 loc="left", fontsize=13, pad=14)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "32_speech_vs_reservation.png")


# ---------------------------------------------------------------------------
# 33. Per-topic vote stance heatmap
# ---------------------------------------------------------------------------
def fig_per_topic_stance() -> Path:
    pt = pd.read_parquet(IN / "party_topic.parquet")
    meta = pd.read_parquet(IN / "topic_meta.parquet")

    def clean_label(row):
        terms = (row.get("label_terms") or "").split(",")
        terms = [t.strip() for t in terms if t.strip()]
        return ", ".join(terms[:3]) if terms else f"topic {row['topic_id']}"

    meta = meta.copy()
    meta["label"] = meta.apply(clean_label, axis=1)
    label_map = dict(zip(meta["topic_id"], meta["label"]))

    pivot = pt.pivot(index="parti", columns="topic_id", values="score")
    pivot = pivot.loc[[p for p in PARTIES if p in pivot.index]]

    var_per_topic = pivot.var(axis=0)
    order = var_per_topic.sort_values(ascending=False).index.tolist()
    pivot = pivot[order]

    n_show = min(20, len(order))
    pivot = pivot.iloc[:, :n_show]
    topics_shown = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn",
                   vmin=-1, vmax=1, aspect="auto")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11, fontweight="bold")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([label_map.get(t, f"t{t}") for t in topics_shown],
                       rotation=40, ha="right", fontsize=9)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.isna(v):
                continue
            color = "white" if abs(v) > 0.5 else "#222"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Where parties disagree most: vote stance per topic\n"
                 f"Top {n_show} topics by cross-party variance. +1 = consistent Yes, -1 = consistent No.",
                 loc="left", fontsize=13, pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02,
                        label="mean vote stance")
    cbar.outline.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "33_per_topic_stance.png")


# ---------------------------------------------------------------------------
# 34. Pair vote vs reasoning scatter
# ---------------------------------------------------------------------------
def fig_pair_vote_vs_reasoning() -> Path:
    """For each unordered party pair compute:
    - vote similarity: 1 - mean |stance_a - stance_b| / 2 across the 28 topics
    - reasoning similarity: mean topic-centred cosine across topics that
      both parties have a reasoning vector for
    Plot as scatter with trend line, highlight three named pairs."""
    pt = pd.read_parquet(IN / "party_topic.parquet")
    pivot = pt.pivot(index="parti", columns="topic_id",
                     values="score").loc[PARTIES]
    X = pivot.to_numpy()
    n = len(PARTIES)
    vote = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vote[i, j] = 1 - np.nanmean(np.abs(X[i] - X[j])) / 2.0

    rv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    V, N = rv["V"], rv["N"]
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
    df = pd.DataFrame(pairs)

    xs, ys = df["vote"].to_numpy(), df["reasoning"].to_numpy()
    a_fit, b_fit = np.polyfit(xs, ys, 1)

    NAMED = {
        ("S", "C"):  ("marriage of convenience: vote alike, argue apart", "below"),
        ("M", "SD"): ("vote alike and argue alike", "above"),
        ("V", "MP"): ("vote alike and argue alike", "above"),
    }

    fig, ax = plt.subplots(figsize=(13, 8))
    xline = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
    ax.plot(xline, a_fit * xline + b_fit, "--", color="#BBB", lw=1.4,
            label="trend (vote predicts reasoning)", zorder=1)

    for _, r in df.iterrows():
        key = (r["a"], r["b"]) if (r["a"], r["b"]) in NAMED else \
              ((r["b"], r["a"]) if (r["b"], r["a"]) in NAMED else None)
        if key is not None:
            colour = "#1B9E77" if NAMED[key][1] == "above" else "#D95F02"
            ax.scatter(r["vote"], r["reasoning"], s=280, alpha=0.95,
                       edgecolor=colour, linewidth=2.6, c="white", zorder=5)
            ax.annotate(r["pair"], (r["vote"], r["reasoning"]),
                        fontsize=13, fontweight="bold", color=colour,
                        xytext=(12, 10), textcoords="offset points",
                        zorder=6)
        else:
            ax.scatter(r["vote"], r["reasoning"], s=120, alpha=0.55,
                       edgecolor="#555", linewidth=0.5, c="#9CB7D4",
                       zorder=3)
            ax.annotate(r["pair"], (r["vote"], r["reasoning"]),
                        fontsize=9, color="#777",
                        xytext=(6, 4), textcoords="offset points",
                        zorder=4)

    ax.text(0.99, 0.04,
            "above trend: closer in reasoning than in votes\n"
            "below trend: closer in votes than in reasoning",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#DDD", lw=0.8))

    ax.set_xlabel("Vote-stance similarity")
    ax.set_ylabel("Reasoning similarity (topic-centred cosine)")
    ax.set_title("Do parties that vote together also reason together?\n"
                 "Each point is one party pair. The dashed line is the cross-pair trend.",
                 loc="left", fontsize=13, pad=14)
    ax.grid(alpha=0.3, color=GRID)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "34_vote_vs_reasoning.png")


# ---------------------------------------------------------------------------
# 35. Intra-party cohesion
# ---------------------------------------------------------------------------
def fig_intra_party_cohesion() -> Path:
    """For each party, mean distance from MPs to party centroid in the
    4D PCA space used for the fit. Lower = more unified caucus."""
    emb = pd.read_parquet(IN / "mp_embedding.parquet")
    fit = emb[emb["in_fit"]].copy()
    pc_cols = ["PC1", "PC2", "PC3", "PC4"]

    rows = []
    for party, g in fit.groupby("party_modal"):
        if party not in PARTIES or len(g) < 3:
            continue
        cent = g[pc_cols].mean().to_numpy()
        dists = np.linalg.norm(g[pc_cols].to_numpy() - cent, axis=1)
        rows.append({"party": party,
                     "n_mps": len(g),
                     "mean_dist": float(dists.mean()),
                     "median_dist": float(np.median(dists)),
                     "p90_dist": float(np.percentile(dists, 90))})
    df = pd.DataFrame(rows).set_index("party").loc[
        [p for p in PARTIES if p in [r["party"] for r in rows]]]
    df = df.sort_values("mean_dist")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    y = np.arange(len(df))
    ax.barh(y, df["mean_dist"],
            color=[PARTY_COLOR[p] for p in df.index],
            edgecolor="white", linewidth=1.2)
    for yi, (p, row) in zip(y, df.iterrows()):
        ax.text(row["mean_dist"] + 0.04, yi,
                f"{row['mean_dist']:.2f}  ({row['n_mps']} MPs)",
                va="center", fontsize=10, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(df.index, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean MP distance to party centroid (4D PCA space)")
    ax.set_title("Which party caucuses are the most unified?\n"
                 "Lower = MPs cluster more tightly around their party's mean voting position.",
                 loc="left", fontsize=13, pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(length=0)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, "35_intra_party_cohesion.png")


# ---------------------------------------------------------------------------
# 36. Manifesto coverage
# ---------------------------------------------------------------------------
def fig_manifesto_coverage() -> Path:
    """For each (party, topic), count manifesto sentences classified to that
    topic with topic_sim >= 0.30. Show as heatmap, parties as rows."""
    ma = pd.read_parquet(IN / "manifesto_topic_assignment.parquet")
    meta = pd.read_parquet(IN / "topic_meta.parquet")

    SIM = 0.30
    keep = ma[ma["topic_sim"] >= SIM]

    pivot = (keep.groupby(["party", "topic_id"])
                 .size().unstack(fill_value=0))
    pivot = pivot.reindex([p.lower() for p in PARTIES])
    pivot.index = [p.upper() for p in pivot.index]

    def clean_label(row):
        terms = (row.get("label_terms") or "").split(",")
        terms = [t.strip() for t in terms if t.strip()]
        return ", ".join(terms[:2]) if terms else f"t{row['topic_id']}"

    meta = meta.copy()
    meta["label"] = meta.apply(clean_label, axis=1)
    label_map = dict(zip(meta["topic_id"], meta["label"]))

    coverage = (pivot >= 5).sum(axis=1)
    total_sents = pivot.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    topic_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    display = pivot[topic_order]
    im = ax.imshow(np.log1p(display.to_numpy()), cmap="Greens",
                   aspect="auto", vmin=0, vmax=np.log1p(50))
    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(topic_order)))
    ax.set_xticklabels([label_map.get(t, f"t{t}") for t in topic_order],
                       rotation=45, ha="right", fontsize=8.5)
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            v = display.iloc[i, j]
            if v == 0:
                continue
            colour = "white" if v > 20 else "#222"
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    fontsize=7.5, color=colour)
    ax.set_title("Manifesto sentence count per (party, topic)",
                 loc="left", fontsize=12)
    ax.tick_params(length=0)

    ax = axes[1]
    cov = coverage.sort_values()
    y = np.arange(len(cov))
    ax.barh(y, cov.values,
            color=[PARTY_COLOR[p] for p in cov.index],
            edgecolor="white", linewidth=1.0)
    for yi, p in zip(y, cov.index):
        ax.text(cov[p] + 0.4, yi,
                f"{cov[p]} of 28\n({int(total_sents[p])} sentences)",
                va="center", fontsize=9, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(cov.index, fontsize=11, fontweight="bold")
    ax.set_xlabel("Topics with >=5 on-topic sentences")
    ax.set_xlim(0, max(cov.max() + 6, 28))
    ax.set_title("Manifesto topical coverage", loc="left", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(length=0)

    fig.suptitle("How much of policy space does each manifesto address?",
                 x=0.02, ha="left", y=1.00, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "36_manifesto_coverage.png")


# ---------------------------------------------------------------------------
# 37. Temporal reasoning drift (SBERT)
# ---------------------------------------------------------------------------
def fig_temporal_reasoning_drift() -> Path:
    """Per (party, parliamentary year) compute the mean SBERT embedding of
    that party's reasoning speeches. Plot year-over-year cosine drift."""
    from sentence_transformers import SentenceTransformer

    rs = pd.read_parquet(IN / "reasoning_speeches.parquet")
    rs["rm"] = rs["dok_datum"].apply(_riksmote_from_date)
    rs = rs[rs["parti"].isin(PARTIES) & rs["rm"].isin(RIKSMOTEN)].reset_index(drop=True)

    print(f"embedding {len(rs)} speeches ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    embs = model.encode(rs["text"].tolist(),
                         batch_size=32, show_progress_bar=True,
                         normalize_embeddings=True)

    rows = []
    for (party, rm), g in rs.groupby(["parti", "rm"]):
        positions = g.index.to_numpy()
        mean_vec = embs[positions].mean(axis=0)
        rows.append({"party": party, "rm": rm, "n": len(g),
                     "vec": mean_vec})
    yearly = pd.DataFrame(rows)

    drift_rows = []
    for party, g in yearly.groupby("party"):
        g = g.set_index("rm").reindex(RIKSMOTEN)
        for i, rm in enumerate(RIKSMOTEN):
            if i == 0:
                drift_rows.append({"party": party, "rm": rm,
                                    "cos_to_first": 1.0})
                continue
            a = g.loc[RIKSMOTEN[0]]["vec"]
            b = g.loc[rm]["vec"]
            if a is None or b is None or (hasattr(a, "__len__") is False) \
               or (hasattr(b, "__len__") is False):
                continue
            try:
                a = np.asarray(a); b = np.asarray(b)
                if a.size and b.size:
                    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                    drift_rows.append({"party": party, "rm": rm,
                                        "cos_to_first": cos})
            except Exception:
                continue
    drift = pd.DataFrame(drift_rows)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(RIKSMOTEN))
    for p in PARTIES:
        sub = (drift[drift["party"] == p].set_index("rm").reindex(RIKSMOTEN))
        if sub["cos_to_first"].notna().sum() < 2:
            continue
        ax.plot(x, sub["cos_to_first"], "-o",
                color=PARTY_COLOR[p], lw=2.2, markersize=8, alpha=0.95)
        ax.annotate(p, (x[-1], sub["cos_to_first"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=11, fontweight="bold",
                    color=PARTY_COLOR[p], va="center")
    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN, fontsize=10)
    ax.set_ylabel("Cosine to 2022/23 baseline")
    ax.set_title("How far each party's chamber rhetoric drifted from year one\n"
                 "Per-party mean SBERT embedding of reasoning speeches per parliamentary year.",
                 loc="left", fontsize=13, pad=14)
    ax.grid(alpha=0.3, color=GRID)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "37_temporal_reasoning_drift.png")


def _riksmote_from_date(d) -> str:
    """Date to riksmöte string. Riksmöte starts mid-September."""
    if pd.isna(d):
        return ""
    y, m = d.year, d.month
    if m >= 9:
        return f"{y}/{str(y+1)[-2:]}"
    return f"{y-1}/{str(y)[-2:]}"


# ---------------------------------------------------------------------------
# 38. Cross-cutting party coherence between four questions (SBERT)
# ---------------------------------------------------------------------------
def fig_party_coherence() -> Path:
    """Per party, embed all sentences labelled diagnosis, end_state, and
    mechanism in the manifesto. Take per-element mean. Compute pairwise
    cosine within each party between the three element vectors.

    High pairwise cosine = internally coherent worldview (diagnosis, vision
    and mechanism use the same vocabulary).
    """
    from sentence_transformers import SentenceTransformer

    ann = pd.read_parquet(IN / "manifesto_sentences.parquet")
    ann = ann[ann["element"] != "other"]

    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    el_vec = {}
    el_n = {}
    for (party, element), g in ann.groupby(["party", "element"]):
        if len(g) < 5:
            continue
        embs = model.encode(g["sentence"].tolist(),
                             batch_size=32, show_progress_bar=False,
                             normalize_embeddings=True)
        el_vec[(party, element)] = embs.mean(axis=0)
        el_n[(party, element)] = len(g)

    rows = []
    for party in [p.lower() for p in PARTIES]:
        keys = [k for k in el_vec.keys() if k[0] == party]
        elements_have = [k[1] for k in keys]
        if not all(e in elements_have for e in ("diagnosis", "end_state", "mechanism")):
            continue
        d = el_vec[(party, "diagnosis")]
        e = el_vec[(party, "end_state")]
        m = el_vec[(party, "mechanism")]
        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        rows.append({"party": party.upper(),
                     "d_e": cos(d, e),
                     "d_m": cos(d, m),
                     "e_m": cos(e, m),
                     "mean": (cos(d, e) + cos(d, m) + cos(e, m)) / 3,
                     "n_d": el_n[(party, "diagnosis")],
                     "n_e": el_n[(party, "end_state")],
                     "n_m": el_n[(party, "mechanism")]})
    df = pd.DataFrame(rows).set_index("party")
    df = df.loc[[p for p in PARTIES if p in df.index]]
    df = df.sort_values("mean", ascending=False)
    print(df)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(df))
    width = 0.27
    ax.bar(x - width, df["d_e"], width, label="diagnosis vs end state",
           color="#3B7DD8", edgecolor="white")
    ax.bar(x, df["d_m"], width, label="diagnosis vs mechanism",
           color="#D8693B", edgecolor="white")
    ax.bar(x + width, df["e_m"], width, label="end state vs mechanism",
           color="#3BD89B", edgecolor="white")
    for i, p in enumerate(df.index):
        ax.text(i, df.loc[p, "mean"] + 0.012, f"mean: {df.loc[p, 'mean']:.2f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
    ax.set_xticks(x); ax.set_xticklabels(df.index, fontsize=12, fontweight="bold")
    ax.set_ylabel("Cosine between element vectors")
    ax.set_ylim(0, max(df[["d_e", "d_m", "e_m"]].to_numpy().max() + 0.10, 1.0))
    ax.set_title("Do parties present internally coherent worldviews?\n"
                 "Within-party cosine between each pair of element vectors (diagnosis, end state, mechanism).",
                 loc="left", fontsize=13, pad=14)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "38_party_coherence.png")


# ---------------------------------------------------------------------------
def main() -> None:
    for f in (fig_centroid_trajectories,
              fig_temporal_how_lexicon,
              fig_temporal_coauthorship,
              fig_defectors,
              fig_speech_vs_reservation,
              fig_per_topic_stance,
              fig_pair_vote_vs_reasoning,
              fig_intra_party_cohesion,
              fig_manifesto_coverage,
              fig_temporal_reasoning_drift,
              fig_party_coherence):
        p = f()
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
