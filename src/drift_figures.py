"""Drift visualizations: party-centroid trajectories and per-riksmöte panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull

from src.figures import PARTY_COLOR, PARTY_ORDER

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

RIKSMOTEN = ["2022/23", "2023/24", "2024/25", "2025/26"]


def load():
    return pd.read_parquet(IN / "embedding_yearly.parquet")


def party_centroids(long: pd.DataFrame) -> pd.DataFrame:
    fit = long[long["in_fit"]]
    return (fit.groupby(["party_modal", "rm", "rm_idx"])[["PC1", "PC2"]]
                .mean().reset_index())


def fig_centroid_trajectories(long: pd.DataFrame) -> Path:
    cents = party_centroids(long).sort_values(["party_modal", "rm_idx"])

    fig, ax = plt.subplots(figsize=(13, 9), dpi=140)
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")

    for party in PARTY_ORDER:
        sub = cents[cents["party_modal"] == party]
        if len(sub) < 2:
            continue
        c = PARTY_COLOR.get(party, "#888")
        # Trajectory line.
        ax.plot(sub["PC1"], sub["PC2"], "-", color=c, lw=1.2, alpha=0.6, zorder=2)
        # Year markers, size growing with year.
        for _, r in sub.iterrows():
            size = 30 + 60 * (r["rm_idx"] / 3)
            ax.scatter(r["PC1"], r["PC2"], s=size, c=c,
                       edgecolor="white", linewidth=0.9, zorder=3)
        # Arrow on the last segment.
        last = sub.iloc[-2:]
        ax.annotate("", xy=(last.iloc[1]["PC1"], last.iloc[1]["PC2"]),
                    xytext=(last.iloc[0]["PC1"], last.iloc[0]["PC2"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.7),
                    zorder=2)
        # Label at the final year.
        end = sub.iloc[-1]
        ax.annotate(party, (end["PC1"], end["PC2"]),
                    fontsize=12, fontweight="bold", ha="center", va="center",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.32", fc=c,
                              ec="white", lw=1.2),
                    zorder=5)

    ax.axhline(0, color="#BBB", lw=0.5, zorder=1)
    ax.axvline(0, color="#BBB", lw=0.5, zorder=1)
    ax.set_xlabel("PC1 — government / opposition", fontsize=11)
    ax.set_ylabel("PC2 — within-bloc structure", fontsize=11)
    ax.set_title("Where each party drifted, 2022/23 → 2025/26\n"
                 "Dot size grows with year. Trajectories are Procrustes-aligned.",
                 fontsize=13, loc="left", pad=12)

    # Legend showing year-size encoding.
    for i, rm in enumerate(RIKSMOTEN):
        ax.scatter([], [], s=30 + 60 * (i / 3), c="#888",
                   edgecolor="white", linewidth=0.8, label=rm)
    ax.legend(loc="lower right", fontsize=9, title="Riksmöte",
              framealpha=0.9)

    fig.tight_layout()
    out = FIG / "05_centroid_trajectories.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_yearly_panels(long: pd.DataFrame) -> Path:
    fit = long[long["in_fit"]]
    cents = party_centroids(long)

    # Common axes from the union over all years.
    x_lo, x_hi = fit["PC1"].min() - 2, fit["PC1"].max() + 2
    y_lo, y_hi = fit["PC2"].min() - 2, fit["PC2"].max() + 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=140, sharex=True, sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, rm in zip(axes.flat, RIKSMOTEN):
        ax.set_facecolor("#FAFAFA")
        sub = fit[fit["rm"] == rm]

        for party in PARTY_ORDER:
            pts = sub[sub["party_modal"] == party][["PC1", "PC2"]].to_numpy()
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        color=PARTY_COLOR[party], alpha=0.10,
                        edgecolor=PARTY_COLOR[party], linewidth=0.5)
            except Exception:
                pass

        for party in PARTY_ORDER:
            ps = sub[sub["party_modal"] == party]
            if ps.empty:
                continue
            ax.scatter(ps["PC1"], ps["PC2"], s=18,
                       c=PARTY_COLOR[party], edgecolor="white", linewidth=0.4,
                       alpha=0.85, zorder=3)

        cs = cents[cents["rm"] == rm]
        for _, row in cs.iterrows():
            ax.annotate(row["party_modal"], (row["PC1"], row["PC2"]),
                        fontsize=10, fontweight="bold", ha="center", va="center",
                        color="white",
                        bbox=dict(boxstyle="circle,pad=0.26",
                                  fc=PARTY_COLOR.get(row["party_modal"], "#444"),
                                  ec="white", lw=1.0),
                        zorder=5)

        ax.axhline(0, color="#BBB", lw=0.4, zorder=1)
        ax.axvline(0, color="#BBB", lw=0.4, zorder=1)
        ax.set_title(rm, fontsize=12, loc="left")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    for ax in axes[1, :]:
        ax.set_xlabel("PC1 — gov / opp")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2 — within-bloc")

    fig.suptitle("The mandate, year by year — same axes (Procrustes-aligned)",
                 fontsize=14, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "06_yearly_panels.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_tido_collapse(long: pd.DataFrame) -> Path:
    """Zoom on the Tidö bloc to show the cluster collapse."""
    fit = long[long["in_fit"]]
    cents = party_centroids(long)
    parties = ["M", "KD", "L", "SD"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=140, sharex=True, sharey=True)
    fig.patch.set_facecolor("#FAFAFA")

    # Common zoom box (around the Tidö cluster across all years).
    tido = fit[fit["party_modal"].isin(parties)]
    x_lo, x_hi = tido["PC1"].min() - 1, tido["PC1"].max() + 1
    y_lo, y_hi = tido["PC2"].min() - 1, tido["PC2"].max() + 1

    for ax, rm in zip(axes, RIKSMOTEN):
        ax.set_facecolor("#FAFAFA")
        sub = fit[(fit["rm"] == rm) & (fit["party_modal"].isin(parties))]
        for party in parties:
            ps = sub[sub["party_modal"] == party]
            ax.scatter(ps["PC1"], ps["PC2"], s=40,
                       c=PARTY_COLOR[party], edgecolor="white", linewidth=0.5,
                       alpha=0.9, label=party if rm == RIKSMOTEN[0] else None,
                       zorder=3)
        cs = cents[(cents["rm"] == rm) & (cents["party_modal"].isin(parties))]
        for _, row in cs.iterrows():
            ax.annotate(row["party_modal"], (row["PC1"], row["PC2"]),
                        fontsize=11, fontweight="bold", ha="center", va="center",
                        color="white",
                        bbox=dict(boxstyle="circle,pad=0.26",
                                  fc=PARTY_COLOR.get(row["party_modal"], "#444"),
                                  ec="white", lw=1.0),
                        zorder=5)

        # Compute pairwise centroid distances within Tidö as a "convergence" stat.
        from itertools import combinations
        cs_xy = {p: cs[cs["party_modal"] == p][["PC1", "PC2"]].iloc[0].to_numpy()
                 if (cs["party_modal"] == p).any() else None
                 for p in parties}
        dists = [np.linalg.norm(cs_xy[a] - cs_xy[b])
                 for a, b in combinations(parties, 2)
                 if cs_xy[a] is not None and cs_xy[b] is not None]
        mean_d = np.mean(dists) if dists else float("nan")
        ax.set_title(f"{rm}\nmean pairwise centroid d = {mean_d:.2f}",
                     fontsize=10, loc="left")
        ax.axhline(0, color="#BBB", lw=0.4, zorder=1)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    axes[0].set_ylabel("PC2")
    for ax in axes:
        ax.set_xlabel("PC1")

    fig.suptitle("The Tidö bloc collapses into one cluster",
                 fontsize=14, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "07_tido_collapse.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    long = load()
    print(f"loaded {len(long):,} (MP × year) rows")
    p1 = fig_centroid_trajectories(long)
    p2 = fig_yearly_panels(long)
    p3 = fig_tido_collapse(long)
    print(f"wrote {p1}\nwrote {p2}\nwrote {p3}")


if __name__ == "__main__":
    main()
