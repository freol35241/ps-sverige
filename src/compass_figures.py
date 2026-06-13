"""Revealed-preference compass figures.

Caveats made visible
--------------------
Position score ∈ [-1, +1] is *agreement with the betänkande's main proposal*
(typically the cabinet's recommendation). It is NOT a free-floating
ideological direction — by construction, cabinet parties (L, KD, M) hit +1
on essentially every topic.

The interesting signal is *relative*: how parties differ from each other on
each topic, with the cabinet's row as the implicit zero point.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.figures import PARTY_COLOR

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def short_topic_label(label_terms: str, max_terms: int = 3, max_chars: int = 40) -> str:
    terms = [t.strip() for t in label_terms.split(",")]
    keep = []
    seen_root = set()
    for t in terms:
        # Skip duplicates (terms that share their first word with already-kept ones).
        head = t.split()[0]
        if head in seen_root:
            continue
        seen_root.add(head)
        keep.append(t)
        if len(keep) >= max_terms:
            break
    out = ", ".join(keep)
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def load_pivot():
    grid = pd.read_parquet(IN / "party_topic.parquet")
    meta = pd.read_parquet(IN / "topic_meta.parquet")
    pivot = (grid.pivot(index="parti", columns="topic_id", values="score")
                 .loc[PARTIES])
    return pivot, meta


def fig_heatmap_dev(pivot: pd.DataFrame, meta: pd.DataFrame) -> Path:
    """Heatmap of party stance, ordered by topic-level variance across parties."""
    centered = pivot.sub(pivot.median(axis=0), axis=1)
    order = pivot.std(axis=0).sort_values(ascending=False).index
    centered = centered[order]
    raw = pivot[order]

    labels = [short_topic_label(meta.set_index("topic_id").loc[t, "label_terms"],
                                max_terms=2, max_chars=32)
              for t in centered.columns]

    fig, ax = plt.subplots(figsize=(18, 8), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    im = ax.imshow(raw.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(PARTIES)))
    ax.set_yticklabels(PARTIES, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(centered.columns)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)

    # Cell text — show the stance value
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw.iat[i, j]
            color = "white" if abs(v) > 0.55 else "#333"
            ax.text(j, i, f"{v:+.2f}".replace("+", ""),
                    ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Revealed-preference heatmap — per-party stance per policy topic\n"
                 "+1 = unanimous Ja with cabinet's recommendation,  −1 = unanimous Nej. "
                 "Topics sorted by inter-party disagreement.",
                 fontsize=12, loc="left", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01,
                        label="stance ∈ [-1, +1]")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out = FIG / "11_compass_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_compass(pivot: pd.DataFrame, meta: pd.DataFrame) -> Path:
    """2D PCA on party × topic score matrix → revealed-preference compass."""
    X = pivot.to_numpy()
    # Standardise per topic so that topics where the cabinet uniformly agrees
    # don't dominate — we care about variation across parties on each topic.
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    p2 = PCA(n_components=2, random_state=0)
    coords = p2.fit_transform(X)
    print(f"compass PCA explained: {p2.explained_variance_ratio_}")

    # Identify which topics load most heavily on each axis for axis labels.
    load1 = pd.Series(p2.components_[0], index=pivot.columns)
    load2 = pd.Series(p2.components_[1], index=pivot.columns)
    print("\nTop +PC1 loadings:")
    for t in load1.sort_values(ascending=False).head(4).index:
        print(f"  topic {t}: {meta.set_index('topic_id').loc[t,'label_terms'][:80]}")
    print("Top -PC1 loadings:")
    for t in load1.sort_values().head(4).index:
        print(f"  topic {t}: {meta.set_index('topic_id').loc[t,'label_terms'][:80]}")
    print("Top +PC2 loadings:")
    for t in load2.sort_values(ascending=False).head(4).index:
        print(f"  topic {t}: {meta.set_index('topic_id').loc[t,'label_terms'][:80]}")
    print("Top -PC2 loadings:")
    for t in load2.sort_values().head(4).index:
        print(f"  topic {t}: {meta.set_index('topic_id').loc[t,'label_terms'][:80]}")

    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    for i, party in enumerate(PARTIES):
        c = PARTY_COLOR.get(party, "#888")
        ax.scatter(coords[i, 0], coords[i, 1], s=400, c=c,
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(party, coords[i], fontsize=14, fontweight="bold",
                    ha="center", va="center", color="white", zorder=4)
    ax.axhline(0, color="#BBB", lw=0.5); ax.axvline(0, color="#BBB", lw=0.5)
    ax.set_xlabel(f"PC1  ({p2.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2  ({p2.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("Revealed-preference compass\n"
                 "Parties projected on policy-stance variance, "
                 f"averaged across {pivot.shape[1]} topics, full mandate",
                 fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    out = FIG / "12_compass_2d.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_radar(pivot: pd.DataFrame, meta: pd.DataFrame) -> Path:
    """Per-party radar over a curated set of policy topics."""
    # Pick the 12 topics where parties differ most (highest std across parties).
    spread = pivot.std(axis=0).sort_values(ascending=False)
    topics_to_show = spread.head(12).index.tolist()
    labels = [short_topic_label(meta.set_index("topic_id").loc[t, "label_terms"],
                                 max_chars=24)
              for t in topics_to_show]

    # Polar coords.
    angles = np.linspace(0, 2 * np.pi, len(topics_to_show), endpoint=False)
    angles_loop = np.r_[angles, angles[:1]]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=140, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FAFAFA")
    for ax, party in zip(axes.flat, PARTIES):
        values = pivot.loc[party, topics_to_show].to_numpy()
        values_loop = np.r_[values, values[:1]]
        ax.plot(angles_loop, values_loop, "-",
                color=PARTY_COLOR.get(party, "#888"), lw=2)
        ax.fill(angles_loop, values_loop,
                color=PARTY_COLOR.get(party, "#888"), alpha=0.20)
        ax.set_ylim(-1, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks([-0.5, 0, 0.5, 1.0])
        ax.set_yticklabels(["-0.5", "0", "+0.5", "+1"], fontsize=6, alpha=0.5)
        ax.set_title(party, color=PARTY_COLOR.get(party, "#888"),
                     fontsize=14, fontweight="bold", pad=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-party revealed preferences over the most contested topics\n"
                 "(distance from centre = mean stance in [-1, +1], 12 highest-variance topics)",
                 fontsize=12, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "13_compass_radar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    pivot, meta = load_pivot()
    p1 = fig_heatmap_dev(pivot, meta)
    p2 = fig_compass(pivot, meta)
    p3 = fig_radar(pivot, meta)
    for p in (p1, p2, p3):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
