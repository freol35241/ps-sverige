"""Render per-mandate per-topic stance figures using unified topic IDs.

Produces:
  33_per_topic_stance.png  — 2022-26 (replaces the existing main-article figure)
  56_per_topic_stance_18_22.png  — 2018-22
"""
from __future__ import annotations

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


def render(tag: str, title: str, out_name: str) -> None:
    stats = pd.read_parquet(IN / f"per_topic_stats_{tag}.parquet")
    meta = pd.read_parquet(IN / "topic_meta_unified.parquet").set_index("topic_id")

    # Pivot to party × topic
    pivot = stats.pivot(index="party", columns="topic_id_unified", values="mean_stance")
    pivot = pivot.reindex(PARTIES)
    # Sort columns by cross-party variance (descending)
    col_var = pivot.var(axis=0).sort_values(ascending=False)
    top_20 = col_var.head(20).index.tolist()
    pivot = pivot[top_20]

    # Topic labels from meta
    topic_labels = []
    for t in top_20:
        if t in meta.index:
            term = meta.loc[t, "label_terms"].split(",")[0].strip()
            topic_labels.append(term[:18])
        else:
            topic_labels.append(f"t{t}")

    fig, ax = plt.subplots(figsize=(13.5, 4.6), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(PARTIES)))
    ax.set_yticklabels(PARTIES, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(topic_labels)))
    ax.set_xticklabels(topic_labels, rotation=40, ha="right", fontsize=9)
    # Cell annotations
    arr = pivot.to_numpy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if pd.notna(v):
                col = "#222" if -0.5 < v < 0.5 else "white"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color=col)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold",
                 color=INK, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="mean vote stance")
    fig.tight_layout()
    p = FIG / out_name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    import shutil
    shutil.copyfile(p, PUB / out_name)
    print(f"  wrote {out_name}")


def main() -> None:
    print("== rendering per-topic stance figures (unified IDs) ==")
    render("22_26",
           "Where parties disagree most: 2022-26 mandate\n"
           "Mean vote stance per (party, topic). +1 = consistent Yes, -1 = consistent No, "
           "0 = mixed. Top 20 topics by cross-party variance, unified topic IDs.",
           "33_per_topic_stance.png")
    render("18_22",
           "Where parties disagree most: 2018-22 mandate\n"
           "Mean vote stance per (party, topic). Same metric and unified topic IDs as the 2022-26 figure.",
           "56_per_topic_stance_18_22.png")


if __name__ == "__main__":
    main()
