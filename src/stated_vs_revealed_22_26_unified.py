"""Rebuild stated-vs-revealed for 2022-26 with unified topic IDs.

Re-embeds 2022 manifesto sentences, assigns to unified topic centroids,
builds per (party, topic_id_unified) stated vector, and computes the
topic-matched cosine against reservation_vectors_22_26_unified.npz.

Then renders figures 26 (per-party gap) and 27 (per-topic gap matrix)
using the unified topic structure, matching the published 2022-26
article's figure conventions but with consistent topic IDs.

Outputs:
  data/processed/stated_vs_revealed_topic_22_26_unified.parquet
  data/processed/manifesto_2022_topic_assignment.parquet
  figures/web/26_topic_matched_gap.png        (replaces existing)
  figures/web/27_party_topic_gap_matrix.png   (replaces existing)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "web"
PUB = ROOT / "site" / "public" / "figures"

sys.path.insert(0, str(ROOT / "src"))

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
INK = "#222222"
BG = "#FAFAFA"

MIN_SENTENCES_PER_CELL = 5
MIN_TOPIC_SIMILARITY = 0.30


def topic_centroids():
    npz_a = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    npz_b = np.load(IN / "embed_recommendations_18_22.npz", allow_pickle=True)
    X = np.vstack([npz_a["X"], npz_b["X"]]).astype(np.float32)
    ids = list(npz_a["votering_id"]) + list(npz_b["votering_id"])
    topics = pd.read_parquet(IN / "topics_unified.parquet")
    id_to_topic = dict(zip(topics["votering_id"].astype(str),
                            topics["topic_id_unified"]))
    bucket: dict[int, list[int]] = {}
    for i, vid in enumerate(ids):
        t = id_to_topic.get(str(vid))
        if t is None:
            continue
        bucket.setdefault(int(t), []).append(i)
    topic_ids = sorted(bucket.keys())
    centroids = np.stack([X[bucket[t]].mean(axis=0) for t in topic_ids])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids, topic_ids


def main() -> None:
    print("== loading 2022 manifesto sentences ==")
    ann = pd.read_parquet(IN / "manifesto_sentences.parquet")
    sub = ann[ann["element"] != "other"].reset_index(drop=True)
    print(f"  {len(sub):,} substantive sentences")

    from sentence_transformers import SentenceTransformer
    print("  loading KBLab Swedish SBERT ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    print(f"  embedding {len(sub):,} sentences ...")
    embs = model.encode(sub["sentence"].tolist(), batch_size=32,
                        show_progress_bar=False, normalize_embeddings=True
                        ).astype(np.float32)
    print(f"  embeddings: {embs.shape}")

    print("\n== assigning to unified topics ==")
    centroids, topic_ids = topic_centroids()
    sims = embs @ centroids.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(embs)), best_idx]
    assigned = np.array([topic_ids[i] for i in best_idx])
    sub = sub.copy()
    sub["topic_id_unified"] = assigned
    sub["topic_sim"] = best_sim
    sub.drop(columns=["sentence"]).to_parquet(
        IN / "manifesto_2022_topic_assignment.parquet", index=False)
    print(f"  wrote manifesto_2022_topic_assignment.parquet")

    print("\n== building per-(party, topic) stated vectors ==")
    cells: dict[tuple[str, int], np.ndarray] = {}
    for (party, topic), g_idx in sub.groupby(
            ["party", "topic_id_unified"]).indices.items():
        idx = np.array(list(g_idx))
        rel_sim = sub.iloc[idx]["topic_sim"].to_numpy()
        keep = rel_sim >= MIN_TOPIC_SIMILARITY
        if keep.sum() < MIN_SENTENCES_PER_CELL:
            continue
        vec = embs[idx[keep]].mean(axis=0)
        cells[(party.upper(), int(topic))] = vec / np.linalg.norm(vec)
    print(f"  cells with stated vector: {len(cells)}")

    rv = np.load(IN / "reservation_vectors_22_26_unified.npz", allow_pickle=True)
    sp = np.load(IN / "reasoning_vectors_22_26_unified.npz", allow_pickle=True)
    res_V, res_N = rv["V"], rv["N"]
    sp_V, sp_N = sp["V"], sp["N"]
    # Use reservation if available, else speech-based (matches main article)
    V = np.where(res_N[..., None] > 0, res_V, sp_V).astype(np.float32)
    N = np.maximum(res_N, sp_N)

    rows = []
    for (party, topic), m_vec in cells.items():
        i = PARTIES.index(party)
        if N[i, topic] == 0:
            continue
        r_vec = V[i, topic]
        na, nb = np.linalg.norm(m_vec), np.linalg.norm(r_vec)
        if not (na and nb):
            continue
        sim = float(m_vec @ r_vec / (na * nb))
        n_man = int(((sub["party"] == party.lower()) &
                     (sub["topic_id_unified"] == topic)).sum())
        rows.append({
            "party": party, "topic_id_unified": topic,
            "n_manifesto_sentences": n_man,
            "stated_revealed_cos": sim,
        })
    gap = pd.DataFrame(rows)
    gap.to_parquet(IN / "stated_vs_revealed_topic_22_26_unified.parquet",
                   index=False)
    print(f"  wrote stated_vs_revealed_topic_22_26_unified.parquet "
          f"({len(gap)} cells)")

    per_party = gap.groupby("party")["stated_revealed_cos"].agg(
        ["mean", "std", "count"]).reset_index()
    print("\n  per-party mean topic-matched cosine, 2022-26 (unified):")
    print(per_party.round(3).to_string(index=False))

    # ---------------------------------------------------------------------
    # Figure 26: per-party bar
    # ---------------------------------------------------------------------
    print("\n== rendering figure 26 (per-party bar) ==")
    party_agg = per_party.set_index("party").reindex(PARTIES).reset_index()
    party_agg = party_agg.dropna(subset=["mean"])

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    parties_in_data = party_agg["party"].tolist()
    x = np.arange(len(parties_in_data))
    means = party_agg["mean"].to_numpy()
    stds = party_agg["std"].to_numpy()
    bars = ax.bar(x, means,
                  color=[PARTY_COLOR[p] for p in parties_in_data],
                  edgecolor="white", linewidth=1.2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#444",
                 capsize=4, lw=1.2, alpha=0.8)
    for bar, mean, n in zip(bars, means, party_agg["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{mean:.2f}\n(n={n})",
                ha="center", va="bottom", fontsize=9, color="#444",
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(parties_in_data, fontsize=12,
                                          fontweight="bold")
    ax.set_ylabel("Topic-matched stated-vs-revealed cosine")
    ax.set_ylim(0, 1)
    ax.set_title("Stated vs revealed, 2022-26 (unified topic IDs)\n"
                 "Mean within-topic cosine between manifesto text and "
                 "reservation text (speech-based for cabinet cells).",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "26_topic_matched_gap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, PUB / out.name)
    print(f"  wrote {out.name}")

    # ---------------------------------------------------------------------
    # Figure 27: per-topic heatmap
    # ---------------------------------------------------------------------
    print("\n== rendering figure 27 (per-topic gap matrix) ==")
    meta = pd.read_parquet(IN / "topic_meta_unified.parquet").set_index("topic_id")
    pivot = gap.pivot(index="party", columns="topic_id_unified",
                       values="stated_revealed_cos")
    pivot = pivot.reindex(index=[p for p in PARTIES if p in pivot.index])
    topic_agg = gap.groupby("topic_id_unified")["stated_revealed_cos"].agg(
        ["mean", "count"]).reset_index()
    topic_agg = topic_agg[topic_agg["count"] >= 3]
    topic_order = topic_agg.sort_values("mean")["topic_id_unified"].tolist()
    topic_order = [t for t in topic_order if t in pivot.columns]
    pivot = pivot[topic_order]
    topic_labels = [
        meta.loc[t, "label_terms"].split(",")[0].strip()[:22]
        if t in meta.index else f"t{t}"
        for t in topic_order
    ]
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0.2, vmax=0.8,
                   aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(topic_labels)))
    ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if pd.notna(v):
                color = "#222" if 0.4 < v < 0.65 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    ax.set_title("Per-topic stated-vs-revealed cosine, 2022-26 "
                 "(unified topic IDs)\n"
                 "Red = manifesto and reservation language diverge; "
                 "green = consistent. Topics sorted by mean across "
                 "parties (most divergent at left).",
                 loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label="cosine")
    fig.tight_layout()
    out = FIG / "27_party_topic_gap_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, PUB / out.name)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
