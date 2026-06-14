"""Topic-matched stated-vs-revealed comparison.

The previous global comparison was contaminated by a register effect: the
cabinet's "revealed" reasoning comes from polished majority-opinion text,
the opposition's from technical reservation prose. Comparing register-
across-source-types mixed up "what was said" with "how it was said".

The fix:
  1. Compute topic centroids from the existing 28-cluster KMeans on
     recommendation embeddings.
  2. For each manifesto sentence, find its nearest topic by cosine.
  3. Per (party, topic) build a manifesto vector (mean of assigned
     sentence embeddings) and a reservation vector (we already have it).
  4. Per cell, cosine(manifesto, reservation) = topic-matched gap.
  5. Aggregate by party (mean across covered topics) — this is the honest
     per-party stated-vs-revealed gap.

Outputs:
  data/processed/manifesto_topic_assignment.parquet
  data/processed/stated_vs_revealed_topic.parquet
  figures/web/26_topic_matched_gap.png  — per-party + per-topic
  figures/web/27_party_topic_gap_matrix.png  — heatmap
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
BG = "#FAFAFA"
INK = "#222222"

MIN_SENTENCES_PER_CELL = 5     # need this many manifesto sentences to compare
MIN_TOPIC_SIMILARITY = 0.30    # below this, sentence is "off-topic" — discard


def compute_topic_centroids() -> tuple[np.ndarray, list[int]]:
    """Centroid per topic from recommendation embeddings."""
    npz = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    X = npz["X"]  # (n_events, 768)
    ids = list(npz["votering_id"])
    topics = pd.read_parquet(IN / "topics.parquet")
    id_to_topic = dict(zip(topics["votering_id"], topics["topic_id"]))

    topic_to_indices: dict[int, list[int]] = {}
    for i, vid in enumerate(ids):
        t = id_to_topic.get(vid)
        if t is None:
            continue
        topic_to_indices.setdefault(int(t), []).append(i)

    topic_ids = sorted(topic_to_indices.keys())
    centroids = np.stack([X[topic_to_indices[t]].mean(axis=0) for t in topic_ids])
    # Normalise for cosine.
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids, topic_ids


def assign_manifesto_to_topics() -> pd.DataFrame:
    """For each manifesto sentence, embed and assign to nearest topic."""
    from sentence_transformers import SentenceTransformer

    print("  loading SBERT ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

    print("  embedding manifesto sentences ...")
    ann = pd.read_parquet(IN / "manifesto_sentences.parquet")
    # Drop "other" sentences — they're noise for this comparison.
    sub = ann[ann["element"] != "other"].copy()
    sentences = sub["sentence"].tolist()
    embs = model.encode(sentences, batch_size=32, show_progress_bar=False,
                         normalize_embeddings=True).astype(np.float32)
    print(f"  embedded {len(embs)} substantive manifesto sentences")

    centroids, topic_ids = compute_topic_centroids()
    sims = embs @ centroids.T  # (n_sentences, n_topics)
    best_topic_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(embs)), best_topic_idx]
    assigned = np.array([topic_ids[i] for i in best_topic_idx])

    sub = sub.reset_index(drop=True)
    sub["topic_id"] = assigned
    sub["topic_sim"] = best_sim
    # Stash the embedding for downstream aggregation. Use parquet-friendly list.
    sub["embedding"] = [e.tolist() for e in embs]
    return sub


def party_topic_vectors(assigned: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """Mean manifesto sentence vector per (party, topic) cell."""
    out = {}
    for (party, topic), g in assigned.groupby(["party", "topic_id"]):
        # Filter to sentences with reasonable on-topic-ness.
        g = g[g["topic_sim"] >= MIN_TOPIC_SIMILARITY]
        if len(g) < MIN_SENTENCES_PER_CELL:
            continue
        embs = np.array(g["embedding"].tolist())
        vec = embs.mean(axis=0)
        out[(party.upper(), int(topic))] = vec / np.linalg.norm(vec)
    return out


def load_reservation_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Per (party, topic) reservation vector, with fallback to speech."""
    rv = np.load(IN / "reservation_vectors.npz", allow_pickle=True)
    sv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    res_V, res_N = rv["V"], rv["N"]
    sp_V, sp_N = sv["V"], sv["N"]
    V = np.where(res_N[..., None] > 0, res_V, sp_V).astype(np.float32)
    N = np.maximum(res_N, sp_N)
    return V, N


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main() -> None:
    print("== assigning manifesto sentences to topics ==")
    assigned = assign_manifesto_to_topics()
    # Save (without embedding column for size).
    assigned.drop(columns=["embedding"]).to_parquet(
        IN / "manifesto_topic_assignment.parquet", index=False)
    print(f"  saved manifesto_topic_assignment.parquet")
    print(f"  topic coverage: {assigned.groupby('topic_id').size().describe().round(1).to_dict()}")

    print("\n== building per (party, topic) manifesto vectors ==")
    manifesto_cells = party_topic_vectors(assigned)
    print(f"  cells covered: {len(manifesto_cells)}")
    print("  per party (rows with ≥{} on-topic sentences):".format(MIN_SENTENCES_PER_CELL))
    per_party = {}
    for (p, t), _ in manifesto_cells.items():
        per_party[p] = per_party.get(p, 0) + 1
    for p in PARTIES:
        print(f"    {p}: {per_party.get(p, 0)} topics with manifesto data")

    print("\n== computing topic-matched stated-vs-revealed gap ==")
    res_V, res_N = load_reservation_vectors()
    rows = []
    for (party_u, topic_id), manifesto_vec in manifesto_cells.items():
        i = PARTIES.index(party_u)
        if res_N[i, topic_id] == 0:
            continue
        reservation_vec = res_V[i, topic_id]
        sim = cosine(manifesto_vec, reservation_vec)
        rows.append({
            "party": party_u,
            "topic_id": int(topic_id),
            "n_manifesto_sentences": int(
                ((assigned["party"] == party_u.lower()) &
                 (assigned["topic_id"] == topic_id)).sum()),
            "stated_revealed_cos": sim,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(IN / "stated_vs_revealed_topic.parquet", index=False)
    print(f"  saved stated_vs_revealed_topic.parquet ({len(df)} cells)")

    # === Per-party aggregate ===
    print("\n== per-party average gap (across covered topics) ==")
    party_agg = df.groupby("party")["stated_revealed_cos"].agg(["mean", "std", "count"]).reset_index()
    party_agg = party_agg.set_index("party").reindex(PARTIES).reset_index()
    party_agg = party_agg.dropna(subset=["mean"])
    print(party_agg.round(3).to_string(index=False))

    # === FIGURE: per-party topic-matched gap ===
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
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
    ax.set_xticks(x); ax.set_xticklabels(parties_in_data, fontsize=12, fontweight="bold")
    ax.set_ylabel("Topic-matched stated-vs-revealed cosine")
    ax.set_ylim(0, 1)
    ax.set_title("Stated vs revealed — topic-matched\n"
                 "For each party, average within-topic cosine between manifesto "
                 "text and reservation text. Error bars are 1σ across topics.",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "web" / "26_topic_matched_gap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, ROOT / "site" / "public" / "figures" / out.name)
    print(f"\n  wrote {out.name}")

    # === Per-topic average — which topics show biggest divergence ===
    topic_meta = pd.read_parquet(IN / "topic_meta.parquet").set_index("topic_id")
    topic_agg = df.groupby("topic_id")["stated_revealed_cos"].agg(["mean", "count"]).reset_index()
    topic_agg = topic_agg[topic_agg["count"] >= 3]  # need at least 3 parties for a topic
    topic_agg["label"] = topic_agg["topic_id"].map(
        lambda t: topic_meta.loc[t, "label_terms"].split(",")[0].strip() if t in topic_meta.index else f"topic {t}")
    topic_agg = topic_agg.sort_values("mean")
    print("\n== per-topic average gap (lowest = biggest stated-vs-revealed divergence) ==")
    print(topic_agg.head(10).round(3).to_string(index=False))

    # === FIGURE: party × topic heatmap ===
    pivot = df.pivot(index="party", columns="topic_id", values="stated_revealed_cos")
    pivot = pivot.reindex(index=[p for p in PARTIES if p in pivot.index])
    # Order topics by mean gap.
    topic_order = topic_agg.sort_values("mean")["topic_id"].tolist()
    topic_order = [t for t in topic_order if t in pivot.columns]
    pivot = pivot[topic_order]
    topic_labels = [
        topic_meta.loc[t, "label_terms"].split(",")[0].strip()[:25]
        if t in topic_meta.index else f"t{t}"
        for t in topic_order
    ]
    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0.2, vmax=0.8, aspect="auto")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(topic_labels))); ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if pd.notna(v):
                color = "#222" if 0.4 < v < 0.65 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Per-topic stated-vs-revealed cosine\n"
                 "Red = manifesto and reservation language diverge; green = consistent. "
                 "Topics sorted by mean across parties (most divergent at left).",
                 loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label="cosine")
    fig.tight_layout()
    out = FIG / "web" / "27_party_topic_gap_matrix.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, ROOT / "site" / "public" / "figures" / out.name)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
