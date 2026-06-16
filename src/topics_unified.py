"""Unified topic clustering on the 2018-22 + 2022-26 union.

Fits KMeans at K=28 on the union of decision-point embeddings from both
mandates, produces canonical topic assignments and labels for the
longitudinal analysis. Does not touch the existing single-mandate
topics.parquet or topic_meta.parquet; outputs go to *_unified
variants so the 2022-26-only pipeline keeps working.

K=28 was selected after a sweep across K in {16, 20, 24, 28, 32, 36};
silhouette gain above K=28 is negligible and smaller clusters become
fragile. See src/k_selection.py for the sweep.

Outputs:
  data/processed/topics_unified.parquet
     votering_id, mandate, topic_id_unified, dist_to_centroid
  data/processed/topic_meta_unified.parquet
     topic_id, size, label_terms, primary_organ, primary_organ_share,
     examples, n_18_22, n_22_26, share_18_22
  data/processed/topic_stability.parquet
     Mapping from old single-mandate topic_id (from topics.parquet) to
     new unified topic_id, with overlap fraction. Lets us check which
     old topics were preserved, merged or split.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

K_TOPICS = 28
RANDOM_STATE = 0

SWEDISH_STOP = """av i och en ett att som på för är till med den det de inte vi
har eller kan ska om har kommer skulle vid utan över under från då också efter
mot där så även mellan därför genom samt enligt blir när alltid något något
någon några måste mycket riksdagen avslår bifaller delvis motion motionerna
yrkande yrkandena förslag förslaget regeringens punkt regeringen m.m mm
övriga övrigt frågor fl fler fler.
""".split()


def load_union():
    """Load embeddings + texts for the 2018-22 + 2022-26 union."""
    npz_a = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    X_a = npz_a["X"]
    ids_a = list(npz_a["votering_id"])
    texts_a = pd.read_parquet(IN / "vote_event_texts.parquet")

    npz_b = np.load(IN / "embed_recommendations_18_22.npz", allow_pickle=True)
    X_b = npz_b["X"]
    ids_b = list(npz_b["votering_id"])
    texts_b = pd.read_parquet(IN / "vote_event_texts_18_22.parquet")

    mandate = ["2022-26"] * len(ids_a) + ["2018-22"] * len(ids_b)
    X = np.vstack([X_a, X_b]).astype(np.float32)
    ids = list(ids_a) + list(ids_b)
    texts = pd.concat([texts_a, texts_b], ignore_index=True)
    texts = texts.drop_duplicates("votering_id").set_index("votering_id")
    return X, ids, mandate, texts


def cluster_labels(M, vocab, labels, k):
    """Top-8 TF-IDF terms per cluster."""
    rows = []
    for t in range(k):
        mask = labels == t
        if mask.sum() == 0:
            rows.append((t, 0, "(empty)", "?", 0.0, ""))
            continue
        mean_vec = np.asarray(M[mask].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:8]
        terms = ", ".join(vocab[i] for i in top_idx)
        rows.append((t, int(mask.sum()), terms))
    return rows


def main() -> None:
    print("== loading union ==")
    X, ids, mandate, texts = load_union()
    print(f"  union: {X.shape}, ids: {len(ids)}")
    print(f"  2018-22: {mandate.count('2018-22')}, 2022-26: {mandate.count('2022-26')}")

    print(f"\n== fitting KMeans (K={K_TOPICS}, random_state={RANDOM_STATE}) ==")
    km = KMeans(n_clusters=K_TOPICS, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    distances = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    sizes = np.bincount(labels, minlength=K_TOPICS)
    print(f"  cluster sizes: min={sizes.min()}, mean={sizes.mean():.0f}, max={sizes.max()}")

    # ---- topic assignment frame ----
    topics = pd.DataFrame({
        "votering_id": ids,
        "mandate": mandate,
        "topic_id_unified": labels.astype(np.int32),
        "dist_to_centroid": distances.astype(np.float32),
    })
    topics.to_parquet(IN / "topics_unified.parquet", index=False)
    print(f"  wrote topics_unified.parquet")

    # ---- topic labels via TF-IDF on bet_titel + rubrik ----
    rubrik = [texts.loc[v, "rubrik"] if v in texts.index else "" for v in ids]
    bet_titel = [texts.loc[v, "bet_titel"] if v in texts.index else "" for v in ids]
    organ = [texts.loc[v, "organ"] if v in texts.index else "" for v in ids]
    rubrik = [r if isinstance(r, str) else "" for r in rubrik]
    bet_titel = [b if isinstance(b, str) else "" for b in bet_titel]
    organ = [o if isinstance(o, str) else "" for o in organ]

    YEAR_RX = re.compile(r"\b(19|20)\d{2}([/:]\d{2})?\b")
    NUM_RX = re.compile(r"\b\d+\b")
    docs = pd.Series([f"{t} {r}".lower() for t, r in zip(bet_titel, rubrik)])
    docs = docs.apply(lambda s: NUM_RX.sub(" ", YEAR_RX.sub(" ", s)))
    vec = TfidfVectorizer(stop_words=SWEDISH_STOP, ngram_range=(1, 2),
                          min_df=3, max_df=0.5, max_features=20_000,
                          sublinear_tf=True)
    M = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    mandate_arr = np.array(mandate)
    organ_arr = np.array(organ)
    rubrik_arr = np.array(rubrik)
    distances_arr = distances

    meta_rows = []
    for t in range(K_TOPICS):
        mask = labels == t
        n = int(mask.sum())
        if n == 0:
            continue
        mean_vec = np.asarray(M[mask].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:8]
        terms = ", ".join(vocab[i] for i in top_idx)
        # primary organ within cluster
        organ_counts = pd.Series(organ_arr[mask]).value_counts(normalize=True)
        primary_organ = organ_counts.index[0] if len(organ_counts) else ""
        primary_organ_share = float(organ_counts.iloc[0]) if len(organ_counts) else 0.0
        # 3 closest-to-centroid example rubriks
        cluster_idx = np.where(mask)[0]
        cluster_dist = distances_arr[cluster_idx]
        ex_idx = cluster_idx[np.argsort(cluster_dist)[:3]]
        examples = " | ".join(rubrik_arr[i] for i in ex_idx)
        # per-mandate balance
        n_18 = int(((mandate_arr == "2018-22") & mask).sum())
        n_22 = int(((mandate_arr == "2022-26") & mask).sum())
        meta_rows.append({
            "topic_id": t,
            "size": n,
            "label_terms": terms,
            "primary_organ": primary_organ,
            "primary_organ_share": primary_organ_share,
            "examples": examples,
            "n_18_22": n_18,
            "n_22_26": n_22,
            "share_18_22": n_18 / n,
        })
    meta = pd.DataFrame(meta_rows).sort_values("size", ascending=False).reset_index(drop=True)
    meta.to_parquet(IN / "topic_meta_unified.parquet", index=False)
    print(f"  wrote topic_meta_unified.parquet")

    # ---- topic stability vs old single-mandate clustering ----
    print("\n== topic stability: old vs new ==")
    old_topics = pd.read_parquet(IN / "topics.parquet")
    old_topics["votering_id"] = old_topics["votering_id"].astype(str)
    topics_str = topics.copy()
    topics_str["votering_id"] = topics_str["votering_id"].astype(str)
    merged = old_topics.merge(
        topics_str[topics_str["mandate"] == "2022-26"][["votering_id", "topic_id_unified"]],
        on="votering_id", how="inner",
    )
    print(f"  matched {len(merged):,} of {len(old_topics):,} old-topic vote events")
    # For each old topic, find the majority new topic
    stability_rows = []
    for old_t in sorted(merged["topic_id"].unique()):
        sub = merged[merged["topic_id"] == old_t]
        if len(sub) == 0:
            continue
        new_counts = sub["topic_id_unified"].value_counts()
        majority_new = int(new_counts.index[0])
        majority_share = float(new_counts.iloc[0] / len(sub))
        stability_rows.append({
            "old_topic_id": int(old_t),
            "old_size": int(len(sub)),
            "new_topic_id": majority_new,
            "majority_share": majority_share,
            "n_new_topics_covered": int(len(new_counts)),
        })
    stability = pd.DataFrame(stability_rows).sort_values("majority_share")
    stability.to_parquet(IN / "topic_stability.parquet", index=False)
    print(f"  wrote topic_stability.parquet")
    print(f"\n  stability stats:")
    print(f"    mean majority share: {stability['majority_share'].mean():.3f}")
    print(f"    median majority share: {stability['majority_share'].median():.3f}")
    print(f"    old topics that landed >=80% in one new topic: "
          f"{(stability['majority_share'] >= 0.8).sum()} / {len(stability)}")
    print(f"    old topics that landed >=50% in one new topic: "
          f"{(stability['majority_share'] >= 0.5).sum()} / {len(stability)}")

    # Print the topic table
    print("\n== unified topics (K=28) ==")
    for _, r in meta.iterrows():
        print(f"\n[{r['topic_id']:>2}] size={r['size']:>3} "
              f"({r['primary_organ']}, {r['primary_organ_share']:.0%}) "
              f"  share 18-22: {r['share_18_22']:.2f}")
        print(f"     terms: {r['label_terms']}")
        print(f"     ex:    {r['examples'][:200]}")


if __name__ == "__main__":
    main()
