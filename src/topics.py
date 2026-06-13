"""Cluster vote-event recommendation embeddings into policy topics.

Outputs:
  data/processed/topics.parquet         votering_id → topic_id, distance
  data/processed/topic_meta.parquet     topic_id → label (top TF-IDF terms), size, organ mix
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

N_TOPICS = 28
SWEDISH_STOP = """av i och en ett att som på för är till med den det de inte vi
har eller kan ska om har kommer skulle vid utan över under från då också efter
mot där så även mellan därför genom samt enligt blir när alltid något något
någon några måste mycket riksdagen avslår bifaller delvis motion motionerna
yrkande yrkandena förslag förslaget regeringens punkt regeringen m.m mm
övriga övrigt frågor fl fler fler.
""".split()


def main() -> None:
    npz = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    X = npz["X"]
    ids = list(npz["votering_id"])
    print(f"loaded embeddings {X.shape}")

    # KMeans clustering. With normalised embeddings, euclidean ≈ angular.
    km = KMeans(n_clusters=N_TOPICS, n_init=20, random_state=0)
    labels = km.fit_predict(X)
    distances = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    print(f"clusters: {N_TOPICS}, sizes: {np.bincount(labels)}")

    topics = pd.DataFrame({
        "votering_id": ids,
        "topic_id": labels,
        "dist_to_centroid": distances,
    })
    topics.to_parquet(OUT / "topics.parquet", index=False)

    # Top TF-IDF terms per cluster as a topic label.
    texts = pd.read_parquet(IN / "vote_event_texts.parquet")
    texts = texts.drop_duplicates("votering_id").set_index("votering_id")
    topics_merged = topics.merge(
        texts[["rubrik", "forslag_text", "bet_titel", "organ", "beteckning"]].reset_index(),
        on="votering_id", how="left")

    # Use only rubrik + bet_titel for labels — those are clean policy titles.
    # forslag_text is dominated by motion-citation noise (MP names, dates).
    import re
    YEAR_RX = re.compile(r"\b(19|20)\d{2}([/:]\d{2})?\b")
    NUM_RX = re.compile(r"\b\d+\b")
    docs = (topics_merged["bet_titel"].fillna("") + " " +
            topics_merged["rubrik"].fillna("")).str.lower()
    docs = docs.apply(lambda s: NUM_RX.sub(" ", YEAR_RX.sub(" ", s)))

    vec = TfidfVectorizer(stop_words=SWEDISH_STOP, ngram_range=(1, 2),
                          min_df=3, max_df=0.5, max_features=20_000,
                          sublinear_tf=True)
    M = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    rows = []
    for t in range(N_TOPICS):
        mask = topics_merged["topic_id"].to_numpy() == t
        if mask.sum() == 0:
            continue
        # Cluster-mean TF-IDF
        mean_vec = np.asarray(M[mask].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:8]
        terms = ", ".join(vocab[i] for i in top_idx)
        # Top organ
        organ_mix = topics_merged.loc[mask, "organ"].value_counts(normalize=True)
        primary_organ = organ_mix.idxmax()
        primary_share = organ_mix.max()
        # Closest-to-centroid examples
        ex_idx = topics_merged.loc[mask].sort_values("dist_to_centroid").head(3)
        examples = " | ".join(ex_idx["rubrik"].tolist())
        rows.append({
            "topic_id": t,
            "size": int(mask.sum()),
            "label_terms": terms,
            "primary_organ": primary_organ,
            "primary_organ_share": float(primary_share),
            "examples": examples,
        })
    meta = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    meta.to_parquet(OUT / "topic_meta.parquet", index=False)

    print("\n=== Topics (sorted by size) ===")
    for _, r in meta.iterrows():
        print(f"\n[{r['topic_id']:>2}] size={r['size']:>3} "
              f"({r['primary_organ']}, {r['primary_organ_share']:.0%})")
        print(f"     terms: {r['label_terms']}")
        print(f"     ex:    {r['examples'][:200]}")


if __name__ == "__main__":
    main()
