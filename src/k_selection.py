"""K-selection sweep for unified topic clustering on the 2018-22 + 2022-26 union.

Reads existing decision-point embeddings from both mandates, concatenates,
runs KMeans for K in a small range, and reports silhouette, sizes and the
top TF-IDF terms per cluster at the recommended K.

Output:
  data/processed/k_selection_scores.csv
  Logged terminal output with per-K interpretability summary.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

K_VALUES = [16, 20, 24, 28, 32, 36]

# Same stopword list as topics.py
SWEDISH_STOP = """av i och en ett att som på för är till med den det de inte vi
har eller kan ska om har kommer skulle vid utan över under från då också efter
mot där så även mellan därför genom samt enligt blir när alltid något något
någon några måste mycket riksdagen avslår bifaller delvis motion motionerna
yrkande yrkandena förslag förslaget regeringens punkt regeringen m.m mm
övriga övrigt frågor fl fler fler.
""".split()


def load_union():
    """Concatenate decision-point embeddings + label texts from both mandates."""
    # 2022-26 (existing)
    npz_a = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    X_a = npz_a["X"]
    ids_a = list(npz_a["votering_id"])
    texts_a = pd.read_parquet(IN / "vote_event_texts.parquet")
    texts_a = texts_a.drop_duplicates("votering_id").set_index("votering_id")

    # 2018-22 (new)
    npz_b = np.load(IN / "embed_recommendations_18_22.npz", allow_pickle=True)
    X_b = npz_b["X"]
    ids_b = list(npz_b["votering_id"])
    texts_b = pd.read_parquet(IN / "vote_event_texts_18_22.parquet")
    texts_b = texts_b.drop_duplicates("votering_id").set_index("votering_id")

    mandate = ["2022-26"] * len(ids_a) + ["2018-22"] * len(ids_b)
    X = np.vstack([X_a, X_b]).astype(np.float32)
    ids = list(ids_a) + list(ids_b)

    # Build text labels (bet_titel + rubrik) for TF-IDF later
    rubrik = []
    bet_titel = []
    for vid in ids:
        # case-insensitive lookup
        row = None
        if vid in texts_a.index:
            row = texts_a.loc[vid]
        elif vid in texts_b.index:
            row = texts_b.loc[vid]
        if row is not None and isinstance(row, pd.Series):
            rubrik.append(row.get("rubrik", "") or "")
            bet_titel.append(row.get("bet_titel", "") or "")
        else:
            rubrik.append("")
            bet_titel.append("")

    return X, ids, mandate, rubrik, bet_titel


def topic_labels(M, vocab, labels, k):
    """Top-8 TF-IDF terms per cluster."""
    out = []
    for t in range(k):
        mask = labels == t
        if mask.sum() == 0:
            out.append((t, 0, "(empty)"))
            continue
        mean_vec = np.asarray(M[mask].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:8]
        terms = ", ".join(vocab[i] for i in top_idx)
        out.append((t, int(mask.sum()), terms))
    return out


def main() -> None:
    print("== loading union ==")
    X, ids, mandate, rubrik, bet_titel = load_union()
    print(f"  union: {X.shape}, ids: {len(ids)} ({mandate.count('2018-22')} from 2018-22, "
          f"{mandate.count('2022-26')} from 2022-26)")

    # Build the TF-IDF matrix once (for labels)
    YEAR_RX = re.compile(r"\b(19|20)\d{2}([/:]\d{2})?\b")
    NUM_RX = re.compile(r"\b\d+\b")
    docs = pd.Series([f"{t} {r}".lower() for t, r in zip(bet_titel, rubrik)])
    docs = docs.apply(lambda s: NUM_RX.sub(" ", YEAR_RX.sub(" ", s)))
    vec = TfidfVectorizer(stop_words=SWEDISH_STOP, ngram_range=(1, 2),
                          min_df=3, max_df=0.5, max_features=20_000,
                          sublinear_tf=True)
    M = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    # Silhouette on a sample to keep it fast
    sample_n = min(2000, len(X))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(X), sample_n, replace=False)

    rows = []
    print("\n== K sweep ==")
    print(f"  silhouette is computed on a {sample_n}-point sample (random_state=0)")
    for k in K_VALUES:
        km = KMeans(n_clusters=k, n_init=20, random_state=0)
        labels = km.fit_predict(X)
        sizes = np.bincount(labels, minlength=k)
        sil = silhouette_score(X[sample_idx], labels[sample_idx], metric="euclidean")
        print(f"\n  K={k:2d}  silhouette={sil:.4f}  "
              f"sizes min={sizes.min()} mean={sizes.mean():.0f} max={sizes.max()}")
        rows.append({"k": k, "silhouette": sil,
                     "size_min": int(sizes.min()), "size_mean": float(sizes.mean()),
                     "size_max": int(sizes.max())})
        # Print labels for K=28 and K=32 in particular (small/large)
        if k in {24, 28, 32}:
            print(f"  --- topic labels at K={k} ---")
            for t, sz, terms in topic_labels(M, vocab, labels, k):
                print(f"    [{t:>2}] n={sz:>4}  {terms[:140]}")

    pd.DataFrame(rows).to_csv(IN / "k_selection_scores.csv", index=False)
    print(f"\n  wrote {IN}/k_selection_scores.csv")

    # Stability check: for the existing K=28, how many topics are well-defined
    # by per-mandate share. A topic that is overwhelmingly populated by one
    # mandate may be a transient policy concern; one that is balanced is stable
    # across the mandate.
    print("\n== K=28 per-mandate balance ==")
    km28 = KMeans(n_clusters=28, n_init=20, random_state=0)
    labels28 = km28.fit_predict(X)
    mandate_arr = np.array(mandate)
    rows28 = []
    for t in range(28):
        mask = labels28 == t
        n = int(mask.sum())
        n_18 = int(((mandate_arr == "2018-22") & mask).sum())
        n_22 = int(((mandate_arr == "2022-26") & mask).sum())
        share_18 = n_18 / n if n else 0
        rows28.append({"topic": t, "n": n, "n_18_22": n_18, "n_22_26": n_22,
                       "share_18_22": share_18})
    print(pd.DataFrame(rows28).sort_values("share_18_22").to_string(index=False))


if __name__ == "__main__":
    main()
