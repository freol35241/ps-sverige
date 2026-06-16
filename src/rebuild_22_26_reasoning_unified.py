"""Rebuild 2022-26 speech-based reasoning vectors against unified topic IDs.

Mirrors src/reasoning_embed.py but uses topics_unified.parquet (filtered to
2022-26) instead of topics.parquet, so the resulting vectors can stand in
for the speech-based fallback in vote-vs-reasoning when reservation cells
are sparse (cabinet parties M, KD, L).

Output: data/processed/reasoning_vectors_22_26_unified.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

MODEL_NAME = "KBLab/sentence-bert-swedish-cased"
PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
N_PER_CELL = 12
MIN_WORDS_PER_SPEECH = 50
MAX_WORDS_PER_SPEECH = 1500
WORDS_TO_EMBED = 400
N_TOPICS = 28


def main() -> None:
    speeches = pd.read_parquet(IN / "speeches.parquet")
    texts = pd.read_parquet(IN / "vote_event_texts.parquet").drop_duplicates("votering_id")
    topics = pd.read_parquet(IN / "topics_unified.parquet")
    topics = topics[topics["mandate"] == "2022-26"][["votering_id", "topic_id_unified"]]
    print(f"loaded {len(speeches):,} speeches, {len(texts):,} vote_event_texts, "
          f"{len(topics):,} 2022-26 topic assignments (unified)")

    e2t = topics.merge(texts[["votering_id", "dok_id"]], on="votering_id")
    e2t["dok_id"] = e2t["dok_id"].str.upper()
    bet_topic = (e2t.groupby("dok_id")["topic_id_unified"]
                    .agg(lambda s: s.mode().iat[0])
                    .reset_index()
                    .rename(columns={"topic_id_unified": "betänkande_topic"}))
    print(f"betänkanden with unified topic assignment: {len(bet_topic):,}")

    sp = speeches[speeches["parti"].isin(PARTIES)].copy()
    sp = sp.assign(rel_upper=sp["rel_dok_id"].astype(str).str.upper())
    sp_linked = sp.merge(bet_topic, left_on="rel_upper", right_on="dok_id", how="inner")
    print(f"linked speeches: {len(sp_linked):,}")
    sp_linked = sp_linked[sp_linked["replik"].fillna("N").str.upper() != "Y"]
    sp_linked = sp_linked[(sp_linked["n_words"] >= MIN_WORDS_PER_SPEECH) &
                          (sp_linked["n_words"] <= MAX_WORDS_PER_SPEECH)]
    print(f"after filter: {len(sp_linked):,}")

    sp_linked = sp_linked.sort_values("n_words", ascending=False)
    picked = (sp_linked.groupby(["parti", "betänkande_topic"], group_keys=False)
                       .head(N_PER_CELL)).copy()

    picked["text_trunc"] = (picked["text"].str.split().str[:WORDS_TO_EMBED]
                            .str.join(" "))
    print(f"picked for embedding: {len(picked):,}")
    coverage = picked.groupby(["parti", "betänkande_topic"]).size().unstack(fill_value=0)
    coverage = coverage.reindex(PARTIES)
    cells_filled = int((coverage > 0).sum().sum())
    print(f"cells filled: {cells_filled} of {len(PARTIES) * N_TOPICS}")

    print(f"\nloading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"embedding {len(picked):,} speeches ...")
    embs = model.encode(picked["text_trunc"].tolist(),
                        batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    picked = picked.assign(_idx=np.arange(len(picked)))

    V = np.zeros((len(PARTIES), N_TOPICS, embs.shape[1]), dtype=np.float32)
    N = np.zeros((len(PARTIES), N_TOPICS), dtype=np.int32)
    for i, party in enumerate(PARTIES):
        for t in range(N_TOPICS):
            mask = ((picked["parti"] == party) &
                    (picked["betänkande_topic"] == t)).to_numpy()
            if mask.sum() == 0:
                continue
            V[i, t] = embs[picked["_idx"].to_numpy()[mask]].mean(axis=0)
            N[i, t] = mask.sum()
    print(f"\nfilled vector cells: {(N > 0).sum()} of {N.size}")

    np.savez_compressed(IN / "reasoning_vectors_22_26_unified.npz",
                        V=V, N=N, parties=np.array(PARTIES, dtype=object))
    print(f"wrote reasoning_vectors_22_26_unified.npz")


if __name__ == "__main__":
    main()
