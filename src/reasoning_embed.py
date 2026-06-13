"""Build per-(party, topic) reasoning vectors from debate speeches.

Strategy:
1. Link each speech to a betänkande via rel_dok_id.
2. Link each betänkande to the dominant topic of its vote events.
3. Drop very short / replik speeches (low signal).
4. Per (party, topic): pick up to N_PER_CELL representative speeches (longest
   non-replik, by party MPs) → embed each → mean → reasoning vector.

Output: data/processed/reasoning_vectors.npz with V[party, topic, 768] and
        data/processed/reasoning_speeches.parquet (the picked excerpts, for
        the voter-facing tool).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

MODEL_NAME = "KBLab/sentence-bert-swedish-cased"
PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
N_PER_CELL = 12   # speeches per (party, topic) cell
MIN_WORDS_PER_SPEECH = 50
MAX_WORDS_PER_SPEECH = 1500
WORDS_TO_EMBED = 400  # truncate to fit SBERT's window cleanly


def main() -> None:
    speeches = pd.read_parquet(IN / "speeches.parquet")
    texts = pd.read_parquet(IN / "vote_event_texts.parquet").drop_duplicates("votering_id")
    topics = pd.read_parquet(IN / "topics.parquet")
    print(f"loaded {len(speeches):,} speeches, {len(texts):,} vote_event_texts")

    # Build dok_id (betänkande) → dominant topic_id. Case-normalize both sides.
    e2t = topics.merge(texts[["votering_id", "dok_id"]], on="votering_id")
    e2t["dok_id"] = e2t["dok_id"].str.upper()
    bet_topic = (e2t.groupby("dok_id")["topic_id"]
                    .agg(lambda s: s.mode().iat[0])
                    .reset_index().rename(columns={"topic_id": "betänkande_topic"}))
    print(f"betänkanden with topic assignment: {len(bet_topic):,}")

    # Filter party MPs, link to betänkanden first (most filtering happens at the join),
    # THEN word-count filter (the strict word filter on the unjoined set throws away
    # too much because the long substantive debate speeches are 500-1500 words).
    sp = speeches[speeches["parti"].isin(PARTIES)]
    sp = sp.assign(rel_upper=sp["rel_dok_id"].astype(str).str.upper())
    sp_linked = sp.merge(bet_topic, left_on="rel_upper", right_on="dok_id", how="inner")
    print(f"linked speeches: {len(sp_linked):,}")
    sp_linked = sp_linked[sp_linked["replik"].fillna("N").str.upper() != "Y"]
    sp_linked = sp_linked[(sp_linked["n_words"] >= MIN_WORDS_PER_SPEECH) &
                          (sp_linked["n_words"] <= MAX_WORDS_PER_SPEECH)]
    print(f"after replik+word filter: {len(sp_linked):,}")

    # Pick up to N_PER_CELL per (party, betänkande_topic), ordered by length (longest first).
    sp_linked = sp_linked.sort_values("n_words", ascending=False)
    picked = (sp_linked.groupby(["parti", "betänkande_topic"], group_keys=False)
                       .head(N_PER_CELL)).copy()

    # Truncate to WORDS_TO_EMBED words so SBERT doesn't silently chop long speeches.
    picked["text_trunc"] = picked["text"].str.split().str[:WORDS_TO_EMBED].str.join(" ")
    print(f"speeches picked for embedding: {len(picked):,}")
    print("\ncells with full N_PER_CELL coverage:")
    coverage = (picked.groupby(["parti", "betänkande_topic"]).size().unstack(fill_value=0))
    print(f"  cells filled: {(coverage > 0).sum().sum()} of {len(PARTIES) * 28}")
    print(f"  mean speeches per cell: {coverage.values[coverage.values > 0].mean():.1f}")

    # Embed.
    print("\nloading model ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"embedding {len(picked):,} speeches ...")
    embs = model.encode(picked["text_trunc"].tolist(),
                        batch_size=32, show_progress_bar=True,
                        normalize_embeddings=True).astype(np.float32)

    picked = picked.assign(_idx=np.arange(len(picked)))

    # Aggregate to (party, topic) mean vector.
    V = np.zeros((len(PARTIES), 28, embs.shape[1]), dtype=np.float32)
    N = np.zeros((len(PARTIES), 28), dtype=np.int32)
    for i, party in enumerate(PARTIES):
        for t in range(28):
            mask = ((picked["parti"] == party) &
                    (picked["betänkande_topic"] == t)).to_numpy()
            if mask.sum() == 0:
                continue
            V[i, t] = embs[picked["_idx"].to_numpy()[mask]].mean(axis=0)
            N[i, t] = mask.sum()
    print(f"\nfilled cells: {(N > 0).sum()} of {N.size}")

    np.savez_compressed(OUT / "reasoning_vectors.npz",
                        V=V, N=N,
                        parties=np.array(PARTIES, dtype=object))

    # Save the picked excerpts (without embedding to keep parquet small).
    picked[["anforande_id", "intressent_id", "talare", "parti",
            "rel_dok_id", "betänkande_topic", "dok_datum",
            "avsnittsrubrik", "n_words", "text"]].to_parquet(
        OUT / "reasoning_speeches.parquet", index=False)

    print(f"\nwrote {OUT}/reasoning_vectors.npz, reasoning_speeches.parquet")


if __name__ == "__main__":
    main()
