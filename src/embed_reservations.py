"""Embed reservation reasoning text per (party, topic).

Strategy:
1. Link each reservation (dok_id, punkt) → vote event → topic_id.
2. Attribute each reservation to every party that signed it.
3. Per (party, topic), pick up to N representative reservations (longest).
4. Truncate to WORDS_TO_EMBED to fit SBERT cleanly. Embed each.
5. Average to a per-(party, topic) reservation vector.

Compare to the speech-based reasoning vectors from session 3 — reservations
are concentrated party-reasoning text, while speeches mix individual MP voice
with topic policy details. The reservation vectors should sharpen the WHY/HOW
signal.
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
N_PER_CELL = 20
WORDS_TO_EMBED = 400


def main() -> None:
    res = pd.read_parquet(IN / "reservations.parquet")
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates(["dok_id", "punkt"]))
    topics = pd.read_parquet(IN / "topics.parquet")
    print(f"loaded {len(res):,} reservations")

    # Link to vote events and topics.
    # Both sides use the same dok_id casing (parser preserved original).
    tex = tex.merge(topics[["votering_id", "topic_id"]],
                    on="votering_id", how="left")
    res_linked = res.merge(tex[["dok_id", "punkt", "topic_id"]],
                            on=["dok_id", "punkt"], how="left")
    print(f"reservations linked to topic: {res_linked['topic_id'].notna().sum():,} "
          f"({res_linked['topic_id'].notna().mean():.0%})")

    # Expand: one row per (reservation, signing party).
    rows = []
    for _, r in res_linked.iterrows():
        if pd.isna(r["topic_id"]):
            continue
        for p in str(r["partier"]).split(";"):
            if p in PARTIES:
                rows.append({
                    "party": p,
                    "topic_id": int(r["topic_id"]),
                    "dok_id": r["dok_id"],
                    "punkt": r["punkt"],
                    "rubrik": r["rubrik"],
                    "reasoning_text": r["reasoning_text"],
                    "n_words": r["n_words"],
                    "n_partier": r["n_partier"],
                })
    expanded = pd.DataFrame(rows)
    print(f"expanded to (party, reservation) rows: {len(expanded):,}")

    # Pick top N per (party, topic) by length.
    expanded = expanded.sort_values("n_words", ascending=False)
    picked = (expanded.groupby(["party", "topic_id"], group_keys=False)
                      .head(N_PER_CELL)).copy()
    print(f"picked for embedding: {len(picked):,}")
    coverage = picked.groupby(["party", "topic_id"]).size().unstack(fill_value=0)
    coverage = coverage.reindex(PARTIES)
    cells_filled = (coverage > 0).sum().sum()
    print(f"cells filled: {cells_filled} of {len(PARTIES) * 28}")
    print(f"mean reservations per filled cell: "
          f"{coverage.values[coverage.values > 0].mean():.1f}")

    # Truncate.
    picked["text_trunc"] = (picked["reasoning_text"]
                            .str.split().str[:WORDS_TO_EMBED].str.join(" "))

    # Embed.
    print("\nloading model ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"embedding {len(picked):,} reservations ...")
    embs = model.encode(picked["text_trunc"].tolist(),
                        batch_size=32, show_progress_bar=True,
                        normalize_embeddings=True).astype(np.float32)

    picked = picked.assign(_idx=np.arange(len(picked)))

    V = np.zeros((len(PARTIES), 28, embs.shape[1]), dtype=np.float32)
    N = np.zeros((len(PARTIES), 28), dtype=np.int32)
    for i, party in enumerate(PARTIES):
        for t in range(28):
            mask = ((picked["party"] == party) &
                    (picked["topic_id"] == t)).to_numpy()
            if mask.sum() == 0:
                continue
            V[i, t] = embs[picked["_idx"].to_numpy()[mask]].mean(axis=0)
            N[i, t] = mask.sum()
    print(f"\nfilled cells: {(N > 0).sum()} of {N.size}")

    np.savez_compressed(OUT / "reservation_vectors.npz",
                        V=V, N=N,
                        parties=np.array(PARTIES, dtype=object))
    picked[["party", "topic_id", "dok_id", "punkt", "n_partier",
            "rubrik", "reasoning_text", "n_words"]].to_parquet(
        OUT / "reservation_picks.parquet", index=False)
    print(f"\nwrote {OUT}/reservation_vectors.npz, reservation_picks.parquet")


if __name__ == "__main__":
    main()
