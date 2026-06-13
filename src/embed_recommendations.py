"""Embed recommendation text per vote event with KBLab Swedish SBERT.

Output: data/processed/embed_recommendations.npz
  X            : (n_events, 768) float32
  votering_id  : (n_events,) object
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


def main() -> None:
    texts = pd.read_parquet(IN / "vote_event_texts.parquet")
    events = pd.read_parquet(IN / "vote_events.parquet")
    # Keep only rows that link to actual chamber votes; deduplicate to one per votering_id.
    voted_ids = events["votering_id"].str.upper().tolist()
    texts = texts[texts["votering_id"].isin(voted_ids)].drop_duplicates("votering_id")
    print(f"events to embed: {len(texts)}")

    # Compose input: betänkande title gives policy context, rubrik gives the
    # specific point. We omit forslag_text because it's dominated by procedural
    # boilerplate ("Riksdagen avslår motion ...") that clusters across topics.
    inputs = (texts["bet_titel"].fillna("") + ". " +
              texts["rubrik"].fillna("")).str.slice(0, 2000).tolist()

    print(f"loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  dim = {model.get_embedding_dimension()}")

    X = model.encode(inputs, batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True).astype(np.float32)
    print(f"embedded: shape {X.shape}")

    np.savez_compressed(
        OUT / "embed_recommendations.npz",
        X=X,
        votering_id=np.array(texts["votering_id"].to_numpy(), dtype=object),
    )
    print(f"wrote {OUT}/embed_recommendations.npz")


if __name__ == "__main__":
    main()
