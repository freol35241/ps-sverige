"""Compute the party × topic position matrix.

For each vote event, per party we compute a stance:
   stance_p,v = (n_Ja - n_Nej) / (n_Ja + n_Nej + n_Avstår)    ∈ [-1, +1]
(Frånvarande excluded — that's "didn't vote", not a position.)

Per (party, topic): mean of stance over all events in that topic.

Output: data/processed/party_topic.parquet  (party × topic_id → score, n_events)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def per_event_stance(long: pd.DataFrame) -> pd.DataFrame:
    """Per (votering_id, parti) stance score in [-1, +1]."""
    counts = (long.groupby(["votering_id", "parti", "rost"])
                  .size().unstack("rost").fillna(0))
    n_ja = counts.get("Ja", pd.Series(0, index=counts.index)).astype(int)
    n_nej = counts.get("Nej", pd.Series(0, index=counts.index)).astype(int)
    n_av = counts.get("Avstår", pd.Series(0, index=counts.index)).astype(int)
    denom = (n_ja + n_nej + n_av).replace(0, np.nan)
    stance = (n_ja - n_nej) / denom
    df = stance.rename("stance").reset_index()
    df["votering_id_upper"] = df["votering_id"].str.upper()
    return df


def main() -> None:
    long = pd.read_parquet(IN / "votes_long.parquet")
    topics = pd.read_parquet(IN / "topics.parquet")
    meta = pd.read_parquet(IN / "topic_meta.parquet")

    stance = per_event_stance(long)
    print(f"per-event stance rows: {len(stance):,}")

    merged = stance.merge(topics,
                          left_on="votering_id_upper", right_on="votering_id",
                          how="inner", suffixes=("_v", "_t"))
    print(f"after join with topics: {len(merged):,}")

    # Per (party, topic) mean stance
    grid = (merged[merged["parti"].isin(PARTIES)]
            .groupby(["parti", "topic_id"])
            .agg(score=("stance", "mean"),
                 n_events=("votering_id_v", "nunique"))
            .reset_index())
    grid.to_parquet(OUT / "party_topic.parquet", index=False)

    # Pivot for inspection.
    pivot = grid.pivot(index="parti", columns="topic_id", values="score")
    pivot = pivot.loc[PARTIES]
    print("\nparty × topic stance matrix:")
    print(pivot.round(2).to_string())

    # Quick: for each topic, which party is most distinct from the median?
    print("\nMost-distinct party per topic (largest |score − median|):")
    for t in sorted(pivot.columns):
        col = pivot[t]
        med = col.median()
        distances = (col - med).abs()
        idx = distances.idxmax()
        label = meta.loc[meta["topic_id"] == t, "label_terms"].iloc[0][:70]
        print(f"  topic {t:>2d} ({label:<70s})  distinct={idx} ({col[idx]:+.2f} vs median {med:+.2f})")


if __name__ == "__main__":
    main()
