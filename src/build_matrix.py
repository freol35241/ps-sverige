"""Parse bulk votering zips into a long DataFrame and an MP×vote matrix.

Outputs:
  data/processed/votes_long.parquet  — one row per (vote_event, MP)
  data/processed/votes_matrix.npz    — numeric matrix, MPs in row order, votes in column order
  data/processed/mps.parquet         — MP metadata (intressent_id, party, name, ...)
  data/processed/vote_events.parquet — vote-event metadata (votering_id, beteckning, datum, organ, ...)
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

RIKSMOTEN = ["202223", "202324", "202425", "202526"]

# Numeric encoding for embedding. Avstår = 0 keeps "present but neutral" distinct
# from Frånvarande which is missing and gets imputed later.
ROST_CODE = {"Ja": 1.0, "Nej": -1.0, "Avstår": 0.0, "Frånvarande": np.nan}

# Fields we keep per row of the long table.
KEEP = ("votering_id", "rm", "beteckning", "punkt", "datum",
        "avser", "votering",
        "intressent_id", "namn", "fornamn", "efternamn",
        "parti", "valkrets", "kon", "fodd", "rost")


def _organ_from_beteckning(bet: str) -> str:
    """Extract committee code (e.g., 'AU' from 'AU10', 'FiU' from 'FiU23')."""
    s = ""
    for ch in bet or "":
        if ch.isalpha():
            s += ch
        else:
            break
    return s


def iter_votes():
    """Yield (riksmote, file_payload_dict) for every vote event in every zip."""
    for rm in RIKSMOTEN:
        zpath = RAW / f"votering-{rm}.json.zip"
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                with z.open(name) as f:
                    yield rm, json.load(f)


def build_long() -> pd.DataFrame:
    rows = []
    for _, payload in iter_votes():
        votering = payload["dokvotering"]["votering"]
        # Some events may have a single dict instead of a list; normalise.
        if isinstance(votering, dict):
            votering = [votering]
        for v in votering:
            rows.append({k: v.get(k) for k in KEEP})
    df = pd.DataFrame(rows)
    df["organ"] = df["beteckning"].map(_organ_from_beteckning)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df


def build_matrix(long: pd.DataFrame):
    """Return (matrix, mp_index, vote_index) where matrix is (n_mps, n_votes)."""
    # Use first-seen order for stability.
    mp_order = (long.drop_duplicates("intressent_id")
                    .sort_values("intressent_id")["intressent_id"].tolist())
    vote_order = (long.drop_duplicates("votering_id")
                      .sort_values(["datum", "votering_id"])["votering_id"].tolist())
    mp_idx = {m: i for i, m in enumerate(mp_order)}
    vote_idx = {v: i for i, v in enumerate(vote_order)}

    n_mps, n_votes = len(mp_order), len(vote_order)
    M = np.full((n_mps, n_votes), np.nan, dtype=np.float32)
    rost_arr = long["rost"].map(ROST_CODE).to_numpy(dtype=np.float32)
    mp_arr = long["intressent_id"].map(mp_idx).to_numpy()
    v_arr = long["votering_id"].map(vote_idx).to_numpy()
    M[mp_arr, v_arr] = rost_arr  # last-write-wins; duplicates rare
    return M, mp_order, vote_order


def mp_metadata(long: pd.DataFrame) -> pd.DataFrame:
    """One row per MP. Party = modal party across the mandate; latest_party = most recent."""
    last = long.sort_values("datum").drop_duplicates("intressent_id", keep="last")
    modal_party = (long.groupby("intressent_id")["parti"]
                       .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
                       .rename("party_modal"))
    n_votes = long.groupby("intressent_id").size().rename("n_vote_events")
    out = (last[["intressent_id", "namn", "fornamn", "efternamn",
                 "parti", "valkrets", "kon", "fodd"]]
           .rename(columns={"parti": "party_latest"})
           .merge(modal_party, on="intressent_id")
           .merge(n_votes, on="intressent_id"))
    out["fodd"] = pd.to_numeric(out["fodd"], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)


def vote_event_metadata(long: pd.DataFrame) -> pd.DataFrame:
    return (long.drop_duplicates("votering_id")
                [["votering_id", "rm", "beteckning", "organ",
                  "punkt", "datum", "avser", "votering"]]
                .sort_values("datum")
                .reset_index(drop=True))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("parsing zips ...")
    long = build_long()
    print(f"  long table: {len(long):,} rows, {long['intressent_id'].nunique()} MPs, "
          f"{long['votering_id'].nunique()} vote events")

    print("vote response distribution:")
    for k, n in long["rost"].value_counts().items():
        print(f"  {k:14s} {n:>9,} ({n/len(long):.1%})")

    print("vote 'avser' distribution:")
    for k, n in long["avser"].value_counts().items():
        print(f"  {k:14s} {n:>9,}")
    print("vote 'votering' (huvud=main, prov=preliminary) distribution:")
    for k, n in long["votering"].value_counts().items():
        print(f"  {k:14s} {n:>9,}")

    mps = mp_metadata(long)
    events = vote_event_metadata(long)
    print(f"\n  MPs: {len(mps)}  events: {len(events)}")
    print("party seat counts (modal party, MPs with >=10 vote events):")
    counted = mps[mps["n_vote_events"] >= 10]["party_modal"].value_counts()
    print(counted.to_string())

    print("\nbuilding matrix ...")
    M, mp_order, vote_order = build_matrix(long)
    print(f"  shape {M.shape}, non-nan {np.isfinite(M).mean():.1%}")

    long.to_parquet(OUT / "votes_long.parquet", index=False)
    mps.to_parquet(OUT / "mps.parquet", index=False)
    events.to_parquet(OUT / "vote_events.parquet", index=False)
    np.savez_compressed(OUT / "votes_matrix.npz",
                        M=M,
                        mp_order=np.array(mp_order, dtype=object),
                        vote_order=np.array(vote_order, dtype=object))
    print(f"  wrote {OUT}/")


if __name__ == "__main__":
    main()
