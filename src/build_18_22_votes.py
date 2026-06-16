"""Build the 2018-22 vote matrix and MP metadata.

Parallel to build_matrix.py, output to *_18_22 names so the existing
2022-26 vote matrix is preserved.

Outputs:
  data/processed/votes_long_18_22.parquet — one row per (vote_event, MP)
  data/processed/votes_matrix_18_22.npz — MP×vote numeric matrix
  data/processed/mps_18_22.parquet — MPs serving in 2018-22
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src"))
from build_matrix import KEEP, ROST_CODE, _organ_from_beteckning  # noqa: E402

RIKSMOTEN = ["201819", "201920", "202021", "202122"]


def iter_votes():
    for rm in RIKSMOTEN:
        zpath = RAW / f"votering-{rm}.json.zip"
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                with z.open(name) as f:
                    payload = json.load(f)
                # payload["dokvotering"]["votering"] is a list of vote rows
                votes = payload.get("dokvotering", {}).get("votering", [])
                if isinstance(votes, dict):
                    votes = [votes]
                for v in votes:
                    yield rm, v


def build_long() -> pd.DataFrame:
    rows = []
    for rm, v in iter_votes():
        row = {k: v.get(k) for k in KEEP}
        rows.append(row)
    df = pd.DataFrame(rows)
    df["organ"] = df["beteckning"].apply(_organ_from_beteckning)
    df["rost_code"] = df["rost"].map(ROST_CODE)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("== building long table ==")
    long = build_long()
    print(f"  long rows: {len(long):,}")
    print(f"  unique vote events: {long['votering_id'].nunique():,}")
    print(f"  unique MPs: {long['intressent_id'].nunique():,}")

    # Save the long table
    long.to_parquet(OUT / "votes_long_18_22.parquet", index=False)
    print(f"  wrote votes_long_18_22.parquet")

    # === MP metadata (matches existing mps.parquet schema) ===
    base_cols = ["intressent_id", "namn", "fornamn", "efternamn",
                 "valkrets", "kon", "fodd"]
    mp_base = (long.dropna(subset=["intressent_id"])
                    .drop_duplicates("intressent_id")[base_cols])
    # party_modal: most-common party across the mandate (handles defectors)
    party_modal = (long.dropna(subset=["intressent_id", "parti"])
                       .groupby("intressent_id")["parti"]
                       .agg(lambda s: s.mode().iat[0])
                       .rename("party_modal"))
    # party_latest: the party associated with each MP's most-recent vote
    sorted_long = long.dropna(subset=["intressent_id", "parti", "datum"]).sort_values("datum")
    party_latest = (sorted_long.drop_duplicates("intressent_id", keep="last")
                                [["intressent_id", "parti"]]
                                .set_index("intressent_id")["parti"]
                                .rename("party_latest"))
    n_vote_events = (long.dropna(subset=["intressent_id", "votering_id"])
                         .drop_duplicates(["intressent_id", "votering_id"])
                         .groupby("intressent_id").size()
                         .rename("n_vote_events"))
    mps = (mp_base.set_index("intressent_id")
                  .join(party_latest)
                  .join(party_modal)
                  .join(n_vote_events)
                  .reset_index())
    print(f"\n  unique MPs in long table: {len(mps):,}")
    print("  party_modal distribution:")
    print(mps["party_modal"].value_counts().to_string())
    mps.to_parquet(OUT / "mps_18_22.parquet", index=False)
    print(f"  wrote mps_18_22.parquet")

    # === Matrix ===
    # MPs × vote_events, encoded as rost_code (Yes=+1, No=-1, Avstår=0, Frånvarande=NaN)
    print("\n== building matrix ==")
    mp_order = mps["intressent_id"].tolist()
    mp_idx = {mp: i for i, mp in enumerate(mp_order)}
    vote_order = sorted(long["votering_id"].dropna().unique())
    event_idx = {e: i for i, e in enumerate(vote_order)}

    M = np.full((len(mp_order), len(vote_order)), np.nan, dtype=np.float32)
    # Vectorised assignment
    valid = long.dropna(subset=["intressent_id", "votering_id", "rost_code"])
    row_ix = valid["intressent_id"].map(mp_idx).to_numpy()
    col_ix = valid["votering_id"].map(event_idx).to_numpy()
    rost = valid["rost_code"].to_numpy(dtype=np.float32)
    M[row_ix, col_ix] = rost

    n_observed = (~np.isnan(M)).sum()
    print(f"  matrix shape: {M.shape}")
    print(f"  observed entries: {n_observed:,} of {M.size:,} "
          f"({n_observed/M.size:.0%})")

    votes_per_mp = (~np.isnan(M)).sum(axis=1)
    print(f"  votes per MP: median={int(np.median(votes_per_mp))}, "
          f"min={int(votes_per_mp.min())}, max={int(votes_per_mp.max())}")

    np.savez_compressed(OUT / "votes_matrix_18_22.npz",
                        M=M,
                        mp_order=np.array(mp_order),
                        vote_order=np.array(vote_order))
    print(f"  wrote votes_matrix_18_22.npz")


if __name__ == "__main__":
    main()
