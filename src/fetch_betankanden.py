"""Fetch utskottsforslag XML for every (rm, beteckning) referenced by a vote event.

dok_id pattern: <prefix><beteckning>, where prefix is
  HA01 / HB01 / HC01 / HD01 for 2022/23 / 2023/24 / 2024/25 / 2025/26.

The utskottsforslag endpoint gives clean structured XML with one
<utskottsforslag> per förslagspunkt, including the votering_id linking back
to the vote events and the recommendation text (<forslag>).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "utskottsforslag"

RM_PREFIX = {"2022/23": "HA01", "2023/24": "HB01",
             "2024/25": "HC01", "2025/26": "HD01"}

POLITE_DELAY = 0.30  # seconds between requests


def dok_id_for(rm: str, beteckning: str) -> str:
    return f"{RM_PREFIX[rm]}{beteckning}"


def fetch_one(dok_id: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"{dok_id}.xml"
    if dest.exists() and dest.stat().st_size > 200:
        return dest
    url = f"https://data.riksdagen.se/utskottsforslag/{dok_id}"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        # Mark as missing so we don't retry.
        dest.write_bytes(b"<!-- 404 -->")
        return dest
    r.raise_for_status()
    dest.write_bytes(r.content)
    time.sleep(POLITE_DELAY)
    return dest


def main() -> None:
    events = pd.read_parquet(IN / "vote_events.parquet")
    unique = events.drop_duplicates(["rm", "beteckning"])[["rm", "beteckning"]]
    print(f"unique betänkanden to fetch: {len(unique)}")

    n_cached = n_new = n_404 = 0
    for i, (_, row) in enumerate(unique.iterrows(), 1):
        dok = dok_id_for(row["rm"], row["beteckning"])
        dest = RAW / f"{dok}.xml"
        was_cached = dest.exists() and dest.stat().st_size > 200
        fetch_one(dok)
        sz = dest.stat().st_size
        if was_cached:
            n_cached += 1
        elif sz < 200:
            n_404 += 1
        else:
            n_new += 1
        if i % 50 == 0 or i == len(unique):
            print(f"  {i}/{len(unique)}  cached={n_cached} new={n_new} 404={n_404}")

    print(f"\ndone. cached={n_cached} new={n_new} 404={n_404}")


if __name__ == "__main__":
    main()
