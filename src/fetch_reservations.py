"""Fetch dokumentstatus XML (with HTML body) for each betänkande.

Output: data/raw/dokumentstatus/<dok_id>.xml
~858 files, ~130 MB total. Polite rate-limit.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "dokumentstatus"
RAW.mkdir(parents=True, exist_ok=True)

POLITE_DELAY = 0.35


def main() -> None:
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates("dok_id"))
    dok_ids = tex["dok_id"].dropna().unique().tolist()
    print(f"betänkanden to fetch: {len(dok_ids)}")

    n_cached = n_new = n_err = 0
    for i, dok_id in enumerate(dok_ids, 1):
        dest = RAW / f"{dok_id}.xml"
        if dest.exists() and dest.stat().st_size > 1000:
            n_cached += 1
            if i % 100 == 0:
                print(f"  {i}/{len(dok_ids)}  cached={n_cached} new={n_new} err={n_err}")
            continue
        url = f"https://data.riksdagen.se/dokumentstatus/{dok_id}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            n_new += 1
        except Exception as e:
            print(f"  ERR {dok_id}: {e}")
            n_err += 1
        time.sleep(POLITE_DELAY)
        if i % 50 == 0:
            print(f"  {i}/{len(dok_ids)}  cached={n_cached} new={n_new} err={n_err}")

    print(f"\ndone. cached={n_cached} new={n_new} err={n_err}")
    total = sum(p.stat().st_size for p in RAW.glob("*.xml"))
    print(f"total raw: {total / 1e6:.1f} MB across {len(list(RAW.glob('*.xml')))} files")


if __name__ == "__main__":
    main()
