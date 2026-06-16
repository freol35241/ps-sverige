"""Build the 2018-22 mandate vote-event corpus and embed it.

Parallel to the existing 2022-26 pipeline, output to *_18_22 names so as
not to clobber the 2022-26 processed data.

Discovered prefix pattern (vote-event filenames in the bulk-dump zips):
  2018/19 = H6, 2019/20 = H7, 2020/21 = H8, 2021/22 = H9
  2022/23 = HA, 2023/24 = HB, 2024/25 = HC, 2025/26 = HD

Stages, all idempotent:
  1. Fetch votering bulk dumps for 201819, 201920, 202021, 202122
  2. Parse to vote_events_18_22.parquet
  3. Fetch betänkanden XML for the (rm, beteckning) pairs referenced,
     using the H6.../H9... prefixes for the 2018-22 mandate
  4. Parse XML to vote_event_texts_18_22.parquet
  5. SBERT-embed the texts, save embed_recommendations_18_22.npz
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW_BET = RAW / "utskottsforslag_18_22"
OUT = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src"))
from parse_betankanden import parse_one as parse_betankande_xml  # noqa: E402

RIKSMOTEN = ["201819", "201920", "202021", "202122"]
RM_PREFIX = {
    "2018/19": "H601",
    "2019/20": "H701",
    "2020/21": "H801",
    "2021/22": "H901",
}
BASE = "https://data.riksdagen.se/dataset"
POLITE_DELAY = 0.30


# ---------------------------------------------------------------------------
# Stage 1: fetch votering bulk dumps
# ---------------------------------------------------------------------------

def fetch_bulk(dataset: str, riksmote: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    name = f"{dataset}-{riksmote}.json.zip"
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{BASE}/{dataset}/{name}"
    print(f"  downloading {url}")
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest


def stage_fetch_bulk() -> None:
    print("== stage 1: fetch votering bulk dumps for 2018-22 ==")
    for rm in RIKSMOTEN:
        p = fetch_bulk("votering", rm)
        print(f"  votering-{rm}.json.zip: {p.stat().st_size:>10,} bytes")


# ---------------------------------------------------------------------------
# Stage 2: parse vote events into a parquet
# ---------------------------------------------------------------------------

def stage_parse_events() -> pd.DataFrame:
    print("\n== stage 2: parse vote events ==")
    KEEP = ("votering_id", "rm", "beteckning", "punkt", "datum",
            "avser", "votering", "intressent_id", "parti", "rost")
    rows = []
    for rm in RIKSMOTEN:
        zpath = RAW / f"votering-{rm}.json.zip"
        n = 0
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                with z.open(name) as f:
                    payload = json.load(f)
                # payload = {"dokvotering": {"votering": [ {...}, ... ] } }
                votes = payload.get("dokvotering", {}).get("votering", [])
                if isinstance(votes, dict):
                    votes = [votes]
                for v in votes:
                    rows.append({k: v.get(k) for k in KEEP})
                    n += 1
        print(f"  {rm}: {n:>7,} vote rows")
    df = pd.DataFrame(rows)
    print(f"  total: {len(df):,} rows ({df['votering_id'].nunique():,} unique vote events)")
    # Reduce to vote-event metadata (one row per votering_id).
    events = (df.drop(columns=["intressent_id", "parti", "rost"])
                .drop_duplicates("votering_id")
                .reset_index(drop=True))
    events.to_parquet(OUT / "vote_events_18_22.parquet", index=False)
    print(f"  wrote vote_events_18_22.parquet ({len(events):,} events)")
    return events


# ---------------------------------------------------------------------------
# Stage 3: fetch betänkande XMLs
# ---------------------------------------------------------------------------

def dok_id_for(rm: str, beteckning: str) -> str:
    return f"{RM_PREFIX[rm]}{beteckning}"


def fetch_xml(dok_id: str) -> Path:
    RAW_BET.mkdir(parents=True, exist_ok=True)
    dest = RAW_BET / f"{dok_id}.xml"
    if dest.exists() and dest.stat().st_size > 200:
        return dest
    url = f"https://data.riksdagen.se/utskottsforslag/{dok_id}"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        dest.write_bytes(b"<!-- 404 -->")
        return dest
    r.raise_for_status()
    dest.write_bytes(r.content)
    time.sleep(POLITE_DELAY)
    return dest


def stage_fetch_betankanden(events: pd.DataFrame) -> None:
    print("\n== stage 3: fetch betänkanden XML ==")
    unique = events.drop_duplicates(["rm", "beteckning"])[["rm", "beteckning"]]
    unique = unique.dropna(subset=["beteckning"])
    unique = unique[unique["rm"].isin(RM_PREFIX)]
    print(f"  unique betänkanden to fetch: {len(unique)}")
    n_cached = n_new = n_404 = 0
    t0 = time.time()
    for i, (_, row) in enumerate(unique.iterrows(), 1):
        dok = dok_id_for(row["rm"], row["beteckning"])
        dest = RAW_BET / f"{dok}.xml"
        was_cached = dest.exists() and dest.stat().st_size > 200
        fetch_xml(dok)
        sz = dest.stat().st_size
        if was_cached:
            n_cached += 1
        elif sz < 200:
            n_404 += 1
        else:
            n_new += 1
        if i % 50 == 0:
            print(f"  {i}/{len(unique)}  cached:{n_cached} new:{n_new} 404:{n_404}  "
                  f"elapsed: {time.time()-t0:.1f}s")
    print(f"  done. cached: {n_cached}, new: {n_new}, 404: {n_404}")


# ---------------------------------------------------------------------------
# Stage 4: parse XMLs to vote_event_texts_18_22.parquet
# ---------------------------------------------------------------------------

def stage_parse_texts() -> pd.DataFrame:
    print("\n== stage 4: parse betänkande XMLs to vote-event texts ==")
    all_rows = []
    for p in sorted(RAW_BET.glob("*.xml")):
        all_rows.extend(parse_betankande_xml(p))
    df = pd.DataFrame(all_rows)
    print(f"  parsed {len(df):,} (votering_id, punkt) rows from "
          f"{len(list(RAW_BET.glob('*.xml')))} XMLs")

    events = pd.read_parquet(OUT / "vote_events_18_22.parquet")
    events["votering_id_upper"] = events["votering_id"].str.upper()
    link_rate = df["votering_id"].isin(events["votering_id_upper"]).mean()
    print(f"  link rate to vote_events_18_22: {link_rate:.1%}")
    df.to_parquet(OUT / "vote_event_texts_18_22.parquet", index=False)
    print(f"  wrote vote_event_texts_18_22.parquet")
    return df


# ---------------------------------------------------------------------------
# Stage 5: SBERT-embed the decision-point texts
# ---------------------------------------------------------------------------

def stage_embed() -> None:
    print("\n== stage 5: SBERT-embed decision-point texts ==")
    from sentence_transformers import SentenceTransformer

    texts = pd.read_parquet(OUT / "vote_event_texts_18_22.parquet")
    texts = texts.drop_duplicates("votering_id").reset_index(drop=True)
    # Mirror the 2022-26 embedding input: bet_titel + rubrik.
    docs = (texts["bet_titel"].fillna("") + " " + texts["rubrik"].fillna("")).tolist()
    print(f"  embedding {len(docs):,} unique vote-event texts ...")

    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    embs = model.encode(docs, batch_size=32, show_progress_bar=False,
                         normalize_embeddings=True).astype(np.float32)
    print(f"  embeddings shape: {embs.shape}")

    np.savez(OUT / "embed_recommendations_18_22.npz",
             X=embs,
             votering_id=np.array(texts["votering_id"].tolist()))
    print(f"  wrote embed_recommendations_18_22.npz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stage_fetch_bulk()
    events = stage_parse_events()
    stage_fetch_betankanden(events)
    stage_parse_texts()
    stage_embed()
    print("\n== done ==")


if __name__ == "__main__":
    main()
