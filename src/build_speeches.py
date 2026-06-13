"""Parse anforande bulk zips into a long table of speeches."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

RIKSMOTEN = ["202223", "202324", "202425", "202526"]

KEEP = ("anforande_id", "intressent_id", "talare", "parti",
        "dok_id", "dok_rm", "dok_datum",
        "avsnittsrubrik", "underrubrik", "kammaraktivitet",
        "rel_dok_id",
        "anforande_nummer", "replik")

TAG_RX = re.compile(r"<[^>]+>")
WS_RX = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    s = TAG_RX.sub(" ", text)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    s = WS_RX.sub(" ", s).strip()
    return s


def iter_speeches():
    for rm in RIKSMOTEN:
        zpath = RAW / f"anforande-{rm}.json.zip"
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                with z.open(name) as f:
                    payload = json.load(f)
                a = payload["anforande"]
                row = {k: a.get(k) for k in KEEP}
                row["text"] = strip_html(a.get("anforandetext"))
                yield row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("parsing speech zips ...")
    rows = list(iter_speeches())
    df = pd.DataFrame(rows)
    df["dok_datum"] = pd.to_datetime(df["dok_datum"], errors="coerce")
    df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)

    print(f"  {len(df):,} speeches, "
          f"{df['intressent_id'].nunique()} unique speakers, "
          f"{df['n_words'].sum():,} total words")

    print("\nPer riksmöte:")
    for rm, g in df.groupby("dok_rm"):
        print(f"  {rm}: {len(g):>6,} speeches, "
              f"{g['n_words'].sum():>10,} words, "
              f"median {g['n_words'].median():>4} words/speech")

    print("\nPer party (all years):")
    by_party = (df.groupby("parti")
                  .agg(n_speeches=("anforande_id", "count"),
                       n_words=("n_words", "sum"),
                       n_speakers=("intressent_id", "nunique"))
                  .sort_values("n_words", ascending=False))
    print(by_party.to_string())

    df.to_parquet(OUT / "speeches.parquet", index=False)
    print(f"\nwrote {OUT}/speeches.parquet  "
          f"({(OUT / 'speeches.parquet').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
