"""Parse utskottsforslag XML into per-vote-event records.

Output: data/processed/vote_event_texts.parquet
Columns:
  votering_id, rm, beteckning, dok_id, punkt, rubrik, forslag_text,
  motforslag_partier (signed parties of the counter-proposal that was voted)

Each row corresponds to exactly one vote event in the chamber. The
recommendation text (rubrik + forslag) is what we'll embed as the
"what the vote was about" layer.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "utskottsforslag"
OUT = ROOT / "data" / "processed"

TAG_RX = re.compile(r"<[^>]+>")
WS_RX = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    s = text.replace("<br />", "\n").replace("<BR/>", "\n").replace("<br/>", "\n")
    s = TAG_RX.sub(" ", s)
    import html
    s = html.unescape(s)
    return WS_RX.sub(" ", s).strip()


def parse_one(xml_path: Path) -> list[dict]:
    if xml_path.stat().st_size < 200:
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    dok = root.find("dokument")
    if dok is None:
        return []
    dok_id = dok.findtext("dok_id") or ""
    rm = dok.findtext("rm") or ""
    beteckning = dok.findtext("beteckning") or ""
    organ = dok.findtext("organ") or ""
    bet_titel = strip_html(dok.findtext("titel") or "")

    # Collect reservation party-sets per punkt (one punkt may have multiple reservations).
    motforslag = root.find("dokmotforslag")
    reservations_by_punkt: dict[str, list[str]] = {}
    if motforslag is not None:
        for mf in motforslag.findall("motforslag"):
            if (mf.findtext("typ") or "").lower() != "reservation":
                continue
            punkt = mf.findtext("utskottsforslag_punkt") or ""
            partier = mf.findtext("partier") or ""
            partier = partier.replace('"', "").strip()
            if punkt:
                reservations_by_punkt.setdefault(punkt, []).append(partier)

    rows = []
    forslag_container = root.find("dokutskottsforslag")
    if forslag_container is None:
        return rows
    for uf in forslag_container.findall("utskottsforslag"):
        votering_id = (uf.findtext("votering_id") or "").upper()
        rubrik = strip_html(uf.findtext("rubrik"))
        forslag = strip_html(uf.findtext("forslag"))
        punkt = uf.findtext("punkt") or ""
        motforslag_partier = strip_html(uf.findtext("motforslag_partier")).replace('"', "")
        rows.append({
            "votering_id": votering_id,
            "dok_id": dok_id,
            "rm": rm,
            "beteckning": beteckning,
            "organ": organ,
            "bet_titel": bet_titel,
            "punkt": punkt,
            "rubrik": rubrik,
            "forslag_text": forslag,
            "motforslag_partier": motforslag_partier,
            "n_reservations": len(reservations_by_punkt.get(punkt, [])),
            "reservation_partier": ";".join(reservations_by_punkt.get(punkt, [])),
        })
    return rows


def main() -> None:
    all_rows = []
    for p in sorted(RAW.glob("*.xml")):
        all_rows.extend(parse_one(p))
    df = pd.DataFrame(all_rows)
    print(f"parsed {len(df):,} (votering_id) rows from {len(list(RAW.glob('*.xml')))} XMLs")

    # Link to our vote_events: case-insensitive votering_id join.
    events = pd.read_parquet(OUT / "vote_events.parquet")
    events["votering_id_upper"] = events["votering_id"].str.upper()
    matched = df.merge(events[["votering_id_upper", "datum"]],
                       left_on="votering_id", right_on="votering_id_upper",
                       how="left")
    coverage = matched["datum"].notna().mean()
    print(f"  link rate to our vote_events: {coverage:.1%}")
    print(f"  vote events without recommendation text: "
          f"{(~events['votering_id'].str.upper().isin(df['votering_id'])).sum()}")

    # Length stats
    df["forslag_words"] = df["forslag_text"].str.split().str.len()
    print(f"\nforslag_text word counts:  "
          f"median={df['forslag_words'].median():.0f}, "
          f"mean={df['forslag_words'].mean():.0f}, "
          f"max={df['forslag_words'].max():.0f}")
    print(f"n_reservations distribution:")
    print(df["n_reservations"].value_counts().sort_index().to_string())

    df.to_parquet(OUT / "vote_event_texts.parquet", index=False)
    print(f"\nwrote {OUT}/vote_event_texts.parquet")


if __name__ == "__main__":
    main()
