"""Parse reservation reasoning text from dokumentstatus XML.

The HTML body inside each XML contains reservation sections marked by
<p class="Reservationsrubrik">N. Title, punkt M (Parties)</p>
followed by the author line and reasoning paragraphs, until the next
Reservationsrubrik or the end of the reservations section.

Output: data/processed/reservations.parquet with one row per reservation:
    dok_id, rm, beteckning, punkt, partier (sorted, semicolon-joined),
    rubrik, reasoning_text, n_words
"""
from __future__ import annotations

import html as html_mod
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "dokumentstatus"
OUT = ROOT / "data" / "processed"

PARTIES = {"V", "S", "MP", "C", "L", "KD", "M", "SD"}

# Headings to recognise.
RES_RUBRIK = re.compile(
    r'<p class="Reservationsrubrik"[^>]*>(.+?)</p>', re.DOTALL)
# Headings that end the reservations section (or start a new top-level section).
SECTION_END = re.compile(
    r'<p class="(Bilaga|Bilagerubrik|Avsnittsrubrik|Rubrik1Dokumentinformation|R1bilaga)"',
    re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
# "title, punkt N (Parties)" — parties group is comma-separated single letters/pairs.
HEADING_RE = re.compile(
    r"^\s*(?P<title>.+?),\s*punkt\s*(?P<punkt>\d+)\s*\((?P<parties>[^)]+)\)\s*$",
    re.DOTALL)


def clean(text: str) -> str:
    s = html_mod.unescape(text)
    s = HTML_TAG.sub(" ", s)
    return WS.sub(" ", s).strip()


def extract_html_body(xml_path: Path) -> str | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    dok = root.find("dokument")
    if dok is None:
        return None
    html_el = dok.find("html")
    if html_el is None or not html_el.text:
        return None
    return html_el.text  # ElementTree already unescaped CDATA


def parse_one(xml_path: Path) -> list[dict]:
    body = extract_html_body(xml_path)
    if body is None:
        return []
    root = ET.parse(xml_path).getroot().find("dokument")
    dok_id = root.findtext("dok_id") or ""
    rm = root.findtext("rm") or ""
    beteckning = root.findtext("beteckning") or ""

    # Find Reservationsrubrik occurrences with their body span.
    heads = list(RES_RUBRIK.finditer(body))
    if not heads:
        return []

    rows = []
    for i, m in enumerate(heads):
        head_text = clean(m.group(1))
        match = HEADING_RE.match(head_text)
        if not match:
            continue
        title = match.group("title")
        punkt = match.group("punkt")
        parties_raw = match.group("parties")
        partier = sorted(p.strip() for p in re.split(r"[,\s]+", parties_raw)
                         if p.strip() in PARTIES)
        if not partier:
            continue

        # Body span: from end of this heading to start of next heading or section end.
        start = m.end()
        if i + 1 < len(heads):
            end = heads[i + 1].start()
        else:
            # Look for section end after this heading.
            after = body[start:]
            se = SECTION_END.search(after)
            end = start + se.start() if se else len(body)

        reasoning = clean(body[start:end])
        # Drop the leading "Av X (Y), Z (W) ..." author line up to the first heading
        # marker if we can identify one. Simple heuristic: keep everything from
        # "Förslag till riksdagsbeslut" onwards if it occurs, else the whole span.
        marker = re.search(
            r"(Förslag till riksdagsbeslut|Ställningstagande|Skälen|Motiverin)",
            reasoning, re.IGNORECASE)
        if marker:
            reasoning = reasoning[marker.start():].strip()

        n_words = len(reasoning.split())
        if n_words < 30:
            continue
        rows.append({
            "dok_id": dok_id,
            "rm": rm,
            "beteckning": beteckning,
            "punkt": punkt,
            "partier": ";".join(partier),
            "n_partier": len(partier),
            "rubrik": title,
            "reasoning_text": reasoning,
            "n_words": n_words,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.xml"))
    print(f"parsing {len(files)} XMLs ...")
    all_rows = []
    n_no_html = 0
    for i, p in enumerate(files, 1):
        rows = parse_one(p)
        if not rows:
            n_no_html += 1
        all_rows.extend(rows)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}  reservations so far: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    print(f"\nparsed {len(df):,} reservations across {df['dok_id'].nunique()} betänkanden")
    print(f"docs with zero reservations parsed: {n_no_html}")
    print(f"\nn_words distribution:")
    print(df["n_words"].describe().round(1).to_string())
    print(f"\nn_partier per reservation:")
    print(df["n_partier"].value_counts().sort_index().to_string())
    print(f"\nSingle-party reservation count by party:")
    single = df[df["n_partier"] == 1]
    print(single["partier"].value_counts().to_string())
    print(f"\nTop multi-party combinations:")
    multi = df[df["n_partier"] > 1]
    print(multi["partier"].value_counts().head(10).to_string())

    df.to_parquet(OUT / "reservations.parquet", index=False)
    print(f"\nwrote {OUT}/reservations.parquet "
          f"({(OUT / 'reservations.parquet').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
