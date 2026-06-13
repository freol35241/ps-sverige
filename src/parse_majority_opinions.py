"""Parse the majority opinion (Utskottets ställningstagande) from each
betänkande's HTML body.

This is the equivalent of reservations for the cabinet side: it's where the
utskott majority explains *why* it recommends what it does. We use it as
the source of cabinet-party excerpts in the voter tool.

Output: data/processed/majority_opinions.parquet with one row per section:
    dok_id, rm, beteckning, organ, rubrik (preceding R3 / R4 heading),
    text, n_words
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

# Headings that mark the start of the majority's reasoning. The text
# "Utskottets ställningstagande" is wrapped in nested span/a tags, so we
# match by allowing any HTML inside the heading <p> element.
R3_RE = re.compile(
    r'<p class="R3"[^>]*>(?:(?!</p>).)*?Utskottets ställningstagande(?:(?!</p>).)*?</p>(.*?)'
    r'(?=<p class="(?:R3|R4|Mellanrubrik|Stdrubrik|Avsnittsrubrik|Bilagerubrik)"[^>]*>|</body>|$)',
    re.DOTALL | re.IGNORECASE)
# Section break: stop when we reach Reservationer or Bilaga.
SECTION_END_RE = re.compile(
    r'<p class="(?:Avsnittsrubrik|Bilagerubrik|Rubrik1bilaga)"',
    re.IGNORECASE)
SOFT_HYPHEN_RE = re.compile(r"\s*­\s*")
HTML_TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")

# Preceding heading: look back for the most recent topic heading.
PRECEDING_HEADING_RE = re.compile(
    r'<p class="(?:R3|R4|Mellanrubrik|Stdrubrik)"[^>]*>(.+?)</p>',
    re.DOTALL)


def clean(text: str | None) -> str:
    if not text:
        return ""
    s = html_mod.unescape(text)
    s = HTML_TAG.sub(" ", s)
    s = SOFT_HYPHEN_RE.sub("", s)
    return WS.sub(" ", s).strip()


def extract_html_body(xml_path: Path) -> tuple[str, dict] | None:
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
    meta = {
        "dok_id": dok.findtext("dok_id") or "",
        "rm": dok.findtext("rm") or "",
        "beteckning": dok.findtext("beteckning") or "",
        "organ": dok.findtext("organ") or "",
    }
    return html_el.text, meta


def parse_one(xml_path: Path) -> list[dict]:
    payload = extract_html_body(xml_path)
    if not payload:
        return []
    body, meta = payload

    # Cap parse at the start of the Reservationer section if any — anything
    # after that is dissenting opinion, not majority.
    end_marker = re.search(
        r'<p class="Avsnittsrubrik"[^>]*>\s*Reservationer\s*</p>',
        body, re.IGNORECASE)
    if end_marker:
        body_majority = body[:end_marker.start()]
    else:
        body_majority = body

    rows = []
    for m in R3_RE.finditer(body_majority):
        # Get the preceding heading (topic context).
        preceding = list(PRECEDING_HEADING_RE.finditer(body_majority, 0, m.start()))
        rubrik = ""
        if preceding:
            # Last R3/R4 heading before the Ställningstagande marker.
            rubrik = clean(preceding[-1].group(1))
            # If it's the same as the marker heading, look further back.
            if "ställningstagande" in rubrik.lower() and len(preceding) > 1:
                rubrik = clean(preceding[-2].group(1))

        text = clean(m.group(1))
        n_words = len(text.split())
        if n_words < 50:
            continue
        rows.append({
            **meta,
            "rubrik": rubrik,
            "text": text,
            "n_words": n_words,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.xml"))
    print(f"parsing {len(files)} XMLs for majority opinions ...")
    all_rows = []
    n_no_section = 0
    for i, p in enumerate(files, 1):
        rows = parse_one(p)
        if not rows:
            n_no_section += 1
        all_rows.extend(rows)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}  sections so far: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    print(f"\nparsed {len(df):,} majority-opinion sections across "
          f"{df['dok_id'].nunique()} betänkanden")
    print(f"docs with no majority-opinion section found: {n_no_section}")
    print(f"\nword counts:")
    print(df["n_words"].describe().round(0).to_string())
    print(f"\nsample:")
    for _, r in df.sample(5, random_state=1).iterrows():
        print(f"\n  [{r['organ']}/{r['beteckning']}] {r['rubrik'][:60]}")
        print(f"    {r['text'][:300]}...")

    df.to_parquet(OUT / "majority_opinions.parquet", index=False)
    print(f"\nwrote {OUT}/majority_opinions.parquet "
          f"({(OUT / 'majority_opinions.parquet').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
