"""Parse, segment and sentence-tokenise the 2022 election manifestos.

Output: data/processed/manifestos.parquet with one row per sentence and
        a position column normalised to [0, 1] for "where in the document
        does this sentence appear" (useful for distinguishing diagnostic
        opening sections from concluding aspirational text).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "manifestos" / "2022_clean"
OUT = ROOT / "data" / "processed"

PARTIES = ["v", "s", "mp", "c", "l", "kd", "m", "sd"]
PARTY_FULL = {
    "v": "Vänsterpartiet", "s": "Socialdemokraterna",
    "mp": "Miljöpartiet", "c": "Centerpartiet",
    "l": "Liberalerna", "kd": "Kristdemokraterna",
    "m": "Moderaterna", "sd": "Sverigedemokraterna",
}

# Sentence boundary: stops on . ! ? followed by whitespace + uppercase or
# end of string. Avoids splitting on common abbreviations like t.ex., m.fl.
ABBREVS = {"t.ex", "bl.a", "m.fl", "m.m", "dvs", "ca", "kr", "s.k", "fr.o.m", "t.o.m"}
SENTENCE_RX = re.compile(r"(?<=[\.!?])\s+(?=[A-ZÅÄÖ])")


def _clean(text: str) -> str:
    """Strip control chars, normalise whitespace."""
    text = text.replace("﻿", "")  # BOM
    text = re.sub(r"\xad", "", text)   # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _sentences(text: str) -> list[str]:
    """Sentence-tokenise. Two-pass to keep abbreviations intact."""
    parts = SENTENCE_RX.split(text)
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Skip headings without terminal punctuation (often all-caps lines).
        if len(p) < 25 and not re.search(r"[\.!?]$", p):
            continue
        sentences.append(p)
    return sentences


def parse_one(party: str) -> pd.DataFrame:
    path = IN / f"{party}-2022.txt"
    raw = path.read_text(encoding="utf-8")
    raw = _clean(raw)

    # Split into "paragraphs" by blank lines, keep order.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    rows = []
    sent_idx = 0
    for para_idx, para in enumerate(paragraphs):
        # Collapse internal newlines to spaces.
        para = re.sub(r"\n", " ", para)
        for s in _sentences(para):
            rows.append({
                "party": party,
                "party_name": PARTY_FULL[party],
                "sentence": s,
                "paragraph_idx": para_idx,
                "sentence_idx": sent_idx,
                "n_words": len(s.split()),
            })
            sent_idx += 1
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["position"] = df["sentence_idx"] / (len(df) - 1) if len(df) > 1 else 0.5
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for party in PARTIES:
        df = parse_one(party)
        print(f"  {party}: {len(df):>4} sentences, "
              f"{df['n_words'].sum():>5,} words, "
              f"avg {df['n_words'].mean():.1f} w/s")
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(OUT / "manifestos.parquet", index=False)
    print(f"\nwrote {OUT}/manifestos.parquet ({len(combined):,} sentences)")


if __name__ == "__main__":
    main()
