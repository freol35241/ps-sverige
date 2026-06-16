"""Per-party manifesto register trajectories across 2018, 2022, 2026.

Re-runs the same keyword classifier and value vocabulary on the 2018
and 2022 manifesto sets (and L 2026 where available), then computes
per-party trajectories on:
  - element shares (diagnosis / end-state / mechanism)
  - signature value frequencies
  - manifesto length

Outputs:
  data/processed/manifesto_trajectories.parquet
    one row per (party, year, dimension, value)

Note: 2022 is loaded from the existing manifesto_sentences.parquet so
the numbers match the main article exactly. 2018 is re-parsed from the
2018_clean directory.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src"))
from manifestos import _clean, _sentences, PARTIES as PARTY_KEYS, PARTY_FULL
from manifesto_elements import classify_all, VALUE_PATTERNS


def parse_year(year_dir: Path, year: int) -> pd.DataFrame:
    """Parse all party manifestos from a year_dir (e.g. data/manifestos/2018_clean)."""
    rows = []
    for p in PARTY_KEYS:
        path = year_dir / f"{p}-{year}.txt"
        if not path.exists():
            continue
        raw = _clean(path.read_text(encoding="utf-8"))
        paras = [pp.strip() for pp in re.split(r"\n\s*\n", raw) if pp.strip()]
        sent_idx = 0
        local = []
        for para_idx, para in enumerate(paras):
            para = re.sub(r"\n", " ", para)
            for s in _sentences(para):
                local.append({
                    "party": p,
                    "party_name": PARTY_FULL[p],
                    "year": year,
                    "sentence": s,
                    "paragraph_idx": para_idx,
                    "sentence_idx": sent_idx,
                    "n_words": len(s.split()),
                })
                sent_idx += 1
        rows.extend(local)
    return pd.DataFrame(rows)


def aggregate_party(annotated: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    value_keys = list(VALUE_PATTERNS.keys())
    for p, g in annotated.groupby("party"):
        n = len(g)
        n_words = int(g["n_words"].sum())
        counts = g["element"].value_counts()
        rows.append({
            "party": p.upper(), "year": year, "n_sentences": n, "n_words": n_words,
            "share_diagnosis": float(counts.get("diagnosis", 0) / n),
            "share_end_state": float(counts.get("end_state", 0) / n),
            "share_mechanism": float(counts.get("mechanism", 0) / n),
        })
        for v in value_keys:
            col = f"v_{v}"
            if col in g.columns:
                rows[-1][f"v_{v}_per_kw"] = float(g[col].sum() / n_words * 1000)
    return rows


def main() -> None:
    out_rows = []

    # === 2018 ===
    print("== 2018 ==")
    df_2018 = parse_year(ROOT / "data" / "manifestos" / "2018_clean", 2018)
    if len(df_2018):
        ann = classify_all(df_2018)
        out_rows.extend(aggregate_party(ann, 2018))
        print(f"  {len(ann):,} sentences across {ann['party'].nunique()} parties")

    # === 2022 (from existing parquet) ===
    print("\n== 2022 ==")
    ann_2022 = pd.read_parquet(IN / "manifesto_sentences.parquet").copy()
    ann_2022["year"] = 2022
    out_rows.extend(aggregate_party(ann_2022, 2022))
    print(f"  {len(ann_2022):,} sentences across {ann_2022['party'].nunique()} parties")

    # === 2026 (L only, from L companion build) ===
    print("\n== 2026 (L only) ==")
    p_2026 = IN / "l_2026_sentences.parquet"
    if p_2026.exists():
        ann_2026 = pd.read_parquet(p_2026)
        out_rows.extend(aggregate_party(ann_2026, 2026))
        print(f"  {len(ann_2026):,} sentences (L only)")

    df = pd.DataFrame(out_rows).sort_values(["party", "year"])
    df.to_parquet(IN / "manifesto_trajectories.parquet", index=False)
    print(f"\nwrote manifesto_trajectories.parquet ({len(df)} rows)")

    # Print a quick per-party summary
    print("\n=== element shares by party and year ===")
    show = df[["party", "year", "share_diagnosis", "share_end_state",
               "share_mechanism", "n_words"]].copy()
    show[["share_diagnosis", "share_end_state", "share_mechanism"]] *= 100
    print(show.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
