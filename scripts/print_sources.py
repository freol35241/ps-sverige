"""Print the best candidate source texts per (topic, party) for manual distillation.

Run with --topic N to focus on one topic, or no arg to see all 12 prioritised
topics. Output is intended to be read by a human editor who writes distilled
~50-word statements based on the source.

Usage:
    .venv/bin/python scripts/print_sources.py --topic 13 > sources_13.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

# 12 topics prioritised for curated distillation, by analytical importance
# and voter relevance. Ordered roughly by per-topic WHAT-spread.
PRIORITY_TOPICS = [13, 18, 7, 11, 9, 23, 3, 6, 21, 15, 24, 2]

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
OPP = {"V", "S", "MP", "C", "SD"}
CABINET = {"L", "KD", "M"}

SOFT_HYPHEN = re.compile(r"\s*­\s*")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = SOFT_HYPHEN.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def best_opposition_sources(topic_id: int, party: str, k: int = 3) -> list[dict]:
    """Top-k opposition source candidates: longest, best-fitting reservations."""
    res = pd.read_parquet(IN / "reservations.parquet")
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates(["dok_id", "punkt"])
             [["dok_id", "punkt", "votering_id"]])
    topics = pd.read_parquet(IN / "topics.parquet")

    res_linked = (res.merge(tex, on=["dok_id", "punkt"], how="left")
                     .merge(topics[["votering_id", "topic_id", "dist_to_centroid"]],
                            on="votering_id", how="left"))
    res_linked = res_linked[res_linked["topic_id"] == topic_id]
    res_linked = res_linked[res_linked["partier"].str.contains(party, na=False)]
    # Only solo or party-led — others may not represent this party's position cleanly.
    res_linked = res_linked[res_linked["partier"].apply(
        lambda s: party in str(s).split(";") and len(str(s).split(";")) <= 2)]
    # Prefer longer ställningstagande and better cluster fit.
    res_linked = res_linked.sort_values(
        ["dist_to_centroid", "n_stallnings_words"],
        ascending=[True, False])
    out = []
    for _, r in res_linked.head(k).iterrows():
        text = clean_text(r["stallningstagande_text"])
        if len(text.split()) < 40:
            continue
        out.append({
            "source": f"{r['dok_id']} punkt {r['punkt']}",
            "rubrik": clean_text(r["rubrik"]),
            "partier": r["partier"],
            "text": text,
        })
    return out


def best_cabinet_sources(topic_id: int, k: int = 5) -> list[dict]:
    """Top-k majority-opinion candidates (cabinet's position)."""
    maj = pd.read_parquet(IN / "majority_opinions.parquet")
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates(["dok_id", "punkt"])
             [["dok_id", "punkt", "votering_id"]])
    topics = pd.read_parquet(IN / "topics.parquet")
    e2t = topics.merge(tex[["dok_id", "votering_id"]], on="votering_id")
    e2t["dok_id"] = e2t["dok_id"].str.upper()
    bet_topic = (e2t.groupby("dok_id")["topic_id"]
                    .agg(lambda s: s.mode().iat[0])
                    .reset_index())
    maj["dok_id_u"] = maj["dok_id"].str.upper()
    maj_linked = maj.merge(bet_topic, left_on="dok_id_u", right_on="dok_id",
                            how="inner", suffixes=("", "_bet"))
    maj_linked = maj_linked[maj_linked["topic_id"] == topic_id]
    maj_linked = maj_linked.sort_values("n_words", ascending=False)
    out = []
    for _, r in maj_linked.head(k).iterrows():
        text = clean_text(r["text"])
        if len(text.split()) < 60:
            continue
        out.append({
            "source": f"{r['dok_id']} (Utskottets ställningstagande)",
            "rubrik": clean_text(r.get("rubrik", "")),
            "text": text,
        })
    return out


def topic_label(topic_id: int) -> str:
    meta = pd.read_parquet(IN / "topic_meta.parquet").set_index("topic_id")
    return meta.loc[topic_id, "label_terms"].split(",")[0]


def print_topic(topic_id: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"## TOPIC {topic_id}: {topic_label(topic_id)}")
    print(f"{'=' * 70}\n")

    for party in PARTIES:
        print(f"\n### [{party}]")
        if party in CABINET:
            # Cabinet parties share majority-opinion source.
            srcs = best_cabinet_sources(topic_id, k=4)
            note = "(från Utskottets ställningstagande — kabinettets gemensamma position)"
        else:
            srcs = best_opposition_sources(topic_id, party, k=4)
            note = ""
        print(f"  {note}\n")
        if not srcs:
            print(f"  [INGA KÄLLOR HITTADE]\n")
            continue
        for i, src in enumerate(srcs, 1):
            t = src["text"]
            if len(t) > 600:
                t = t[:600] + "…"
            print(f"  **{i}. {src['rubrik']}** _({src['source']}, {src.get('partier', '')})_")
            print(f"  > {t}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=int, default=None,
                    help="Specific topic id to print")
    ap.add_argument("--all", action="store_true",
                    help="Print all 12 priority topics")
    args = ap.parse_args()

    if args.topic is not None:
        print_topic(args.topic)
    elif args.all:
        for t in PRIORITY_TOPICS:
            print_topic(t)
    else:
        # Default: print just topic 13 (migration) as a sample
        print_topic(13)


if __name__ == "__main__":
    main()
