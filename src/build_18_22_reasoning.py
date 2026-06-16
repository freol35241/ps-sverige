"""Build reservation reasoning vectors for both mandates, keyed by unified topic IDs.

Stages:
  1. Fetch dokumentstatus XML for each 2018-22 betänkande (1,052 files)
  2. Parse reservations from those XMLs to reservations_18_22.parquet
  3. SBERT-embed reservations for both mandates and build per
     (party, topic_id_unified) reasoning vectors:
       reservation_vectors_18_22.npz
       reservation_vectors_22_26_unified.npz   (re-keyed 2022-26 vectors)

The 2022-26 stage A (fetch + parse to reservations.parquet) has already
been done by the existing pipeline; we only re-key its embeddings into
the unified topic indexing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parse_reservations import (  # noqa: E402
    extract_html_body,
    clean,
    RES_RUBRIK,
    SECTION_END,
    HEADING_RE,
    PARTIES as RES_PARTIES,
    extract_stallningstagande,
)

RAW = ROOT / "data" / "raw"
RAW_DOK_18_22 = RAW / "dokumentstatus_18_22"
IN = ROOT / "data" / "processed"

POLITE_DELAY = 0.35
PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
N_TOPICS = 28
N_PER_CELL = 20
WORDS_TO_EMBED = 400


# ---------------------------------------------------------------------------
# Stage 1: fetch dokumentstatus XMLs for 2018-22
# ---------------------------------------------------------------------------

def stage_fetch_dokumentstatus() -> None:
    print("== stage 1: fetch dokumentstatus XMLs for 2018-22 ==")
    RAW_DOK_18_22.mkdir(parents=True, exist_ok=True)
    tex = (pd.read_parquet(IN / "vote_event_texts_18_22.parquet")
             .drop_duplicates("dok_id"))
    dok_ids = tex["dok_id"].dropna().unique().tolist()
    print(f"  dok_ids to fetch: {len(dok_ids)}")

    n_cached = n_new = n_err = 0
    t0 = time.time()
    for i, dok_id in enumerate(dok_ids, 1):
        dest = RAW_DOK_18_22 / f"{dok_id}.xml"
        if dest.exists() and dest.stat().st_size > 1000:
            n_cached += 1
            continue
        url = f"https://data.riksdagen.se/dokumentstatus/{dok_id}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            n_new += 1
            time.sleep(POLITE_DELAY)
        except requests.RequestException as e:
            print(f"  err on {dok_id}: {e}")
            n_err += 1
        if i % 100 == 0:
            print(f"  {i}/{len(dok_ids)}  cached={n_cached} new={n_new} err={n_err}  "
                  f"elapsed={time.time()-t0:.1f}s")
    print(f"  done. cached={n_cached}, new={n_new}, err={n_err}")


# ---------------------------------------------------------------------------
# Stage 2: parse reservations
# ---------------------------------------------------------------------------

def parse_reservations_dir(raw_dir: Path) -> pd.DataFrame:
    """Inlined parser, since parse_reservations.main() hardcodes RAW."""
    import re

    rows = []
    for p in sorted(raw_dir.glob("*.xml")):
        body = extract_html_body(p)
        if not body:
            continue
        dok_id = p.stem
        # Find each reservation by its rubrik.
        matches = list(RES_RUBRIK.finditer(body))
        for i, m in enumerate(matches):
            heading_raw = clean(m.group(1))
            heading = HEADING_RE.match(heading_raw)
            if not heading:
                continue
            title = heading.group("title").strip()
            punkt = heading.group("punkt")
            parties_raw = heading.group("parties").replace('"', "").strip()
            partier = ";".join(
                p.strip() for p in parties_raw.split(",") if p.strip() in RES_PARTIES
            )
            if not partier:
                continue
            # Body: from end of this rubrik to start of next rubrik or end-section
            start = m.end()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                next_sec = SECTION_END.search(body, start)
                end = next_sec.start() if next_sec else len(body)
            body_text = clean(body[start:end])
            stallning = extract_stallningstagande(body_text)
            rows.append({
                "dok_id": dok_id,
                "punkt": punkt,
                "partier": partier,
                "n_partier": len(partier.split(";")),
                "rubrik": title,
                "reasoning_text": body_text,
                "stallningstagande_text": stallning,
                "n_words": len(body_text.split()),
            })
    return pd.DataFrame(rows)


def stage_parse_reservations_18_22() -> pd.DataFrame:
    print("\n== stage 2: parse reservations for 2018-22 ==")
    df = parse_reservations_dir(RAW_DOK_18_22)
    print(f"  parsed {len(df):,} reservations from "
          f"{len(list(RAW_DOK_18_22.glob('*.xml')))} XMLs")
    df.to_parquet(IN / "reservations_18_22.parquet", index=False)
    print(f"  wrote reservations_18_22.parquet")
    return df


# ---------------------------------------------------------------------------
# Stage 3: embed reservations for both mandates using unified topic IDs
# ---------------------------------------------------------------------------

def build_vectors_for_mandate(mandate_key: str, out_name: str) -> None:
    print(f"\n  -- {mandate_key}: building per-(party, unified topic) vectors --")
    # Load relevant reservations
    if mandate_key == "2018-22":
        res = pd.read_parquet(IN / "reservations_18_22.parquet")
        tex = (pd.read_parquet(IN / "vote_event_texts_18_22.parquet")
                 .drop_duplicates(["dok_id", "punkt"]))
    elif mandate_key == "2022-26":
        res = pd.read_parquet(IN / "reservations.parquet")
        tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
                 .drop_duplicates(["dok_id", "punkt"]))
    else:
        raise ValueError(mandate_key)

    # Topic assignment from the unified clustering
    topics = pd.read_parquet(IN / "topics_unified.parquet")
    topics = topics[topics["mandate"] == mandate_key][
        ["votering_id", "topic_id_unified"]].rename(
        columns={"topic_id_unified": "topic_id"})
    # Case-normalise votering_id since vote_event_texts has mixed case
    topics["votering_id"] = topics["votering_id"].astype(str)
    tex["votering_id"] = tex["votering_id"].astype(str)

    tex = tex.merge(topics, on="votering_id", how="left")
    # Normalise dok_id casing on both sides
    tex["dok_id"] = tex["dok_id"].astype(str)
    res["dok_id"] = res["dok_id"].astype(str)
    res_linked = res.merge(tex[["dok_id", "punkt", "topic_id"]],
                            on=["dok_id", "punkt"], how="left")
    n_linked = res_linked["topic_id"].notna().sum()
    print(f"    reservations linked to a unified topic: "
          f"{n_linked:,} / {len(res_linked):,} ({n_linked/len(res_linked):.0%})")

    # Expand to (party, reservation) rows
    rows = []
    for _, r in res_linked.iterrows():
        if pd.isna(r["topic_id"]):
            continue
        for p in str(r["partier"]).split(";"):
            if p in PARTIES:
                rows.append({
                    "party": p,
                    "topic_id": int(r["topic_id"]),
                    "dok_id": r["dok_id"],
                    "punkt": r["punkt"],
                    "rubrik": r["rubrik"],
                    "reasoning_text": r["reasoning_text"],
                    "n_words": int(r["n_words"]),
                })
    expanded = pd.DataFrame(rows)
    print(f"    expanded to (party, reservation): {len(expanded):,}")
    expanded = expanded.sort_values("n_words", ascending=False)
    picked = (expanded.groupby(["party", "topic_id"], group_keys=False)
                      .head(N_PER_CELL)).copy()
    print(f"    picked for embedding: {len(picked):,}")
    coverage = picked.groupby(["party", "topic_id"]).size().unstack(fill_value=0)
    coverage = coverage.reindex(PARTIES)
    cells_filled = int((coverage > 0).sum().sum())
    print(f"    cells filled: {cells_filled} of {len(PARTIES) * N_TOPICS}")

    picked["text_trunc"] = (picked["reasoning_text"]
                            .str.split().str[:WORDS_TO_EMBED].str.join(" "))

    from sentence_transformers import SentenceTransformer
    print(f"    loading SBERT ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    print(f"    embedding {len(picked):,} reservations ...")
    embs = model.encode(picked["text_trunc"].tolist(),
                        batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    picked = picked.assign(_idx=np.arange(len(picked)))

    V = np.zeros((len(PARTIES), N_TOPICS, embs.shape[1]), dtype=np.float32)
    N = np.zeros((len(PARTIES), N_TOPICS), dtype=np.int32)
    for i, party in enumerate(PARTIES):
        for t in range(N_TOPICS):
            mask = ((picked["party"] == party) &
                    (picked["topic_id"] == t)).to_numpy()
            if mask.sum() == 0:
                continue
            V[i, t] = embs[picked["_idx"].to_numpy()[mask]].mean(axis=0)
            N[i, t] = int(mask.sum())
    print(f"    filled vector cells: {(N > 0).sum()} of {N.size}")

    np.savez_compressed(IN / out_name,
                        V=V, N=N,
                        parties=np.array(PARTIES, dtype=object))
    print(f"    wrote {out_name}")


def stage_embed_both_mandates() -> None:
    print("\n== stage 3: embed reservations for both mandates against unified topic IDs ==")
    build_vectors_for_mandate("2018-22", "reservation_vectors_18_22.npz")
    build_vectors_for_mandate("2022-26", "reservation_vectors_22_26_unified.npz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stage_fetch_dokumentstatus()
    stage_parse_reservations_18_22()
    stage_embed_both_mandates()
    print("\n== done ==")


if __name__ == "__main__":
    main()
