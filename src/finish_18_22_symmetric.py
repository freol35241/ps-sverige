"""Complete the analytical symmetry between 2018-22 and 2022-26.

Two pieces of work:
  1. SBERT-embed the 2018 manifesto sentences and build per
     (party, topic_id_unified) mean stated vectors, then compute
     2018-22 stated-vs-revealed gap.
  2. Fetch the 2018-22 anforande bulk dumps, parse into
     speeches_18_22.parquet, and build reasoning_vectors_18_22.npz
     for symmetric speech-based fallback.

Output:
  data/processed/manifesto_2018_sentences.parquet   (annotated 2018)
  data/processed/manifesto_2018_topic_assignment.parquet
  data/processed/stated_vs_revealed_topic_18_22.parquet
  data/processed/speeches_18_22.parquet
  data/processed/reasoning_vectors_18_22.npz
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from manifestos import _clean, _sentences, PARTY_FULL  # noqa: E402
from manifestos import PARTIES as MANIFESTO_PARTIES  # noqa: E402
from manifesto_elements import classify_all  # noqa: E402

IN = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

MODEL_NAME = "KBLab/sentence-bert-swedish-cased"
PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
N_TOPICS = 28
RIKSMOTEN = ["201819", "201920", "202021", "202122"]

# Manifesto-topic config (matches stated_vs_revealed_topic.py)
MIN_SENTENCES_PER_CELL = 5
MIN_TOPIC_SIMILARITY = 0.30

# Speech embedding config (matches reasoning_embed.py)
N_PER_CELL = 12
MIN_WORDS_PER_SPEECH = 50
MAX_WORDS_PER_SPEECH = 1500
WORDS_TO_EMBED = 400


# =============================================================================
# STAGE 1: 2018 manifesto embed + stated_vs_revealed
# =============================================================================

def parse_2018_manifestos() -> pd.DataFrame:
    """Same logic as src/manifestos.py but pointed at 2018_clean/."""
    year_dir = ROOT / "data" / "manifestos" / "2018_clean"
    rows = []
    for p in MANIFESTO_PARTIES:
        path = year_dir / f"{p}-2018.txt"
        if not path.exists():
            continue
        raw = _clean(path.read_text(encoding="utf-8"))
        paras = [pp.strip() for pp in re.split(r"\n\s*\n", raw) if pp.strip()]
        sent_idx = 0
        for para_idx, para in enumerate(paras):
            para = re.sub(r"\n", " ", para)
            for s in _sentences(para):
                rows.append({
                    "party": p, "party_name": PARTY_FULL[p], "year": 2018,
                    "sentence": s, "paragraph_idx": para_idx,
                    "sentence_idx": sent_idx, "n_words": len(s.split()),
                })
                sent_idx += 1
    df = pd.DataFrame(rows)
    return df


def topic_centroids():
    """Compute centroids per unified topic from the union embeddings."""
    npz_a = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    npz_b = np.load(IN / "embed_recommendations_18_22.npz", allow_pickle=True)
    X = np.vstack([npz_a["X"], npz_b["X"]]).astype(np.float32)
    ids = list(npz_a["votering_id"]) + list(npz_b["votering_id"])
    topics = pd.read_parquet(IN / "topics_unified.parquet")
    id_to_topic = dict(zip(topics["votering_id"].astype(str),
                            topics["topic_id_unified"]))
    bucket: dict[int, list[int]] = {}
    for i, vid in enumerate(ids):
        t = id_to_topic.get(str(vid))
        if t is None:
            continue
        bucket.setdefault(int(t), []).append(i)
    topic_ids = sorted(bucket.keys())
    centroids = np.stack([X[bucket[t]].mean(axis=0) for t in topic_ids])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids, topic_ids


def stage_manifesto_2018():
    print("== stage 1: 2018 manifesto SBERT embed + stated_vs_revealed ==")
    df = parse_2018_manifestos()
    print(f"  parsed {len(df):,} 2018 manifesto sentences")
    ann = classify_all(df)
    sub = ann[ann["element"] != "other"].reset_index(drop=True)
    print(f"  substantive sentences: {len(sub):,}")

    from sentence_transformers import SentenceTransformer
    print(f"  loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  embedding {len(sub):,} sentences ...")
    embs = model.encode(sub["sentence"].tolist(), batch_size=32,
                        show_progress_bar=False, normalize_embeddings=True
                        ).astype(np.float32)
    print(f"  embeddings shape: {embs.shape}")

    print("  assigning to unified topics ...")
    centroids, topic_ids = topic_centroids()
    sims = embs @ centroids.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(embs)), best_idx]
    assigned = np.array([topic_ids[i] for i in best_idx])
    sub["topic_id_unified"] = assigned
    sub["topic_sim"] = best_sim

    sub.drop(columns=["sentence"]).to_parquet(
        IN / "manifesto_2018_topic_assignment.parquet", index=False)
    print(f"  wrote manifesto_2018_topic_assignment.parquet")

    # Per (party, topic) mean vector for stated-vs-revealed
    print("  building stated vectors and computing gap ...")
    cells: dict[tuple[str, int], np.ndarray] = {}
    for (party, topic), g_idx in sub.groupby(
            ["party", "topic_id_unified"]).indices.items():
        idx = np.array(list(g_idx))
        rel_sim = sub.iloc[idx]["topic_sim"].to_numpy()
        keep = rel_sim >= MIN_TOPIC_SIMILARITY
        if keep.sum() < MIN_SENTENCES_PER_CELL:
            continue
        vec = embs[idx[keep]].mean(axis=0)
        cells[(party.upper(), int(topic))] = vec / np.linalg.norm(vec)

    rv = np.load(IN / "reservation_vectors_18_22.npz", allow_pickle=True)
    res_V, res_N = rv["V"], rv["N"]

    rows = []
    for (party, topic), m_vec in cells.items():
        i = PARTIES.index(party)
        if res_N[i, topic] == 0:
            continue
        r_vec = res_V[i, topic]
        na, nb = np.linalg.norm(m_vec), np.linalg.norm(r_vec)
        if not (na and nb):
            continue
        sim = float(m_vec @ r_vec / (na * nb))
        rows.append({
            "party": party, "topic_id_unified": topic,
            "n_manifesto_sentences": int((
                (sub["party"] == party.lower()) &
                (sub["topic_id_unified"] == topic)).sum()),
            "stated_revealed_cos": sim,
        })
    gap = pd.DataFrame(rows)
    gap.to_parquet(IN / "stated_vs_revealed_topic_18_22.parquet", index=False)
    print(f"  wrote stated_vs_revealed_topic_18_22.parquet ({len(gap)} cells)")
    per_party = gap.groupby("party")["stated_revealed_cos"].agg(
        ["mean", "std", "count"]).reset_index()
    print("\n  per-party mean topic-matched cosine (2018 stated vs 2018-22 revealed):")
    print(per_party.round(3).to_string(index=False))


# =============================================================================
# STAGE 2: 2018-22 speeches + reasoning vectors
# =============================================================================

def fetch_anforande_bulk():
    print("\n== stage 2a: fetch anforande bulk dumps for 2018-22 ==")
    RAW.mkdir(parents=True, exist_ok=True)
    for rm in RIKSMOTEN:
        name = f"anforande-{rm}.json.zip"
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  cached: {name} ({dest.stat().st_size:,} bytes)")
            continue
        url = f"https://data.riksdagen.se/dataset/anforande/{name}"
        print(f"  downloading {url}")
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        print(f"  {name}: {dest.stat().st_size:,} bytes")


def parse_speeches() -> pd.DataFrame:
    print("\n== stage 2b: parse 2018-22 anforande zips ==")
    KEEP = ("anforande_id", "intressent_id", "talare", "parti",
            "dok_id", "dok_rm", "dok_datum",
            "avsnittsrubrik", "underrubrik", "kammaraktivitet",
            "rel_dok_id", "anforande_nummer", "replik")
    TAG_RX = re.compile(r"<[^>]+>")
    WS_RX = re.compile(r"\s+")

    def strip_html(text):
        if not text:
            return ""
        s = TAG_RX.sub(" ", text)
        s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
        return WS_RX.sub(" ", s).strip()

    rows = []
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
                rows.append(row)
    df = pd.DataFrame(rows)
    df["dok_datum"] = pd.to_datetime(df["dok_datum"], errors="coerce")
    df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)
    print(f"  {len(df):,} speeches, "
          f"{df['intressent_id'].nunique()} unique speakers, "
          f"{df['n_words'].sum():,} total words")
    df.to_parquet(IN / "speeches_18_22.parquet", index=False)
    print(f"  wrote speeches_18_22.parquet")
    return df


def build_reasoning_vectors_18_22():
    print("\n== stage 2c: build 2018-22 reasoning vectors (speech-based, unified topics) ==")
    speeches = pd.read_parquet(IN / "speeches_18_22.parquet")
    texts = pd.read_parquet(IN / "vote_event_texts_18_22.parquet"
                            ).drop_duplicates("votering_id")
    topics = pd.read_parquet(IN / "topics_unified.parquet")
    topics = topics[topics["mandate"] == "2018-22"][
        ["votering_id", "topic_id_unified"]]
    print(f"  loaded {len(speeches):,} speeches, "
          f"{len(texts):,} vote_event_texts_18_22, "
          f"{len(topics):,} 2018-22 topic assignments")

    e2t = topics.merge(texts[["votering_id", "dok_id"]], on="votering_id")
    e2t["dok_id"] = e2t["dok_id"].str.upper()
    bet_topic = (e2t.groupby("dok_id")["topic_id_unified"]
                    .agg(lambda s: s.mode().iat[0])
                    .reset_index().rename(
                        columns={"topic_id_unified": "betänkande_topic"}))
    print(f"  betänkanden with topic: {len(bet_topic):,}")

    sp = speeches[speeches["parti"].isin(PARTIES)].copy()
    sp = sp.assign(rel_upper=sp["rel_dok_id"].astype(str).str.upper())
    sp_linked = sp.merge(bet_topic, left_on="rel_upper",
                         right_on="dok_id", how="inner")
    print(f"  linked speeches: {len(sp_linked):,}")
    sp_linked = sp_linked[sp_linked["replik"].fillna("N").str.upper() != "Y"]
    sp_linked = sp_linked[(sp_linked["n_words"] >= MIN_WORDS_PER_SPEECH) &
                          (sp_linked["n_words"] <= MAX_WORDS_PER_SPEECH)]
    print(f"  after filter: {len(sp_linked):,}")

    sp_linked = sp_linked.sort_values("n_words", ascending=False)
    picked = (sp_linked.groupby(["parti", "betänkande_topic"], group_keys=False)
                       .head(N_PER_CELL)).copy()
    picked["text_trunc"] = (picked["text"].str.split().str[:WORDS_TO_EMBED]
                            .str.join(" "))
    print(f"  picked for embedding: {len(picked):,}")
    cov = picked.groupby(["parti", "betänkande_topic"]).size().unstack(fill_value=0)
    cov = cov.reindex(PARTIES)
    print(f"  cells filled (pre-embed): {int((cov > 0).sum().sum())} of "
          f"{len(PARTIES) * N_TOPICS}")

    from sentence_transformers import SentenceTransformer
    print(f"  loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  embedding {len(picked):,} speeches ...")
    embs = model.encode(picked["text_trunc"].tolist(), batch_size=32,
                        show_progress_bar=False, normalize_embeddings=True
                        ).astype(np.float32)
    picked = picked.assign(_idx=np.arange(len(picked)))

    V = np.zeros((len(PARTIES), N_TOPICS, embs.shape[1]), dtype=np.float32)
    N = np.zeros((len(PARTIES), N_TOPICS), dtype=np.int32)
    for i, party in enumerate(PARTIES):
        for t in range(N_TOPICS):
            mask = ((picked["parti"] == party) &
                    (picked["betänkande_topic"] == t)).to_numpy()
            if mask.sum() == 0:
                continue
            V[i, t] = embs[picked["_idx"].to_numpy()[mask]].mean(axis=0)
            N[i, t] = int(mask.sum())
    print(f"  filled vector cells: {(N > 0).sum()} of {N.size}")

    np.savez_compressed(IN / "reasoning_vectors_18_22.npz",
                        V=V, N=N, parties=np.array(PARTIES, dtype=object))
    print(f"  wrote reasoning_vectors_18_22.npz")


def main() -> None:
    t0 = time.time()
    stage_manifesto_2018()
    print(f"\n[stage 1 elapsed: {time.time()-t0:.0f}s]")

    t1 = time.time()
    fetch_anforande_bulk()
    parse_speeches()
    build_reasoning_vectors_18_22()
    print(f"\n[stage 2 elapsed: {time.time()-t1:.0f}s]")
    print(f"\n== total elapsed: {time.time()-t0:.0f}s ==")


if __name__ == "__main__":
    main()
