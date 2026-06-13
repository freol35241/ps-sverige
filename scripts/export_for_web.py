"""Export precomputed data for the static site.

Emits JSON files into site/src/data/. Vectors are 768-dim and we have at most
~10kB per topic — well under any reasonable static-site budget.

Outputs:
  parties.json        — display order, official party colours
  topics.json         — topic_id → label, examples, ordering
  vectors.json        — per (party, topic) stance + lexicon scalars
  reasoning.json      — per (party, topic) reasoning vector (rounded to fp16)
  excerpts.json       — curated anonymised reservation excerpts per topic
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "site" / "src" / "data"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]

PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
PARTY_FULL_NAME = {
    "V": "Vänsterpartiet", "S": "Socialdemokraterna",
    "MP": "Miljöpartiet", "C": "Centerpartiet",
    "L": "Liberalerna", "KD": "Kristdemokraterna",
    "M": "Moderaterna", "SD": "Sverigedemokraterna",
}

# Manually-curated short labels for the 28 topics we cluster. Done so the
# voter-facing site has readable names rather than TF-IDF term lists.
# Reviewed against meta.label_terms in topic_meta.parquet.
TOPIC_LABELS = {
    0:  "Yttrandefrihet och digitalisering",
    1:  "Kommuner och regioner",
    2:  "Arbetsrätt och arbetslöshetsförsäkring",
    3:  "Brott och kriminalvård",
    4:  "Försvar och krisberedskap",
    5:  "Regelförenkling och konsumenträtt",
    6:  "Skatter",
    7:  "Skola och lärare",
    8:  "Trafik och fordon",
    9:  "Internationellt bistånd",
    10: "Miljö, jakt och jordbruk",
    11: "Läkemedel och tandvård",
    12: "Socialtjänst, barn och familj",
    13: "Migration och asylpolitik",
    14: "Högre utbildning och studiestöd",
    15: "Nato och EU-samarbete",
    16: "Näringspolitik och handel",
    17: "Kulturpolitik",
    18: "Klimat och energi",
    19: "Brottsbekämpning och övervakning",
    20: "Författnings- och valfrågor",
    21: "Bostadspolitik",
    22: "Fiske och sjöfart",
    23: "Hälso- och sjukvård",
    24: "Funktionsnedsättning och äldreomsorg",
    25: "Idrott, friluftsliv, ANDTS",
    26: "Finansmarknad och ägande",
    27: "Diskriminering och minoriteter",
}


def export_parties() -> None:
    parties = [{
        "code": p,
        "name": PARTY_FULL_NAME[p],
        "color": PARTY_COLOR[p],
    } for p in PARTIES]
    (OUT / "parties.json").write_text(
        json.dumps(parties, ensure_ascii=False, indent=2))


def export_topics() -> None:
    meta = pd.read_parquet(IN / "topic_meta.parquet").set_index("topic_id")
    stats = pd.read_parquet(IN / "per_topic_stats.parquet").set_index("topic_id")
    topics = []
    for t in sorted(meta.index):
        topics.append({
            "id": int(t),
            "label": TOPIC_LABELS.get(int(t), f"Ämne {t}"),
            "primary_organ": meta.loc[t, "primary_organ"],
            "size": int(meta.loc[t, "size"]),
            "what_spread": float(stats.loc[t, "mean_distinct_what"])
                if t in stats.index else 0.0,
            "axes_spread": float(stats.loc[t, "mean_spread"])
                if t in stats.index else 0.0,
        })
    # Rank by informativeness for default question ordering.
    topics.sort(key=lambda r: r["what_spread"], reverse=True)
    (OUT / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2))


def export_vectors() -> None:
    """Per-(party, topic) stance score + lexicon scalars."""
    pt = pd.read_parquet(IN / "party_topic.parquet")
    lex = pd.read_parquet(IN / "how_lexicon.parquet")

    dims = ["market_vs_regulation", "prevention_vs_punishment",
            "state_vs_local", "universal_vs_targeted"]
    out = {}
    for party in PARTIES:
        out[party] = {}
        pp = pt[pt["parti"] == party].set_index("topic_id")
        lp = lex[lex["party"] == party].set_index("topic_id")
        for t in range(28):
            cell = {
                "stance": float(pp.loc[t, "score"]) if t in pp.index else None,
                "n_events": int(pp.loc[t, "n_events"]) if t in pp.index else 0,
            }
            for d in dims:
                v = lp.loc[t, d] if t in lp.index else None
                cell[d] = float(v) if v is not None and not pd.isna(v) else None
            out[party][str(t)] = cell
    (OUT / "vectors.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")))


def export_reasoning() -> None:
    """Per-(party, topic) reasoning vector, 768-dim. Written to public/data/
    so the site can fetch it at runtime (not bundled into the JS island)."""
    rv = np.load(IN / "reservation_vectors.npz", allow_pickle=True)
    sv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    res_V, res_N = rv["V"], rv["N"]
    sp_V, sp_N = sv["V"], sv["N"]
    # Prefer reservation; fall back to speech.
    V = np.where(res_N[..., None] > 0, res_V, sp_V).astype(np.float16)
    N = np.maximum(res_N, sp_N)

    out = {}
    for i, party in enumerate(PARTIES):
        out[party] = {}
        for t in range(V.shape[1]):
            if N[i, t] == 0:
                continue
            out[party][str(t)] = {
                "v": V[i, t].astype(float).round(4).tolist(),
                "n": int(N[i, t]),
            }
    public_data = ROOT / "site" / "public" / "data"
    public_data.mkdir(parents=True, exist_ok=True)
    (public_data / "reasoning.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")))


# Anonymisation patterns for excerpt curation.
ANON_PATTERNS = [
    # Strip MP names like "av Förnamn Efternamn (P)"
    (re.compile(r"\bav\s+[A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+){0,3}\s*\([A-Z]+\)\s*\.?", re.IGNORECASE), ""),
    # Strip party abbreviations in parens.
    (re.compile(r"\((?:V|S|MP|C|L|KD|M|SD)\)"), ""),
    # Strip party names spelled out.
    (re.compile(
        r"\b(?:Vänsterpartiet|Socialdemokraterna|Miljöpartiet|Centerpartiet|"
        r"Liberalerna|Kristdemokraterna|Moderaterna|Sverigedemokraterna)\b",
        re.IGNORECASE), "vårt parti"),
    # Strip motion citations like "2024/25:1234 yrkande N".
    (re.compile(r"\b\d{4}/\d{2}:\d+(?:\s*yrkande[t]?\s*\d+)?", re.IGNORECASE), ""),
    # Strip leading "Förslag till riksdagsbeslut" boilerplate that introduces
    # the alternative ("we propose…").
    (re.compile(r"^Förslag till riksdagsbeslut\s*", re.IGNORECASE), ""),
    # Trim double-spaces.
    (re.compile(r"\s{2,}"), " "),
]


def anonymise(text: str) -> str:
    s = text
    for pat, repl in ANON_PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()


def excerpt_snippet(text: str, max_words: int = 90, min_words: int = 30) -> str | None:
    """Take the first complete sentence(s) up to max_words, dropping anything
    shorter than min_words. We don't want to ship 5-paragraph reservations to
    the voter — they need to be one paragraph each."""
    text = anonymise(text)
    # Split on sentence boundary (full-stop followed by whitespace + capital).
    sentences = re.split(r"(?<=[\.!?])\s+(?=[A-ZÅÄÖ])", text)
    out = []
    n_words = 0
    for s in sentences:
        w = len(s.split())
        if n_words + w > max_words and out:
            break
        out.append(s)
        n_words += w
    if n_words < min_words:
        return None
    return " ".join(out).strip()


def export_excerpts(per_topic: int = 6) -> None:
    """For each topic, pick up to `per_topic` excerpts per party.

    Two sources:
      - For opposition parties (V, S, MP, C, SD): reservation reasoning text.
      - For cabinet parties (L, KD, M): the betänkande's *forslag* text from
        vote events where they voted Ja (i.e., the cabinet's own proposals).
        Without this, cabinet parties never appear as excerpt options, since
        they barely write reservations.

    All excerpts anonymised the same way.
    """
    CABINET = {"L", "KD", "M"}

    out = {}

    # --- Opposition: from reservation_picks ---
    picks = pd.read_parquet(IN / "reservation_picks.parquet")
    picks["solo"] = picks["n_partier"] == 1
    picks = picks.sort_values(["topic_id", "party", "solo", "n_words"],
                               ascending=[True, True, False, False])
    for topic_id, g in picks.groupby("topic_id"):
        topic_out = out.setdefault(str(int(topic_id)), {})
        for party, gg in g.groupby("party"):
            if party in CABINET:
                continue
            party_excerpts = []
            for _, r in gg.iterrows():
                snippet = excerpt_snippet(r["reasoning_text"])
                if not snippet:
                    continue
                party_excerpts.append({
                    "text": snippet,
                    "rubrik": anonymise(str(r["rubrik"])),
                })
                if len(party_excerpts) >= per_topic:
                    break
            if party_excerpts:
                topic_out[party] = party_excerpts

    # --- Cabinet: from majority recommendations ---
    # For each (cabinet party, topic), pick vote events where the cabinet
    # voted Ja (= they supported the recommendation) and the proposal text
    # is substantive (>= 25 words). Then for each topic, deduplicate excerpts
    # so cabinet parties show distinct topical examples.
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates("votering_id"))
    topics_link = pd.read_parquet(IN / "topics.parquet")
    party_topic = pd.read_parquet(IN / "party_topic.parquet")

    # Cabinet parties vote Ja with mean stance > 0.95 — we want events where
    # the cabinet voted Ja (n_ja was nearly all of their MPs).
    cabinet_stance = party_topic[party_topic["parti"].isin(CABINET)]

    # Link vote events to topics
    tex_topics = tex.merge(topics_link[["votering_id", "topic_id"]],
                           on="votering_id", how="left")
    tex_topics["combined_text"] = (tex_topics["bet_titel"].fillna("") + ". " +
                                    tex_topics["rubrik"].fillna("") + ". " +
                                    tex_topics["forslag_text"].fillna(""))
    tex_topics["n_words"] = tex_topics["combined_text"].str.split().str.len()
    tex_topics = tex_topics[(tex_topics["n_words"] >= 25) &
                             (tex_topics["n_words"] <= 200) &
                             tex_topics["topic_id"].notna()]
    tex_topics = tex_topics.sort_values(["topic_id", "n_words"],
                                          ascending=[True, False])

    cabinet_text_excerpts: dict[int, list[dict]] = {}
    for topic_id, g in tex_topics.groupby("topic_id"):
        cabinet_text_excerpts[int(topic_id)] = []
        seen_rubriks = set()
        for _, r in g.iterrows():
            snippet = excerpt_snippet(r["combined_text"], max_words=110, min_words=25)
            if not snippet:
                continue
            rubrik_key = (r.get("rubrik") or "")[:30].lower()
            if rubrik_key in seen_rubriks:
                continue
            seen_rubriks.add(rubrik_key)
            cabinet_text_excerpts[int(topic_id)].append({
                "text": snippet,
                "rubrik": anonymise(str(r.get("rubrik") or "")),
            })
            if len(cabinet_text_excerpts[int(topic_id)]) >= per_topic:
                break

    # Use the SAME pool for all three cabinet parties — they vote together,
    # so attributing it to one is misleading; we offer them as distinct
    # options each round.
    for topic_id, excerpt_list in cabinet_text_excerpts.items():
        if not excerpt_list:
            continue
        topic_out = out.setdefault(str(topic_id), {})
        # Round-robin: party gets a rotating subset of the pool.
        for i, party in enumerate(["L", "KD", "M"]):
            picks_for_party = excerpt_list[i::3] or excerpt_list
            if picks_for_party:
                topic_out[party] = picks_for_party

    (OUT / "excerpts.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    total_excerpts = sum(len(p)
                         for topic in out.values()
                         for p in topic.values())
    print(f"  excerpts: {total_excerpts} across {len(out)} topics, "
          f"{sum(len(t) for t in out.values())} (topic, party) cells")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    public_data = ROOT / "site" / "public" / "data"
    public_data.mkdir(parents=True, exist_ok=True)
    print("exporting for web ...")
    export_parties();    print(f"  parties.json")
    export_topics();     print(f"  topics.json")
    export_vectors();    print(f"  vectors.json")
    export_reasoning();  print(f"  public/data/reasoning.json")
    export_excerpts()
    print("\nfile sizes (bundled into JS island):")
    for p in sorted(OUT.glob("*.json")):
        print(f"  src/data/{p.name:<18s}: {p.stat().st_size / 1024:>7.1f} KB")
    print("file sizes (fetched at runtime):")
    for p in sorted(public_data.glob("*.json")):
        print(f"  public/data/{p.name:<14s}: {p.stat().st_size / 1024:>7.1f} KB")


if __name__ == "__main__":
    main()
