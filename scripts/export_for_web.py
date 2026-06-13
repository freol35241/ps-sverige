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

# Manually-curated short labels and per-topic question framings for the
# voter tool. The label is the topic name shown above each question; the
# question is the concrete framing the voter sees. Reviewed against
# meta.label_terms in topic_meta.parquet.
TOPIC_INFO = {
    0:  ("Yttrandefrihet och digitalisering",
         "Yttrandefrihet och digital integritet"),
    1:  ("Kommuner och regioner",
         "Stat, regioner och kommuner — vem ska bestämma vad?"),
    2:  ("Arbetsrätt och arbetslöshet",
         "Arbetsrätt och arbetslöshetsförsäkring"),
    3:  ("Brott och kriminalvård",
         "Hur ska Sverige hantera brott och kriminalvård?"),
    4:  ("Försvar och krisberedskap",
         "Försvar och samhällets krisberedskap"),
    5:  ("Regelförenkling och företagande",
         "Företagande och regelförenkling"),
    6:  ("Skatter",
         "Hur ska skatterna se ut?"),
    7:  ("Skola och lärare",
         "Hur ska den svenska skolan styras och finansieras?"),
    8:  ("Trafik och fordon",
         "Trafik och fordon — vilken inriktning ska politiken ha?"),
    9:  ("Internationellt bistånd",
         "Sveriges internationella bistånd"),
    10: ("Miljö, jakt och jordbruk",
         "Miljö, jakt och jordbruk"),
    11: ("Läkemedel och tandvård",
         "Läkemedel och tandvård"),
    12: ("Socialtjänst, barn och familj",
         "Socialtjänst, barn och familjepolitik"),
    13: ("Migration och asylpolitik",
         "Hur ska migrations- och asylpolitiken utformas?"),
    14: ("Högre utbildning och studiestöd",
         "Högre utbildning och studiestöd"),
    15: ("Nato och utrikespolitik",
         "Sveriges roll i Nato och EU"),
    16: ("Näringspolitik och handel",
         "Näringspolitik och internationell handel"),
    17: ("Kulturpolitik",
         "Hur ska kulturpolitiken se ut?"),
    18: ("Klimat och energi",
         "Klimat och energi — hur ska omställningen göras?"),
    19: ("Brottsbekämpning och övervakning",
         "Brottsbekämpning och övervakning"),
    20: ("Författnings- och valfrågor",
         "Författningsfrågor och hur valet ska gå till"),
    21: ("Bostadspolitik",
         "Bostadspolitik"),
    22: ("Fiske och sjöfart",
         "Fiske och sjöfart"),
    23: ("Hälso- och sjukvård",
         "Hur ska sjukvården organiseras?"),
    24: ("Funktionsnedsättning och äldreomsorg",
         "Stöd vid funktionsnedsättning och äldreomsorg"),
    25: ("Idrott, friluftsliv, alkohol och narkotika",
         "Idrott, friluftsliv, alkohol- och narkotikapolitik"),
    26: ("Finansmarknad och ägande",
         "Finansmarknad och ägarfrågor"),
    27: ("Diskriminering och minoriteter",
         "Diskriminering, jämställdhet och minoritetsfrågor"),
}
TOPIC_LABELS = {k: v[0] for k, v in TOPIC_INFO.items()}
TOPIC_QUESTIONS = {k: v[1] for k, v in TOPIC_INFO.items()}

# Per-topic vocabulary used as a *content* filter on excerpts. An excerpt
# is kept only if it contains at least one of these terms (stems suffice).
# Catches the case where a misclustered vote event drags an off-topic
# reservation into the wrong cluster.
TOPIC_KEYWORDS = {
    0: ["yttrande", "tryck", "digital", "internet", "media", "press", "upphovsrätt", "data", "integritet"],
    1: ["kommun", "region", "landsting"],
    2: ["arbetslös", "arbetsrätt", "arbetsmiljö", "anställ", "facklig", "arbetsförmedling", "lön"],
    3: ["brott", "straff", "påföljd", "kriminal", "polis", "fängels", "narkotika", "vål"],
    4: ["försvar", "krisberedskap", "totalförsvar", "militär", "civilt försvar", "säkerhet"],
    5: ["regelförenkl", "konsument", "företag", "näringsförbud", "marknad"],
    6: ["skatt", "punktskatt", "moms", "mervärdesskatt", "avdrag"],
    7: ["skola", "lärare", "elev", "förskola", "klassrum", "läromedel", "skolpeng", "betyg", "läroplan"],
    8: ["trafik", "fordon", "bil", "cykel", "väg", "körkort", "taxi", "kollektivtrafik"],
    9: ["bistånd", "internationella", "utveckling", "fn", "afrika", "ukraina", "humanitär"],
    10: ["miljö", "djur", "skog", "natur", "jakt", "viltvård", "rovdjur", "klimat"],
    11: ["läkemedel", "tandvård", "apotek", "vaccin"],
    12: ["socialtjänst", "barn", "familj", "föräldra", "vårdnad", "famil"],
    13: ["migration", "asyl", "uppehållstillstånd", "medborgarskap", "invandring", "flykt", "anhörig"],
    14: ["högskola", "universitet", "studiestöd", "csn", "gymnasium", "vuxenutbildning"],
    15: ["nato", "eu", "europeisk", "utrik", "säkerhetspolitik", "försvarsallians"],
    16: ["näring", "handel", "export", "import", "industri", "innovation", "tillväxt"],
    17: ["kultur", "konst", "musik", "litteratur", "bibliotek", "museum", "språk"],
    18: ["klimat", "energi", "el", "kärnkraft", "vindkraft", "fossil", "utsläpp", "koldioxid"],
    19: ["brottsbekämp", "övervakning", "spaning", "tvångsmedel", "polis", "cyber", "terror"],
    20: ["författning", "val", "riksdag", "grundlag", "lobby", "partifinansiering"],
    21: ["bostad", "hyresrätt", "byggande", "hyra", "bostadsförsörj"],
    22: ["fiske", "sjöfart", "hav", "fartyg", "trål"],
    23: ["sjukvård", "vård", "hälso", "patient", "primärvård", "sjukhus", "folkhälsa"],
    24: ["funktionsneds", "äldre", "äldreomsorg", "assistans", "anhörig", "hemtjänst"],
    25: ["idrott", "friluftsliv", "alkohol", "narkotika", "tobak", "spel", "andts", "sport"],
    26: ["finansmarknad", "ägande", "aktie", "bank", "pension", "kapital", "börs"],
    27: ["diskriminering", "minorit", "jämställd", "rasism", "etnisk", "hbtq", "religion"],
}


def matches_topic(text: str, topic_id: int) -> bool:
    keywords = TOPIC_KEYWORDS.get(topic_id, [])
    if not keywords:
        return True
    lower = text.lower()
    return any(k in lower for k in keywords)


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
            "question": TOPIC_QUESTIONS.get(int(t),
                "Vilket resonemang ligger närmast dig?"),
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


# --- Anonymisation and cleaning ---

ANON_PATTERNS = [
    # Soft hyphens left over from PDF→HTML conversion.
    (re.compile(r"\s*­\s*"), ""),
    # Common stray-space-inside-word artefacts after PDF conversion:
    # "U tskottet" → "Utskottet", "R egeringen" → "Regeringen". Only fix
    # capital-letter cases since lowercase single-letter words exist in
    # Swedish ("i", "å") and naive merging breaks them.
    (re.compile(r"\b([A-ZÅÄÖ])\s+([a-zåäö]{3,})\b"), r"\1\2"),
    # Mid-word hyphen-with-spaces: "samhälls­-­ekonomi" (after soft-hyphen
    # stripping this becomes "samhälls - ekonomi").
    (re.compile(r"([a-zåäö])\s+-\s+([a-zåäö])"), r"\1\2"),
    # MP-name + party patterns: "av X Y (P)" or "X Y (P)"
    (re.compile(
        r"\b(?:av\s+)?[A-ZÅÄÖ][a-zåäö]+(?:[\s-][A-ZÅÄÖ][a-zåäö]+){0,3}\s*\((?:V|S|MP|C|L|KD|M|SD|-)(?:\s*,\s*(?:V|S|MP|C|L|KD|M|SD|-))*\)",
        re.IGNORECASE), ""),
    # Party abbreviations in parens like "(V)" or "(M, KD, L)".
    (re.compile(r"\(\s*(?:V|S|MP|C|L|KD|M|SD|-)(?:\s*,\s*(?:V|S|MP|C|L|KD|M|SD|-))*\s*\)"), ""),
    # Party names spelled out — replace with neutral noun.
    (re.compile(
        r"\bSverigedemokraterna(?:s)?\b|\bVänsterpartiet(?:s)?\b|"
        r"\bMiljöpartiet(?:s)?\b|\bCenterpartiet(?:s)?\b|"
        r"\bLiberalerna(?:s)?\b|\bKristdemokraterna(?:s)?\b|"
        r"\bModeraterna(?:s)?\b|\bSocialdemokraterna(?:s)?\b",
        re.IGNORECASE), "vårt parti"),
    # Motion citations like "2024/25:1234 yrkande N" or "yrkandena 1-3".
    (re.compile(r"\b\d{4}/\d{2}:\d+(?:[–-]\d+)?", re.IGNORECASE), ""),
    (re.compile(r"\byrkande(?:na)?\s+\d+(?:\s*[–-]\s*\d+|(?:\s*,\s*\d+)*(?:\s+och\s+\d+)?)", re.IGNORECASE), ""),
    # References to motion "av X y.fl."
    (re.compile(r"\bmotion(?:erna)?\s+av\s+[^.]+?(?=\.|\,|$)", re.IGNORECASE), ""),
    (re.compile(r"\bm\.\s*fl\.", re.IGNORECASE), ""),
    # Procedural boilerplate.
    (re.compile(r"\bDärmed (?:bifaller|avslår) riksdagen[^.]*\.", re.IGNORECASE), ""),
    (re.compile(r"\bRiksdagen (?:bifaller|avslår|antar|godkänner)[^.]*\.", re.IGNORECASE), ""),
    (re.compile(r"\bsom anförs i reservationen och tillkännager detta för regeringen", re.IGNORECASE), ""),
    # The two boilerplate intros to opposition reservations.
    (re.compile(r"^Förslag till riksdagsbeslut\s*\.?\s*", re.IGNORECASE), ""),
    (re.compile(r"^(?:Vi|Jag) anser att förslaget till riksdagsbeslut under punkt \d+ borde ha följande lydelse\s*:?\s*", re.IGNORECASE), ""),
    (re.compile(r"^Riksdagen ställer sig bakom det[^.]*\.\s*", re.IGNORECASE), ""),
    # Trim double-spaces and stray punctuation.
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r"\s+([.,;:!?])"), r"\1"),
    (re.compile(r"^[\s.,;:]+"), ""),
]


def anonymise(text: str) -> str:
    s = text
    for pat, repl in ANON_PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()


# Phrases that signal the excerpt is still procedural even after anonymisation.
JUNK_PHRASES = re.compile(
    r"\b(?:bifaller riksdagen|avslår riksdagen|antar riksdagen|"
    r"behandlade dokument|tillkännager för regeringen|"
    r"yrkandet bör avslås|motionsyrkand|propositionspunkt|"
    r"borde ha följande lydelse|därmed avslår)",
    re.IGNORECASE)
# Cap a single digit count — too many citations = junk.
DIGIT_GROUP = re.compile(r"\d+")


def is_clean(text: str, max_digit_groups: int = 3) -> bool:
    """Heuristic quality check after anonymisation."""
    if JUNK_PHRASES.search(text):
        return False
    if len(DIGIT_GROUP.findall(text)) > max_digit_groups:
        return False
    return True


def excerpt_snippet(text: str, max_words: int = 75, min_words: int = 30) -> str | None:
    """Anonymise + clip to the first complete sentence(s) up to max_words.
    Returns None if the cleaned result is too short or fails quality check."""
    text = anonymise(text)
    if not text:
        return None
    sentences = re.split(r"(?<=[\.!?])\s+(?=[A-ZÅÄÖ])", text)
    out, n_words = [], 0
    for s in sentences:
        w = len(s.split())
        if n_words + w > max_words and out:
            break
        out.append(s)
        n_words += w
        if n_words >= max_words:
            break
    snippet = " ".join(out).strip()
    if len(snippet.split()) < min_words:
        return None
    if not is_clean(snippet):
        return None
    return snippet


def export_excerpts(per_topic: int = 5) -> None:
    """For each topic, pick up to `per_topic` excerpts per party.

    Sources:
      - Opposition parties (V, S, MP, C, SD): text AFTER 'Ställningstagande'
        in their reservations (the substantive argument, not the procedural
        intro). Solo reservations preferred.
      - Cabinet parties (L, KD, M): 'Utskottets ställningstagande' sections
        from the betänkanden — the majority opinion equivalent.

    Topic alignment: we filter excerpts by the source vote event's
    dist_to_centroid (its fit to the KMeans cluster) AND by a small keyword
    check against the topic's expected vocabulary. Misclustered events
    (e.g., a crime reservation tagged as Migration) are dropped before
    they reach the voter.
    """
    CABINET = {"L", "KD", "M"}
    OPP = {"V", "S", "MP", "C", "SD"}

    out = {}

    # --- Opposition: ställningstagande_text from reservations ---
    res = pd.read_parquet(IN / "reservations.parquet")
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates(["dok_id", "punkt"])
             [["dok_id", "punkt", "votering_id"]])
    topics_link = pd.read_parquet(IN / "topics.parquet")
    # Include dist_to_centroid for topic-fit filtering.
    res_linked = (res.merge(tex, on=["dok_id", "punkt"], how="left")
                     .merge(topics_link[["votering_id", "topic_id",
                                          "dist_to_centroid"]],
                            on="votering_id", how="left"))
    res_linked = res_linked[res_linked["topic_id"].notna()]
    res_linked["topic_id"] = res_linked["topic_id"].astype(int)
    # Drop the worst-fitting members of each cluster (likely misclassified).
    thresholds = res_linked.groupby("topic_id")["dist_to_centroid"].quantile(0.85)
    res_linked = res_linked[
        res_linked.apply(
            lambda r: r["dist_to_centroid"] <= thresholds[r["topic_id"]], axis=1)
    ]

    # Expand multi-party reservations to one row per signer.
    rows = []
    for _, r in res_linked.iterrows():
        for p in str(r["partier"]).split(";"):
            if p in OPP:
                rows.append({
                    "party": p,
                    "topic_id": r["topic_id"],
                    "rubrik": r["rubrik"],
                    "text": r["stallningstagande_text"] or "",
                    "n_partier": r["n_partier"],
                    "n_stallnings_words": r["n_stallnings_words"],
                })
    opp_df = pd.DataFrame(rows)
    opp_df = opp_df[opp_df["n_stallnings_words"] >= 40]
    # Prefer solo reservations (n_partier == 1) and longer ställningstagande.
    opp_df["solo"] = (opp_df["n_partier"] == 1).astype(int)
    opp_df = opp_df.sort_values(["topic_id", "party", "solo", "n_stallnings_words"],
                                  ascending=[True, True, False, False])

    n_opp_cells = 0
    n_dropped_misclass = 0
    for (topic_id, party), g in opp_df.groupby(["topic_id", "party"]):
        topic_out = out.setdefault(str(int(topic_id)), {})
        party_excerpts = []
        seen_rubriks = set()
        for _, r in g.iterrows():
            snippet = excerpt_snippet(r["text"])
            if not snippet:
                continue
            # Drop excerpts whose content doesn't match the topic's vocabulary
            # (catches misclustered events).
            if not matches_topic(snippet + " " + (r["rubrik"] or ""), int(topic_id)):
                n_dropped_misclass += 1
                continue
            rubrik_key = (r["rubrik"] or "")[:40].lower().strip()
            if rubrik_key in seen_rubriks:
                continue
            seen_rubriks.add(rubrik_key)
            party_excerpts.append({
                "text": snippet,
                "rubrik": anonymise(str(r["rubrik"] or "")),
            })
            if len(party_excerpts) >= per_topic:
                break
        if party_excerpts:
            topic_out[party] = party_excerpts
            n_opp_cells += 1

    # --- Cabinet: majority_opinions ---
    maj = pd.read_parquet(IN / "majority_opinions.parquet")
    # Need topic_id per majority opinion. dok_id maps to multiple vote events
    # (possibly multiple topics); we attribute to the dominant topic.
    e2t = topics_link.merge(
        tex[["dok_id", "votering_id"]], on="votering_id")
    e2t["dok_id"] = e2t["dok_id"].str.upper()
    bet_topic = (e2t.groupby("dok_id")["topic_id"]
                    .agg(lambda s: s.mode().iat[0])
                    .reset_index())
    maj["dok_id_u"] = maj["dok_id"].str.upper()
    maj_linked = maj.merge(bet_topic, left_on="dok_id_u", right_on="dok_id",
                           how="inner", suffixes=("", "_bet"))
    maj_linked = maj_linked.sort_values(["topic_id", "n_words"],
                                          ascending=[True, False])

    # Lightly recast "Utskottet" voice to "vi" so the cabinet excerpts read
    # like first-person positions, matching the reservation voice.
    def recast(text: str) -> str:
        # Order matters — replace longer phrases first.
        s = re.sub(r"\bUtskottet vill därför\b", "vi vill därför", text)
        s = re.sub(r"\bUtskottet anser därför\b", "vi anser därför", s)
        s = re.sub(r"\bUtskottet välkomnar\b", "vi välkomnar", s)
        s = re.sub(r"\bUtskottet ser\b", "vi ser", s)
        s = re.sub(r"\bUtskottet anser\b", "vi anser", s)
        s = re.sub(r"\bUtskottet menar\b", "vi menar", s)
        s = re.sub(r"\bUtskottet konstaterar\b", "vi konstaterar", s)
        s = re.sub(r"\bUtskottet bedömer\b", "vi bedömer", s)
        s = re.sub(r"\bUtskottet vill\b", "vi vill", s)
        s = re.sub(r"\butskottet[s]?\b", "vi", s, flags=re.IGNORECASE)
        return s

    n_cab_cells = 0
    cabinet_excerpts_by_topic = {}
    for topic_id, g in maj_linked.groupby("topic_id"):
        per_topic_list = []
        seen_rubriks = set()
        for _, r in g.iterrows():
            text = recast(r["text"])
            snippet = excerpt_snippet(text)
            if not snippet:
                continue
            if not matches_topic(snippet + " " + (r["rubrik"] or ""), int(topic_id)):
                n_dropped_misclass += 1
                continue
            rubrik_key = (r["rubrik"] or "")[:40].lower().strip()
            if rubrik_key in seen_rubriks:
                continue
            seen_rubriks.add(rubrik_key)
            per_topic_list.append({
                "text": snippet,
                "rubrik": anonymise(str(r["rubrik"] or "")),
            })
            if len(per_topic_list) >= per_topic * 3:
                break
        cabinet_excerpts_by_topic[int(topic_id)] = per_topic_list

    # Round-robin distribute to L/KD/M.
    for topic_id, excerpt_list in cabinet_excerpts_by_topic.items():
        if not excerpt_list:
            continue
        topic_out = out.setdefault(str(topic_id), {})
        for i, party in enumerate(["L", "KD", "M"]):
            picks_for_party = excerpt_list[i::3]
            if picks_for_party:
                topic_out[party] = picks_for_party[:per_topic]
                n_cab_cells += 1

    (OUT / "excerpts.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    total = sum(len(p) for topic in out.values() for p in topic.values())
    n_cells = sum(len(t) for t in out.values())
    print(f"  excerpts: {total} across {len(out)} topics, "
          f"{n_cells} (topic, party) cells "
          f"(opp={n_opp_cells}, cab={n_cab_cells}, "
          f"misclass dropped={n_dropped_misclass})")


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
