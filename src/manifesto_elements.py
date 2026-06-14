"""Extract the four political-process elements from manifestos:
   Diagnosis  — what does the party identify as wrong with Sweden now?
   End state  — what does the party want Sweden to be?
   Mechanism  — how does the party propose to get from here to there?
   Values     — which underlying values does the party invoke?

The element classifier is keyword-based — fast, transparent, easy to audit.
Each sentence is scored on each element by matching against a curated
regex pool. A sentence gets the element with the highest score, or
"other" if no element scores above threshold.

Outputs:
    data/processed/manifesto_sentences.parquet — original sentences plus
        element label, scores, and value-word hits
    data/processed/manifesto_party_profile.parquet — per-party aggregates
        for the four-element view
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

# ============================================================================
# Element classifier — keyword-based per element.
# ============================================================================

# DIAGNOSIS markers: language describing the current problematic state.
# Negative-framed, often past-tense observations.
DIAGNOSIS_PATTERNS = [
    # Present-state observation
    r"\b(kris|problem|brist|misslyckande)\w*",
    r"\b(otrygg|otryggt|orättvis|orättvist|ojämlik|ojämlikt|oacceptabel|oacceptabelt)\w*",
    r"\b(försämr|försvag|försvun|urholk|skenat|skenade|skenar|sliter|slitit)\w*",
    r"\b(har blivit|har gått åt fel|har lett till|håller på att)\b",
    r"\b(allt fler|allt mer|allt färre|allt mindre)\b",
    r"\b(sverige (är|har blivit|står inför))\b",
    r"\b(klyfto|spricka|sönderfall|splittring|polarisering)\w*",
    r"\b(misslyckats|stannat upp|går baklänges|backat)\b",
    r"\b(hot|hota[tr])\b",
    r"\b(för (många|få|länge|svår|dyr|liten|stor))\b",
    # Comparative critique
    r"\b(sämre|färre|fler|större|mindre|dyrare|längre)\s+än\b",
    # Past-tense degradation
    r"\b(har vuxit|har ökat|har minskat|har stigit|har fallit)\b",
    # Direct statements of failure
    r"\b(det är inte rimligt|det är inte värdigt|det duger inte)\b",
]

# END STATE markers: aspirational future-state statements.
# "We want Sweden to be …" / "Sweden should be …"
END_STATE_PATTERNS = [
    r"\b(vi vill (se|ha|bygga|skapa)|vi strävar efter|vår vision|vårt mål)\b",
    r"\bsverige ska (vara|bli|ha)\b",
    r"\b(ska kunna|ska få|borde få|borde kunna)\b",
    r"\b(framtid|framtiden|kommande generation)\w*",
    r"\b(ett land där|ett samhälle där|ett sverige där)\b",
    r"\b(vi tror på|vi drömmer om|vi ser fram)\b",
    r"\b(välfärd|frihet|trygghet|jämlikhet|jämställdhet|hållbarhet)\w*\s+för (alla|hela)\b",
    r"\b(generation|barnbarn|våra barn)\w*\s+ska\b",
    r"\b(ingen ska behöva|alla ska kunna|var och en)\b",
    # Positive vision framing
    r"\b(blomstrande|växande|stark[t]?|välmående)\b",
]

# MECHANISM markers: concrete policy proposals — "we will …".
MECHANISM_PATTERNS = [
    # Action verbs in future/proposal tense
    r"\b(vi ska|vi vill (införa|höja|sänka|ändra|reformera|förbjuda|skärpa|stärka|återinföra))\b",
    r"\bregeringen (bör|ska|måste)\b",
    r"\briksdagen (bör|ska|tillkännage)\b",
    # Specific instruments
    r"\b(höja|sänka|avskaffa|införa)\s+(skatten|skatter|avgift|moms)",
    r"\b(ny lag|lagändring|lagstift|reglera|tillsyn|kontroll)\w*",
    r"\b(förbjud|tillåt|bevilja|avslå)\w*",
    r"\b(satsa|investera|anslå|fördubbla|tredubbla)\b",
    # Numeric/policy specifics
    r"\b\d+\s+(miljard|miljon|procent|kr|kronor)\b",
    r"\b(myndighet|ombudsman|nämnd)\w*",
    # Reform language
    r"\b(reform|omläggning|systemskifte|paradigmskifte)\w*",
    r"\b(öka|minska|stärka|försvaga|bygga ut|trappa upp)\b",
    # Specific Swedish policy verbs
    r"\b(återinför|återupprätta|återställ|återinrätta)\w*",
    r"\b(avskaffa|avveckla|fasa ut)\w*",
]

# VALUES dictionary: each value gets a regex pattern. Frequency analysis.
# Built to cover the core Swedish political-rhetoric value vocabulary.
VALUE_PATTERNS: dict[str, str] = {
    "frihet": r"\b(frihet|fri[tt]?\s|frivillig|frigörelse)\w*",
    "jämlikhet": r"\b(jämlik|jämlikhet|likvärdig|lika villkor)\w*",
    "jämställdhet": r"\b(jämställd|jämställdhet|kvinnor och män|könsmakt)\w*",
    "trygghet": r"\b(trygg|trygghet|säkerhet|säker(t|a)?\s)\w*",
    "rättvisa": r"\b(rättvis|rättvisa|rätt(en|igheter)?)\w*",
    "solidaritet": r"\b(solidari|gemensam|tillsammans|sammanhållning)\w*",
    "ansvar": r"\b(ansvar|ansvarstagande|ansvarsfull)\w*",
    "demokrati": r"\b(demokrati|demokratisk|folkstyre)\w*",
    "ordning": r"\b(ordning|ordnat|reda|kontroll|disciplin)\w*",
    "tradition": r"\b(tradition|traditionell|kulturarv|historia|rötter)\w*",
    "framsteg": r"\b(framsteg|utveckling|innovation|modern|framtidstro)\w*",
    "valfrihet": r"\b(valfrihet|val[fm]öjlighet|välja)\w*",
    "individens": r"\b(individ|enskild|personlig)\w*",
    "kollektivets": r"\b(kollektiv|gemenskap|samhället|välfärden)\w*",
    "klimat": r"\b(klimat|miljö|hållbar|fossilfri|biologisk mångfald)\w*",
    "tillväxt": r"\b(tillväxt|välstånd|jobb|sysselsätt|näringsliv)\w*",
    "nationell_identitet": r"\b(svensk[t]?|sverige[s]?|nationell|identitet)\w*",
    "öppenhet": r"\b(öppen|öppenhet|tolerans|inkluderande)\w*",
    "marknaden": r"\b(marknad|konkurrens|företag(samhet)?|entreprenör)\w*",
    "staten": r"\b(staten|nationell|centralt|regeringen|samhället)\w*",
}

# Compile once.
_DIAGNOSIS_RX = [re.compile(p, re.IGNORECASE) for p in DIAGNOSIS_PATTERNS]
_END_STATE_RX = [re.compile(p, re.IGNORECASE) for p in END_STATE_PATTERNS]
_MECHANISM_RX = [re.compile(p, re.IGNORECASE) for p in MECHANISM_PATTERNS]
_VALUE_RX = {k: re.compile(p, re.IGNORECASE) for k, p in VALUE_PATTERNS.items()}


def score_sentence(sentence: str) -> dict:
    """Return raw element scores and value hits for a sentence."""
    d = sum(1 for rx in _DIAGNOSIS_RX if rx.search(sentence))
    e = sum(1 for rx in _END_STATE_RX if rx.search(sentence))
    m = sum(1 for rx in _MECHANISM_RX if rx.search(sentence))
    values = {k: len(rx.findall(sentence)) for k, rx in _VALUE_RX.items()}
    return {
        "diagnosis_score": d,
        "end_state_score": e,
        "mechanism_score": m,
        "values": values,
    }


def classify(sentence: str) -> str:
    """Assign an element label. 'other' if no clear winner."""
    s = score_sentence(sentence)
    scores = {
        "diagnosis": s["diagnosis_score"],
        "end_state": s["end_state_score"],
        "mechanism": s["mechanism_score"],
    }
    top = max(scores.items(), key=lambda kv: kv[1])
    if top[1] == 0:
        return "other"
    # If tied, return the first by precedence: end_state > diagnosis > mechanism
    # because end-state is rarer and more diagnostic when it appears.
    if list(scores.values()).count(top[1]) > 1:
        for k in ("end_state", "diagnosis", "mechanism"):
            if scores[k] == top[1]:
                return k
    return top[0]


def classify_all(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Add element label and score columns to a sentence-level frame."""
    rows = []
    value_keys = list(VALUE_PATTERNS.keys())
    for _, r in sentences_df.iterrows():
        s = score_sentence(r["sentence"])
        row = {
            "diagnosis_score": s["diagnosis_score"],
            "end_state_score": s["end_state_score"],
            "mechanism_score": s["mechanism_score"],
            "element": classify(r["sentence"]),
        }
        for k in value_keys:
            row[f"v_{k}"] = s["values"][k]
        rows.append(row)
    annotated = pd.concat([
        sentences_df.reset_index(drop=True),
        pd.DataFrame(rows),
    ], axis=1)
    return annotated


def build_party_profile(annotated: pd.DataFrame) -> pd.DataFrame:
    """Per-party aggregates of the four-element view."""
    value_cols = [c for c in annotated.columns if c.startswith("v_")]
    rows = []
    for party, g in annotated.groupby("party"):
        n_total = len(g)
        element_counts = g["element"].value_counts()
        row = {
            "party": party,
            "n_sentences": n_total,
            "n_diagnosis": int(element_counts.get("diagnosis", 0)),
            "n_end_state": int(element_counts.get("end_state", 0)),
            "n_mechanism": int(element_counts.get("mechanism", 0)),
            "n_other": int(element_counts.get("other", 0)),
            "share_diagnosis": float(element_counts.get("diagnosis", 0) / n_total),
            "share_end_state": float(element_counts.get("end_state", 0) / n_total),
            "share_mechanism": float(element_counts.get("mechanism", 0) / n_total),
        }
        # Per-value totals, normalised by total words in the manifesto.
        n_words = g["n_words"].sum()
        for c in value_cols:
            row[c] = int(g[c].sum())
            row[c + "_per_kw"] = float(g[c].sum() / n_words * 1000) if n_words else 0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    sentences = pd.read_parquet(IN / "manifestos.parquet")
    print(f"loaded {len(sentences):,} sentences across {sentences['party'].nunique()} parties")

    print("classifying ...")
    annotated = classify_all(sentences)
    print("element distribution:")
    print(annotated.groupby(["party", "element"]).size().unstack(fill_value=0))

    profile = build_party_profile(annotated)
    print("\nshares of element labels (%) per party:")
    print((profile[["party", "share_diagnosis", "share_end_state",
                     "share_mechanism"]]
            .set_index("party") * 100).round(1).to_string())

    annotated.to_parquet(OUT / "manifesto_sentences.parquet", index=False)
    profile.to_parquet(OUT / "manifesto_party_profile.parquet", index=False)
    print(f"\nwrote {OUT}/manifesto_sentences.parquet, manifesto_party_profile.parquet")


if __name__ == "__main__":
    main()
