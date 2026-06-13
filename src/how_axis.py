"""The HOW axis — theory of change.

Two complementary signals:

(A) Reservation co-authorship.
    Each betänkande punkt may have reservations signed by party-sets like
    {S,V,MP}. Two parties co-signing means they propose the same alternative
    mechanism. We compute pairwise cosine similarity in reservation-sign
    space — a structural HOW signal that comes for free from the parsed XML.

(B) Curated theory-of-change lexicon.
    Four mechanism dimensions, each a pair of small Swedish vocabularies.
    For each speech in reasoning_speeches.parquet we score each dimension:
        score = (n_left_terms - n_right_terms) / (n_left + n_right)
        ∈ [-1, +1].
    Then aggregate per (party, topic). The four dimensions are chosen for
    coverage of Swedish political debate; they are explicit so they can be
    revised. Loaded-ness of these choices is acknowledged in the diary.

Outputs:
    data/processed/how_coauthor.npz     — party-party co-authorship sim matrix
    data/processed/how_lexicon.parquet  — per (party, topic) dimension scores
    data/processed/how_pair_sim.parquet — pair-level HOW similarity (combined)
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]

# ---------------------------------------------------------------------------
# Curated theory-of-change lexicon. Each dimension is (left_terms, right_terms).
# "Left" / "right" are just labels for the two sides of the axis — not ideology.
# Wordlists are deliberately short and well-known; revise in this file.
# ---------------------------------------------------------------------------
LEXICON: dict[str, tuple[list[str], list[str]]] = {
    "market_vs_regulation": (
        # Market / choice / private provision side
        ["marknad", "marknaden", "marknader", "marknadens",
         "konkurrens", "konkurrenskraft", "konkurrera",
         "valfrihet", "valfriheten", "fritt val",
         "privata", "privat", "privatisering",
         "mångfald av aktörer", "aktörer",
         "sänkt skatt", "sänka skatten", "skattesänkning", "skattesänkningar",
         "avreglering", "avregler", "regelförenkling",
         "entreprenör", "entreprenörer", "näringsfrihet"],
        # Regulation / state intervention side
        ["lagstifta", "lagstiftning", "ny lag", "lagar",
         "reglera", "reglering", "regleringar",
         "förbjuda", "förbud", "förbjudet",
         "tillsyn", "tillsynsmyndighet",
         "kontroll", "kontrollera", "övervaka",
         "höjd skatt", "höja skatten", "skattehöjning",
         "statliga åtgärder", "hårdare regler", "skarpare regler"],
    ),
    "prevention_vs_punishment": (
        # Prevention / social-cause side
        ["förebyggande", "förebygga", "tidiga insatser", "tidigt",
         "sociala insatser", "social bakgrund", "bakomliggande orsaker",
         "rehabilitering", "behandling", "vård",
         "stöd till", "stödja", "stötta",
         "riskfaktorer", "skyddsfaktorer", "trygghetsskapande",
         "skolan", "föräldraskap", "fritidsgård"],
        # Punishment / deterrence side
        ["straff", "straffet", "skärpta straff", "strängare straff",
         "påföljd", "påföljder", "skärpta påföljder",
         "fängelse", "fängelsestraff", "anstalt",
         "hårdare tag", "tuffare tag",
         "strängare", "skärpa", "skärpning",
         "tvångsmedel", "frihetsberövande",
         "utvisning", "utvisa", "avvisa"],
    ),
    "state_vs_local": (
        # State / centralisation side
        ["staten", "statlig", "statliga", "statligt",
         "nationell", "nationella", "nationellt",
         "regeringen ska", "centralt", "centraliser",
         "riksdagen beslutar", "myndighet", "myndigheter",
         "enhetlig", "likvärdighet"],
        # Municipal / decentralisation side
        ["kommunen", "kommunerna", "kommunalt", "kommuner",
         "regionen", "regionerna", "regionalt",
         "lokalt", "lokala", "lokal",
         "civilsamhället", "ideell sektor", "föreningsliv",
         "närhetsprincipen", "subsidiaritet"],
    ),
    "universal_vs_targeted": (
        # Universalism side
        ["generell", "generella", "generellt", "universellt",
         "alla", "för alla", "till alla",
         "allmän", "allmänt",
         "lika villkor", "samma villkor",
         "välfärden", "gemensam välfärd"],
        # Targeted / means-tested side
        ["behovsprövad", "behovsprövning", "behovsprövat",
         "riktade insatser", "riktad", "riktade",
         "för dem som behöver", "prioritera",
         "särskilda insatser", "särskilt utsatta",
         "selekterad", "selektiv", "selektivt"],
    ),
}


# ---------------------------------------------------------------------------
# (A) Co-authorship
# ---------------------------------------------------------------------------

def parse_party_sets(s: str) -> list[tuple[str, ...]]:
    if not isinstance(s, str) or not s:
        return []
    out = []
    for grp in s.split(";"):
        ps = tuple(sorted(p.strip() for p in grp.split(",")
                          if p.strip() in PARTIES))
        if ps:
            out.append(ps)
    return out


def coauthorship_matrix() -> np.ndarray:
    tex = (pd.read_parquet(IN / "vote_event_texts.parquet")
             .drop_duplicates("votering_id"))
    n_sign = Counter()  # how often each party signs anything
    n_co = Counter()    # how often each pair co-signs
    for s in tex["reservation_partier"].dropna():
        for party_set in parse_party_sets(s):
            for p in party_set:
                n_sign[p] += 1
            for a, b in combinations(party_set, 2):
                n_co[(a, b)] += 1
                n_co[(b, a)] += 1

    # Cosine similarity in reservation-sign space: each party is a vector over
    # vote events, 1 if it signed a reservation, 0 otherwise. We approximate
    # the inner product directly with n_co, and use n_sign as the norm.
    M = np.zeros((len(PARTIES), len(PARTIES)), dtype=np.float64)
    for i, a in enumerate(PARTIES):
        for j, b in enumerate(PARTIES):
            if i == j:
                M[i, j] = 1.0
                continue
            denom = (n_sign[a] * n_sign[b]) ** 0.5
            M[i, j] = (n_co[(a, b)] / denom) if denom else 0.0
    return M, dict(n_sign), dict(n_co)


# ---------------------------------------------------------------------------
# (B) Lexicon scoring
# ---------------------------------------------------------------------------

def _build_pattern(terms: list[str]) -> re.Pattern:
    # Sort longest-first so multi-word phrases match before their substrings.
    terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in terms]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def lexicon_scores() -> pd.DataFrame:
    speeches = pd.read_parquet(IN / "reasoning_speeches.parquet")
    # Each row's text has length, party, betänkande_topic.
    patterns = {k: (_build_pattern(L), _build_pattern(R))
                for k, (L, R) in LEXICON.items()}
    rows = []
    for _, sp in speeches.iterrows():
        text = (sp["text"] or "").lower()
        row = {"party": sp["parti"],
               "topic_id": sp["betänkande_topic"],
               "anforande_id": sp["anforande_id"]}
        for dim, (lp, rp) in patterns.items():
            n_l = len(lp.findall(text))
            n_r = len(rp.findall(text))
            row[f"{dim}_l"] = n_l
            row[f"{dim}_r"] = n_r
            row[f"{dim}_score"] = ((n_l - n_r) / (n_l + n_r)
                                    if (n_l + n_r) > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_cell(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (party, topic) mean of each dimension's signed score. Word-count
    weighted via the sum-of-counts numerator so noisier speeches don't dominate.
    """
    dims = list(LEXICON.keys())
    out_rows = []
    for (party, topic), g in scores.groupby(["party", "topic_id"]):
        row = {"party": party, "topic_id": int(topic),
               "n_speeches": len(g)}
        for d in dims:
            l_sum = g[f"{d}_l"].sum()
            r_sum = g[f"{d}_r"].sum()
            row[f"{d}_n"] = int(l_sum + r_sum)
            row[d] = ((l_sum - r_sum) / (l_sum + r_sum)
                      if (l_sum + r_sum) > 0 else np.nan)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def party_dim_mean(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Per-party mean over topics, weighted by topic n. Provides an at-a-glance
    HOW posture per party for the heatmap."""
    dims = list(LEXICON.keys())
    out = {}
    for party, g in per_cell.groupby("party"):
        rec = {}
        for d in dims:
            weights = g[f"{d}_n"]
            vals = g[d]
            mask = vals.notna() & (weights > 0)
            if mask.sum() == 0:
                rec[d] = np.nan
            else:
                rec[d] = float(np.average(vals[mask], weights=weights[mask]))
            rec[f"{d}_n"] = int(weights[mask].sum())
        out[party] = rec
    return pd.DataFrame(out).T.loc[PARTIES]


# ---------------------------------------------------------------------------
# (C) Pair similarity in HOW space
# ---------------------------------------------------------------------------

def pair_how_sim(per_cell: pd.DataFrame, coauthor: np.ndarray) -> pd.DataFrame:
    """For each party pair, compute:
        coauthor_sim — cosine in reservation-sign space
        lex_sim      — 1 − (mean absolute difference) of lexicon vectors
                       across topics, averaged over dimensions
        how_sim      — average of the two, after rescaling each to [0,1]
    """
    dims = list(LEXICON.keys())
    cell_pivot = {d: per_cell.pivot(index="party", columns="topic_id", values=d)
                     .reindex(index=PARTIES)
                  for d in dims}

    rows = []
    for i, a in enumerate(PARTIES):
        for j, b in enumerate(PARTIES):
            if j <= i:
                continue
            lex_sims = []
            for d in dims:
                pa, pb = cell_pivot[d].loc[a], cell_pivot[d].loc[b]
                mask = pa.notna() & pb.notna()
                if mask.sum() == 0:
                    continue
                mad = (pa[mask] - pb[mask]).abs().mean()
                lex_sims.append(1 - mad / 2.0)
            lex_sim = float(np.mean(lex_sims)) if lex_sims else np.nan
            rows.append({"a": a, "b": b,
                         "coauthor_sim": float(coauthor[i, j]),
                         "lex_sim": lex_sim})
    df = pd.DataFrame(rows)
    # Each on [0,1]-ish — normalise.
    df["coauthor_z"] = (df["coauthor_sim"] - df["coauthor_sim"].min()) / \
                      (df["coauthor_sim"].max() - df["coauthor_sim"].min())
    df["lex_z"] = (df["lex_sim"] - df["lex_sim"].min()) / \
                  (df["lex_sim"].max() - df["lex_sim"].min())
    df["how_sim"] = (df["coauthor_z"] + df["lex_z"]) / 2.0
    return df


def main() -> None:
    print("(A) Reservation co-authorship ...")
    coauthor, n_sign, n_co = coauthorship_matrix()
    coauth_df = pd.DataFrame(coauthor, index=PARTIES, columns=PARTIES)
    print(coauth_df.round(2).to_string())
    print("\nparty solo signing counts:")
    for p in PARTIES:
        print(f"  {p:>3}: signed any={n_sign.get(p, 0):>4}")

    print("\n(B) Lexicon scoring on reasoning speeches ...")
    scores = lexicon_scores()
    per_cell = aggregate_per_cell(scores)
    print(f"  per-(party, topic) cells: {len(per_cell)}")
    party_mean = party_dim_mean(per_cell)
    print("\nMean dimension scores per party (signed in [-1,+1]):")
    print(party_mean[list(LEXICON.keys())].round(2).to_string())

    print("\n(C) Pair HOW similarity ...")
    pair = pair_how_sim(per_cell, coauthor)
    print(pair.sort_values("how_sim", ascending=False).round(3).to_string(index=False))

    np.savez_compressed(OUT / "how_coauthor.npz",
                        M=coauthor,
                        parties=np.array(PARTIES, dtype=object))
    per_cell.to_parquet(OUT / "how_lexicon.parquet", index=False)
    pair.to_parquet(OUT / "how_pair_sim.parquet", index=False)
    print(f"\nwrote {OUT}/how_coauthor.npz, how_lexicon.parquet, how_pair_sim.parquet")


if __name__ == "__main__":
    main()
