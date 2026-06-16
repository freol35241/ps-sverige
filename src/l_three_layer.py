"""L's three-layer comparison: 2022 manifesto, 2022-26 chamber record, 2026 manifesto.

Liberalerna is the first party to publish a finalised 2026 valmanifest
("För din frihet", 2 June 2026). All other parties are in draft or
unpublished state three months out from the September 2026 election.
This script runs a focused three-layer comparison for L only, using the
same classifier, value vocabulary and SBERT model as the main pipeline.

To reproduce, fetch the source PDF and extract to plain text:
  curl -sSL -o data/manifestos/2026_raw/l-2026.pdf \\
    https://www.liberalerna.se/wp-content/uploads/liberalernas-valmanifest-2026-40s-komprimerad.pdf
  .venv/bin/python -c "import pdfplumber; from pathlib import Path; \\
    text='\\n\\n'.join((p.extract_text() or '') for p in \\
    pdfplumber.open('data/manifestos/2026_raw/l-2026.pdf').pages); \\
    Path('data/manifestos/2026_clean/l-2026.txt').write_text(text, encoding='utf-8')"

Then run:
  .venv/bin/python -m src.l_three_layer

Outputs:
  data/processed/l_2026_sentences.parquet
  figures/web/40_l_element_shares.png
  figures/web/41_l_signature_values.png
  figures/web/42_l_topic_gap_2022_vs_2026.png
  figures/web/43_l_2026_vs_party_reservations.png
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from manifestos import _clean, _sentences  # noqa: E402
from manifesto_elements import classify_all, VALUE_PATTERNS  # noqa: E402

IN = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "web"
PUB = ROOT / "site" / "public" / "figures"

L_2026_PATH = ROOT / "data" / "manifestos" / "2026_clean" / "l-2026.txt"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
L_BLUE = "#006AB3"
L_BLUE_LIGHT = "#7FB5D9"
INK = "#222222"
BG = "#FAFAFA"


def parse_l_2026() -> pd.DataFrame:
    """Sentence-tokenise L's 2026 manifesto using the same logic as 2022."""
    import re

    raw = L_2026_PATH.read_text(encoding="utf-8")
    raw = _clean(raw)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    rows = []
    sent_idx = 0
    for para_idx, para in enumerate(paragraphs):
        para = re.sub(r"\n", " ", para)
        for s in _sentences(para):
            rows.append({
                "party": "l",
                "party_name": "Liberalerna",
                "year": 2026,
                "sentence": s,
                "paragraph_idx": para_idx,
                "sentence_idx": sent_idx,
                "n_words": len(s.split()),
            })
            sent_idx += 1
    df = pd.DataFrame(rows)
    df["position"] = df["sentence_idx"] / (len(df) - 1) if len(df) > 1 else 0.5
    return df


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Run the element classifier and value-frequency counter."""
    return classify_all(df)


def element_shares(annotated: pd.DataFrame) -> dict:
    """Element shares for the supplied annotated frame."""
    n = len(annotated)
    counts = annotated["element"].value_counts()
    return {
        "n_sentences": n,
        "n_words": int(annotated["n_words"].sum()),
        "share_diagnosis": float(counts.get("diagnosis", 0) / n),
        "share_end_state": float(counts.get("end_state", 0) / n),
        "share_mechanism": float(counts.get("mechanism", 0) / n),
        "share_other": float(counts.get("other", 0) / n),
    }


def value_frequencies(annotated: pd.DataFrame) -> dict:
    """Per-1000-word frequency for each value."""
    n_words = annotated["n_words"].sum()
    out = {}
    for v in VALUE_PATTERNS:
        col = f"v_{v}"
        if col in annotated.columns:
            out[v] = float(annotated[col].sum() / n_words * 1000)
    return out


def sbert_embed(sentences: list[str]) -> np.ndarray:
    """SBERT-embed sentences with the same model the rest of the pipeline uses."""
    from sentence_transformers import SentenceTransformer
    print("  loading KBLab/sentence-bert-swedish-cased ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
    print(f"  embedding {len(sentences)} sentences ...")
    embs = model.encode(sentences, batch_size=32, show_progress_bar=False,
                         normalize_embeddings=True).astype(np.float32)
    return embs


# ---------------------------------------------------------------------------
# Topic assignment and gap measurement, mirroring stated_vs_revealed_topic.py
# ---------------------------------------------------------------------------

MIN_SENTENCES_PER_CELL = 5
MIN_TOPIC_SIMILARITY = 0.30


def topic_centroids() -> tuple[np.ndarray, list[int]]:
    npz = np.load(IN / "embed_recommendations.npz", allow_pickle=True)
    X = npz["X"]
    ids = list(npz["votering_id"])
    topics = pd.read_parquet(IN / "topics.parquet")
    id_to_topic = dict(zip(topics["votering_id"], topics["topic_id"]))
    bucket: dict[int, list[int]] = {}
    for i, vid in enumerate(ids):
        t = id_to_topic.get(vid)
        if t is None:
            continue
        bucket.setdefault(int(t), []).append(i)
    topic_ids = sorted(bucket.keys())
    centroids = np.stack([X[bucket[t]].mean(axis=0) for t in topic_ids])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids, topic_ids


def cell_vectors(embs: np.ndarray, sub: pd.DataFrame,
                 centroids: np.ndarray, topic_ids: list[int]
                 ) -> dict[int, np.ndarray]:
    """Per-topic mean embedding for L, using sentences with topic_sim >= threshold."""
    sims = embs @ centroids.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(embs)), best_idx]
    assigned = np.array([topic_ids[i] for i in best_idx])
    out: dict[int, np.ndarray] = {}
    sub = sub.reset_index(drop=True)
    for topic_id in set(assigned):
        mask = (assigned == topic_id) & (best_sim >= MIN_TOPIC_SIMILARITY)
        if mask.sum() < MIN_SENTENCES_PER_CELL:
            continue
        v = embs[mask].mean(axis=0)
        out[int(topic_id)] = v / np.linalg.norm(v)
    return out


def load_l_reservation_vectors() -> tuple[np.ndarray, np.ndarray]:
    """L's per-topic reservation/speech reasoning vectors."""
    rv = np.load(IN / "reservation_vectors.npz", allow_pickle=True)
    sv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    V = np.where(rv["N"][..., None] > 0, rv["V"], sv["V"]).astype(np.float32)
    N = np.maximum(rv["N"], sv["N"])
    return V, N


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_element_shares(shares_2022: dict, shares_2026: dict) -> Path:
    elems = ["diagnosis", "end_state", "mechanism"]
    labels = ["Diagnosis", "End state", "Mechanism"]
    v22 = [shares_2022[f"share_{e}"] * 100 for e in elems]
    v26 = [shares_2026[f"share_{e}"] * 100 for e in elems]

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(elems))
    w = 0.36
    bars_a = ax.bar(x - w / 2, v22, w, label="L 2022", color=L_BLUE_LIGHT,
                    edgecolor="white", linewidth=1)
    bars_b = ax.bar(x + w / 2, v26, w, label="L 2026", color=L_BLUE,
                    edgecolor="white", linewidth=1)
    for bars in (bars_a, bars_b):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f"{b.get_height():.1f} %",
                    ha="center", va="bottom", fontsize=10, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Share of substantive sentences (%)", fontsize=11)
    ax.set_ylim(0, max(v22 + v26) * 1.18 + 1)
    ax.set_title("Liberalerna's manifesto shape, 2022 to 2026\n"
                 "Share of sentences classified into each element register.",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout()
    out = FIG / "40_l_element_shares.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def fig_signature_values(values_2022: dict, values_2026: dict) -> Path:
    # Take the union of top 10 values by 2022 freq + top 10 by 2026 freq.
    top22 = sorted(values_2022.items(), key=lambda kv: -kv[1])[:10]
    top26 = sorted(values_2026.items(), key=lambda kv: -kv[1])[:10]
    keys = []
    for k, _ in top22 + top26:
        if k not in keys:
            keys.append(k)
    # Sort by 2022 frequency (highest first) for readability.
    keys.sort(key=lambda k: -values_2022.get(k, 0))

    v22 = [values_2022.get(k, 0) for k in keys]
    v26 = [values_2026.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, v22, w, label="L 2022", color=L_BLUE_LIGHT,
           edgecolor="white", linewidth=1)
    ax.bar(x + w / 2, v26, w, label="L 2026", color=L_BLUE,
           edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mentions per 1,000 words", fontsize=11)
    ax.set_title("Value vocabulary, L 2022 vs L 2026\n"
                 "Frequency per 1,000 words for each value. Union of the top 10 "
                 "values in each manifesto, sorted by 2022 frequency.",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    out = FIG / "41_l_signature_values.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def fig_topic_gap(gap_2022: float, gap_2026: float,
                  n_topics_2022: int, n_topics_2026: int) -> Path:
    """Two bars: L 2022 vs L 2026, each showing the topic-matched stated-vs-revealed gap."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(2)
    means = [gap_2022, gap_2026]
    counts = [n_topics_2022, n_topics_2026]
    bars = ax.bar(x, means, color=[L_BLUE_LIGHT, L_BLUE],
                  edgecolor="white", linewidth=1.5, width=0.55)
    for b, m, n in zip(bars, means, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{m:.2f}\n(n={n})",
                ha="center", va="bottom", fontsize=11,
                color="#333", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["L 2022 manifesto", "L 2026 manifesto"],
                                          fontsize=11)
    ax.set_ylabel("Mean topic-matched cosine vs L's chamber reasoning")
    ax.set_ylim(0, 1)
    ax.set_title("Does L's manifesto still describe its chamber work?\n"
                 "Higher = the manifesto language matches the parliamentary record on the same topics.",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "42_l_topic_gap_2022_vs_2026.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def fig_l2026_vs_party(party_sims_2026: dict, party_sims_2022: dict) -> Path:
    """How similar is L's 2026 manifesto to each party's chamber reasoning?
    Compare against the same measure for L 2022."""
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    parties = PARTIES
    x = np.arange(len(parties))
    w = 0.36
    v22 = [party_sims_2022.get(p, 0) for p in parties]
    v26 = [party_sims_2026.get(p, 0) for p in parties]
    ax.bar(x - w / 2, v22, w, label="L 2022 manifesto",
           color=L_BLUE_LIGHT, edgecolor="white", linewidth=1)
    bars26 = ax.bar(x + w / 2, v26, w, label="L 2026 manifesto",
                    color=L_BLUE, edgecolor="white", linewidth=1)
    # Highlight which party the 2026 manifesto is most similar to.
    if v26:
        top_idx = int(np.argmax(v26))
        bars26[top_idx].set_edgecolor("#222")
        bars26[top_idx].set_linewidth(2.2)
    ax.set_xticks(x); ax.set_xticklabels(parties, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean topic-matched cosine vs party's chamber reasoning")
    ax.set_ylim(0, max(v22 + v26) * 1.15 + 0.02)
    ax.set_title("L's 2026 manifesto against every party's chamber reasoning\n"
                 "Bar per party: mean within-topic cosine between L's manifesto and "
                 "that party's reservation/speech vectors. The bordered bar is the maximum.",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout()
    out = FIG / "43_l_2026_vs_party_reservations.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)

    # --- 1. Parse and classify the 2026 manifesto ---
    print("== parsing L 2026 ==")
    df26 = parse_l_2026()
    print(f"  {len(df26):,} sentences, {df26['n_words'].sum():,} words")
    print("== classifying L 2026 ==")
    ann26 = annotate(df26)
    ann26.to_parquet(IN / "l_2026_sentences.parquet", index=False)
    shares26 = element_shares(ann26)
    values26 = value_frequencies(ann26)

    # --- 2. Load L 2022 from the existing annotated parquet ---
    print("\n== loading L 2022 ==")
    all22 = pd.read_parquet(IN / "manifesto_sentences.parquet")
    ann22 = all22[all22["party"] == "l"].copy()
    shares22 = element_shares(ann22)
    values22 = value_frequencies(ann22)

    print("\nelement shares (%):")
    print(f"  diagnosis:  2022 {shares22['share_diagnosis']*100:5.1f}   2026 {shares26['share_diagnosis']*100:5.1f}")
    print(f"  end_state:  2022 {shares22['share_end_state']*100:5.1f}   2026 {shares26['share_end_state']*100:5.1f}")
    print(f"  mechanism:  2022 {shares22['share_mechanism']*100:5.1f}   2026 {shares26['share_mechanism']*100:5.1f}")
    print(f"  words:      2022 {shares22['n_words']:>6,}   2026 {shares26['n_words']:>6,}")

    # --- 3. SBERT-embed the 2026 manifesto ---
    print("\n== embedding L 2026 ==")
    sub26 = ann26[ann26["element"] != "other"].reset_index(drop=True)
    embs26 = sbert_embed(sub26["sentence"].tolist())

    # --- 4. Topic assignment, per-topic cells ---
    print("\n== assigning L 2026 sentences to topics ==")
    centroids, topic_ids = topic_centroids()
    cells_2026 = cell_vectors(embs26, sub26, centroids, topic_ids)
    print(f"  L 2026 covers {len(cells_2026)} topics with ≥{MIN_SENTENCES_PER_CELL} on-topic sentences")

    # --- 5. Load L's chamber reasoning vectors ---
    res_V, res_N = load_l_reservation_vectors()
    L_IDX = PARTIES.index("L")

    # --- 6. Per-topic gap, L 2022 ---
    # Load existing L 2022 topic-matched data.
    gap_2022_df = pd.read_parquet(IN / "stated_vs_revealed_topic.parquet")
    gap_2022_l = gap_2022_df[gap_2022_df["party"] == "L"]
    gap_2022_mean = float(gap_2022_l["stated_revealed_cos"].mean())
    n_topics_2022 = int(gap_2022_l.shape[0])
    print(f"\n  L 2022 mean topic-matched gap: {gap_2022_mean:.3f} (n={n_topics_2022})")

    # --- 7. Per-topic gap, L 2026 ---
    gap_2026_sims = []
    for topic_id, manifesto_vec in cells_2026.items():
        if res_N[L_IDX, topic_id] == 0:
            continue
        sim = cos(manifesto_vec, res_V[L_IDX, topic_id])
        gap_2026_sims.append((topic_id, sim))
    gap_2026_mean = float(np.mean([s for _, s in gap_2026_sims])) if gap_2026_sims else 0.0
    n_topics_2026 = len(gap_2026_sims)
    print(f"  L 2026 mean topic-matched gap: {gap_2026_mean:.3f} (n={n_topics_2026})")

    # --- 8. Cross-party: L 2026 vs every party's reservations ---
    party_sims_2026: dict[str, float] = {}
    for p in PARTIES:
        i = PARTIES.index(p)
        sims = []
        for topic_id, manifesto_vec in cells_2026.items():
            if res_N[i, topic_id] == 0:
                continue
            sims.append(cos(manifesto_vec, res_V[i, topic_id]))
        party_sims_2026[p] = float(np.mean(sims)) if sims else 0.0
    print("\n  L 2026 vs each party's chamber reasoning (mean across covered topics):")
    for p, v in sorted(party_sims_2026.items(), key=lambda kv: -kv[1]):
        print(f"    {p}: {v:.3f}")

    # For comparison, compute L 2022 vs every party's reservations
    # (using L 2022 cell vectors — re-extract from existing parquet).
    print("\n  computing L 2022 vs each party for comparison ...")
    l22_assignment = pd.read_parquet(IN / "manifesto_topic_assignment.parquet")
    l22_assignment = l22_assignment[l22_assignment["party"] == "l"]
    # Need embeddings — re-embed L 2022 only (cheap, ~250 sentences).
    sub22 = ann22[ann22["element"] != "other"].copy()
    embs22 = sbert_embed(sub22["sentence"].tolist())
    cells_2022 = cell_vectors(embs22, sub22, centroids, topic_ids)
    party_sims_2022: dict[str, float] = {}
    for p in PARTIES:
        i = PARTIES.index(p)
        sims = []
        for topic_id, manifesto_vec in cells_2022.items():
            if res_N[i, topic_id] == 0:
                continue
            sims.append(cos(manifesto_vec, res_V[i, topic_id]))
        party_sims_2022[p] = float(np.mean(sims)) if sims else 0.0

    # --- 9. Figures ---
    print("\n== rendering figures ==")
    paths = [
        fig_element_shares(shares22, shares26),
        fig_signature_values(values22, values26),
        fig_topic_gap(gap_2022_mean, gap_2026_mean, n_topics_2022, n_topics_2026),
        fig_l2026_vs_party(party_sims_2026, party_sims_2022),
    ]
    for p in paths:
        target = PUB / p.name
        shutil.copyfile(p, target)
        print(f"  wrote {p.name}")

    # Final summary table for the article
    print("\n== summary for the article ==")
    print(f"  L 2022 manifesto: {shares22['n_words']:,} words, {shares22['n_sentences']:,} sentences")
    print(f"  L 2026 manifesto: {shares26['n_words']:,} words, {shares26['n_sentences']:,} sentences")
    print(f"  diagnosis share:  2022 {shares22['share_diagnosis']*100:.1f}%   2026 {shares26['share_diagnosis']*100:.1f}%")
    print(f"  end-state share:  2022 {shares22['share_end_state']*100:.1f}%   2026 {shares26['share_end_state']*100:.1f}%")
    print(f"  mechanism share:  2022 {shares22['share_mechanism']*100:.1f}%   2026 {shares26['share_mechanism']*100:.1f}%")
    print(f"  topic-matched gap vs L chamber: 2022 {gap_2022_mean:.3f}, 2026 {gap_2026_mean:.3f}")
    print(f"  L 2026 most resembles (by chamber reasoning):")
    for p, v in sorted(party_sims_2026.items(), key=lambda kv: -kv[1])[:3]:
        print(f"    {p}: {v:.3f}")
    print(f"  L 2026 value top-5 by per-1000-word frequency:")
    for v, freq in sorted(values26.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {v}: {freq:.2f}")


if __name__ == "__main__":
    main()
