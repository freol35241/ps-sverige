"""Stated vs revealed — the headline manifesto analysis.

For each party we compute two reasoning vectors:
   STATED    — SBERT mean of all "diagnosis + end_state + mechanism" sentences
                from their 2022 manifesto
   REVEALED  — the per-party reasoning vector we already built from their
                reservation texts during the mandate

For each party we compute cosine(stated, revealed): high = manifesto and
parliamentary behaviour use the same register; low = the party speaks
differently when it can afford to (in a manifesto) than when it has to
work inside the chamber (in reservations).

Then we ask the more interesting question: cross-party. For each pair of
parties, is the gap between manifesto-A and reservations-A larger than the
gap between manifesto-A and reservations-B? In other words, when a party
deviates from its own stated position, which other party does its behaviour
look more like?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES_LOWER = ["v", "s", "mp", "c", "l", "kd", "m", "sd"]
PARTIES_UPPER = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}

BG = "#FAFAFA"
INK = "#222222"


def build_stated_vectors() -> dict[str, np.ndarray]:
    """SBERT-embed all 'substantive' (non-other) manifesto sentences per party,
    average to a single party vector."""
    from sentence_transformers import SentenceTransformer

    print("  loading SBERT for manifesto stated vectors ...")
    model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

    ann = pd.read_parquet(IN / "manifesto_sentences.parquet")
    out = {}
    for party in PARTIES_LOWER:
        sub = ann[(ann["party"] == party) & (ann["element"] != "other")]
        if len(sub) < 5:
            print(f"    {party}: too few substantive sentences ({len(sub)}), skipping")
            continue
        embs = model.encode(sub["sentence"].tolist(),
                            batch_size=16, show_progress_bar=False,
                            normalize_embeddings=True)
        out[party] = embs.mean(axis=0)
        print(f"    {party}: stated vector from {len(sub)} sentences")
    return out


def build_revealed_vectors() -> dict[str, np.ndarray]:
    """Reuse the per-party reservation reasoning vectors built earlier. We
    average the per-topic vectors to a single party-level vector."""
    rv = np.load(IN / "reservation_vectors.npz", allow_pickle=True)
    sv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    res_V, res_N = rv["V"], rv["N"]
    sp_V, sp_N = sv["V"], sv["N"]
    # Prefer reservation; fall back to speech.
    V = np.where(res_N[..., None] > 0, res_V, sp_V).astype(np.float32)
    N = np.maximum(res_N, sp_N)
    parties_in_file = list(rv["parties"])

    out = {}
    for i, p_upper in enumerate(parties_in_file):
        p_lower = p_upper.lower()
        if p_lower not in PARTIES_LOWER:
            continue
        valid = N[i] > 0
        if not valid.any():
            continue
        mean_vec = V[i, valid].mean(axis=0)
        out[p_lower] = mean_vec.astype(np.float64)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main() -> None:
    stated = build_stated_vectors()
    print(f"\nbuilt {len(stated)} stated vectors")
    revealed = build_revealed_vectors()
    print(f"built {len(revealed)} revealed vectors")

    parties = [p for p in PARTIES_LOWER if p in stated and p in revealed]
    print(f"\nparties with both: {parties}")

    # === Per-party stated-vs-revealed gap ===
    self_sim = {}
    for p in parties:
        self_sim[p] = cosine(stated[p], revealed[p])
    print("\nManifesto-vs-reservation cosine (own):")
    for p in parties:
        print(f"  {p.upper():>3s}: {self_sim[p]:.3f}")

    # === Cross-party: voter X's manifesto vs each party's revealed reasoning ===
    n = len(parties)
    cross = np.zeros((n, n))
    for i, a in enumerate(parties):
        for j, b in enumerate(parties):
            cross[i, j] = cosine(stated[a], revealed[b])

    # The row for party X: how similar X's manifesto is to each other party's
    # reservation behaviour. If the diagonal entry is NOT the max in its row,
    # that party's stated position drifted closer to another party's behaviour.
    print("\nstated vs each party's revealed (rows: manifesto, cols: revealed):")
    df = pd.DataFrame(cross,
                      index=[f"manifest_{p.upper()}" for p in parties],
                      columns=[f"revealed_{p.upper()}" for p in parties])
    print(df.round(3).to_string())

    # === Figure 1: per-party stated-vs-revealed bar chart ===
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(parties))
    bars = ax.bar(x, [self_sim[p] for p in parties],
                  color=[PARTY_COLOR[p.upper()] for p in parties],
                  edgecolor="white", linewidth=1.2)
    for bar, p in zip(bars, parties):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{self_sim[p]:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#444")
    ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in parties], fontsize=12, fontweight="bold")
    ax.set_ylabel("Cosine similarity between manifesto and reservations")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Stated vs revealed reasoning, per party\n"
                 "How closely does each party's manifesto language match its parliamentary "
                 "reservation language during the mandate?",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "web" / "24_stated_vs_revealed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    import shutil
    shutil.copyfile(out, ROOT / "site" / "public" / "figures" / out.name)
    print(f"  wrote {out.name}")

    # === Figure 2: cross-party heatmap centred on self-row ===
    # For each row (manifesto), subtract the row mean to highlight whose
    # revealed reasoning is closer than average.
    centred = cross - cross.mean(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    im = ax.imshow(centred, cmap="RdBu_r", vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(n)); ax.set_xticklabels(
        [p.upper() for p in parties], fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(
        [p.upper() for p in parties], fontsize=11, fontweight="bold")
    ax.set_xlabel("revealed (reservations)", fontsize=10)
    ax.set_ylabel("stated (manifesto)", fontsize=10)
    for i in range(n):
        for j in range(n):
            v = centred[i, j]
            color = "white" if abs(v) > 0.06 else "#333"
            marker = "■" if i == j else ""
            ax.text(j, i, f"{v:+.2f}{marker}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title("Whose parliamentary behaviour does each party's manifesto resemble?\n"
                 "Row-centred similarity. The boxed cell on the diagonal is the party's\n"
                 "stated-vs-own-revealed match. Red off-diagonal = the manifesto looks more like that party's behaviour.",
                 loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="deviation from manifesto's own row mean")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    out = FIG / "web" / "25_stated_vs_revealed_cross.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, ROOT / "site" / "public" / "figures" / out.name)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
