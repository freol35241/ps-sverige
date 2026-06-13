"""Stylometric MP embedding from speech text.

For each MP we concatenate all their speeches in the mandate, TF-IDF vectorize
(unigrams + bigrams), reduce dim with Truncated SVD, and compare to the vote
layout.

Cabinet-minister caveat: ministers read prepared government statements.
Their "voice" partly reflects the government's voice, not their personal
style. Tagged in the output but not removed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.figures import PARTY_COLOR, PARTY_ORDER

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# Drop title speakers (TALMANNEN, KONUNGEN, etc.) — they're not MPs.
TITLE_PARTIES = {"TALMANNEN", "ANDRE VICE TALMANNEN", "TREDJE VICE TALMANNEN",
                 "FÖRSTE VICE TALMANNEN", "HANS MAJESTÄT KONUNGEN",
                 "TJÄNSTGÖRANDE ÅLDERSPRESIDENTEN"}
MIN_WORDS = 10_000   # words of speech to be embeddable
MIN_SPEECHES = 30    # discard MPs who barely spoke
N_SVD = 50           # truncated SVD dims, then PCA(2) on top for plotting


def main() -> None:
    speeches = pd.read_parquet(IN / "speeches.parquet")
    speeches = speeches[~speeches["parti"].isin(TITLE_PARTIES)]
    speeches = speeches[speeches["intressent_id"].notna()]
    print(f"speeches retained (MPs only): {len(speeches):,}")

    # Aggregate per MP.
    agg = (speeches.groupby("intressent_id")
                    .agg(text=("text", " ".join),
                         n_words=("n_words", "sum"),
                         n_speeches=("anforande_id", "count"),
                         party_modal=("parti", lambda s: s.mode().iat[0]),
                         talare=("talare", lambda s: s.iloc[-1]))
                    .reset_index())
    print(f"speakers: {len(agg):,}")
    agg = agg[(agg["n_words"] >= MIN_WORDS) & (agg["n_speeches"] >= MIN_SPEECHES)]
    print(f"speakers with >= {MIN_WORDS:,} words AND >= {MIN_SPEECHES} speeches: {len(agg):,}")

    # Word 1-2grams. Character n-grams were tried; they added noise (style is
    # swamped by the chamber's institutional language). See diary.
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.9,
        sublinear_tf=True, max_features=120_000,
    )
    X = vec.fit_transform(agg["text"])
    print(f"TF-IDF: {X.shape[1]:,} features over {X.shape[0]} docs, density {X.nnz/(X.shape[0]*X.shape[1]):.3%}")

    svd = TruncatedSVD(n_components=N_SVD, random_state=0)
    Y = svd.fit_transform(X)
    print(f"SVD var explained (first 4): {svd.explained_variance_ratio_[:4]}")
    print(f"SVD var explained (cumulative @ {N_SVD}): {svd.explained_variance_ratio_.sum():.1%}")

    # Standardise to give all dims comparable weight before the 2D projection.
    Ys = StandardScaler(with_std=True).fit_transform(Y)

    # 2D projection: just take the top 2 SVD dims AFTER PCA on the standardised SVD basis,
    # to align with the meaningful axes.
    from sklearn.decomposition import PCA
    p2 = PCA(n_components=2, random_state=0)
    coords = p2.fit_transform(Ys)
    print(f"2D PCA on SVD: variance {p2.explained_variance_ratio_}")

    agg["SC1"] = coords[:, 0]
    agg["SC2"] = coords[:, 1]
    out_path = OUT / "speech_embedding.parquet"
    agg.drop(columns=["text"]).to_parquet(out_path, index=False)
    print(f"wrote {out_path}")

    # === Compare against vote embedding ===
    vote = pd.read_parquet(IN / "mp_embedding.parquet")[
        ["intressent_id", "PC1", "PC2", "in_fit"]]
    merged = agg.merge(vote, on="intressent_id", how="inner")
    merged = merged[merged["in_fit"]]
    print(f"\nmerged set (in vote fit + has speeches): {len(merged)}")

    # Procrustes-align speech coords onto vote coords for visual comparison.
    A = merged[["SC1", "SC2"]].to_numpy()
    B = merged[["PC1", "PC2"]].to_numpy()
    Ac, Bc = A - A.mean(axis=0), B - B.mean(axis=0)
    R, _ = orthogonal_procrustes(Ac, Bc)
    A_rot = Ac @ R
    s = np.sqrt((Bc ** 2).sum() / (A_rot ** 2).sum())
    A_rot *= s
    A_rot += B.mean(axis=0)
    merged["SC1_aln"] = A_rot[:, 0]
    merged["SC2_aln"] = A_rot[:, 1]

    # Correlation between the two layouts.
    pearson = np.corrcoef(merged["SC1_aln"], merged["PC1"])[0, 1]
    pearson2 = np.corrcoef(merged["SC2_aln"], merged["PC2"])[0, 1]
    print(f"Correlation speech↔vote: PC1={pearson:.3f}, PC2={pearson2:.3f}")

    # ----- Figure: side-by-side scatter -----
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    for ax, (xcol, ycol, title) in zip(axes, [
        ("PC1", "PC2", "Votes — what MPs do"),
        ("SC1_aln", "SC2_aln", "Speeches — what MPs say"),
    ]):
        ax.set_facecolor("#FAFAFA")
        for party in PARTY_ORDER:
            sub = merged[merged["party_modal"] == party]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub[ycol], s=40,
                       c=PARTY_COLOR[party], edgecolor="white", linewidth=0.5,
                       alpha=0.9, zorder=3)
        cents = merged.groupby("party_modal")[[xcol, ycol]].mean()
        for party, row in cents.iterrows():
            ax.annotate(party, (row[xcol], row[ycol]),
                        fontsize=12, fontweight="bold", ha="center", va="center",
                        color="white",
                        bbox=dict(boxstyle="circle,pad=0.32",
                                  fc=PARTY_COLOR.get(party, "#444"),
                                  ec="white", lw=1.0),
                        zorder=5)
        ax.axhline(0, color="#BBB", lw=0.4); ax.axvline(0, color="#BBB", lw=0.4)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("primary axis")
        ax.set_ylabel("secondary axis")

    fig.suptitle("Two independent embeddings of the same parliament\n"
                 f"PC1 corr = {pearson:.2f}, PC2 corr = {pearson2:.2f}, "
                 f"Procrustes-aligned",
                 fontsize=13, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG / "09_speech_vs_vote.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # ----- Defectors: speech vs vote disagreement -----
    # MP-level: distance between vote position and speech position (normalised).
    merged["delta"] = np.linalg.norm(
        merged[["PC1", "PC2"]].to_numpy() - merged[["SC1_aln", "SC2_aln"]].to_numpy(),
        axis=1,
    )
    top = merged.sort_values("delta", ascending=False).head(15)
    print("\nMPs whose speech-position diverges most from their vote-position:")
    print(top[["talare", "party_modal", "PC1", "PC2",
               "SC1_aln", "SC2_aln", "delta", "n_speeches"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
