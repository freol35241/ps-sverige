"""PCA embedding of MPs into 2D behavioural space.

Encoding:
  Ja=+1, Nej=-1, Avstår=0, Frånvarande=NaN.
Imputation:
  column mean (the average of present voters on that vote).
Filter:
  MPs with < MIN_VOTES non-missing votes are excluded from the fit
  (substitutes who served briefly), but their coordinates are still computed
  by projecting onto the principal axes for completeness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

MIN_VOTES = 200  # exclude MPs with < this many present votes from the PCA fit


def load():
    data = np.load(IN / "votes_matrix.npz", allow_pickle=True)
    M = data["M"]                             # (n_mps, n_votes), float32, NaN-coded
    mp_order = list(data["mp_order"])
    vote_order = list(data["vote_order"])
    mps = pd.read_parquet(IN / "mps.parquet")
    events = pd.read_parquet(IN / "vote_events.parquet")
    mps = mps.set_index("intressent_id").loc[mp_order].reset_index()
    events = events.set_index("votering_id").loc[vote_order].reset_index()
    return M, mps, events


def embed(M: np.ndarray, mps: pd.DataFrame, n_components: int = 4):
    present = np.isfinite(M).sum(axis=1)
    fit_mask = present >= MIN_VOTES

    # Column means computed on the fit set, then used to impute everyone.
    col_mean = np.nanmean(M[fit_mask], axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)

    M_imp = np.where(np.isnan(M), col_mean, M).astype(np.float32)
    M_imp -= col_mean  # centre at the "average MP"

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(M_imp[fit_mask])
    coords = pca.transform(M_imp)

    out = mps.copy()
    for k in range(n_components):
        out[f"PC{k+1}"] = coords[:, k]
    out["n_present_votes"] = present
    out["in_fit"] = fit_mask
    return out, pca, col_mean


def main() -> None:
    M, mps, events = load()
    print(f"matrix {M.shape}, {np.isfinite(M).mean():.1%} present")

    embedded, pca, col_mean = embed(M, mps)
    var = pca.explained_variance_ratio_
    print(f"variance explained: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))

    print(f"\nfit-set: {embedded['in_fit'].sum()} of {len(embedded)} MPs (>= {MIN_VOTES} present votes)")

    # Inspect axis interpretation: which parties land where.
    fit = embedded[embedded["in_fit"]]
    print("\nparty centroids on PC1 / PC2:")
    centroids = fit.groupby("party_modal")[["PC1", "PC2"]].mean().sort_values("PC1")
    print(centroids.round(3).to_string())

    embedded.to_parquet(OUT / "mp_embedding.parquet", index=False)
    np.savez(OUT / "pca.npz",
             components=pca.components_, var_ratio=pca.explained_variance_ratio_,
             col_mean=col_mean)
    print(f"\nwrote {OUT}/mp_embedding.parquet")


if __name__ == "__main__":
    main()
