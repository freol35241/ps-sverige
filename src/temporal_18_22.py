"""Per-riksmöte PCA for the 2018-22 mandate, Procrustes-aligned across years.

Mirrors src/temporal.py but reads votes_matrix_18_22.npz and writes
embedding_yearly_18_22.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src"))
from temporal import align_to_reference, N_COMPONENTS, MIN_VOTES_PER_YEAR  # noqa: E402

RIKSMOTEN = ["2018/19", "2019/20", "2020/21", "2021/22"]


def load():
    data = np.load(IN / "votes_matrix_18_22.npz", allow_pickle=True)
    M = data["M"]
    mp_order = list(data["mp_order"])
    vote_order = list(data["vote_order"])
    events = pd.read_parquet(IN / "vote_events_18_22.parquet").set_index("votering_id")
    mps = pd.read_parquet(IN / "mps_18_22.parquet")
    mps = mps.set_index("intressent_id").loc[mp_order].reset_index()
    return M, mp_order, vote_order, mps, events


def embed_year(M, mp_order, vote_order, mps, events, riksmote):
    vote_mask = events.loc[vote_order, "rm"].to_numpy() == riksmote
    M_yr = M[:, vote_mask]
    if M_yr.shape[1] == 0:
        return None, None
    present = np.isfinite(M_yr).sum(axis=1)
    fit_mask = present >= MIN_VOTES_PER_YEAR
    col_mean = np.nanmean(M_yr[fit_mask], axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    M_imp = np.where(np.isnan(M_yr), col_mean, M_yr).astype(np.float32)
    M_imp -= col_mean
    pca = PCA(n_components=N_COMPONENTS, random_state=0)
    pca.fit(M_imp[fit_mask])
    coords = pca.transform(M_imp)
    df = pd.DataFrame({
        "intressent_id": mp_order,
        "namn": mps["namn"].to_numpy(),
        "party_modal": mps["party_modal"].to_numpy(),
        "n_present": present,
        "in_fit": fit_mask,
        "rm": riksmote,
    })
    for k in range(N_COMPONENTS):
        df[f"PC{k+1}"] = coords[:, k]
    return df, pca


def main() -> None:
    M, mp_order, vote_order, mps, events = load()
    yearly = []
    for rm in RIKSMOTEN:
        df, _ = embed_year(M, mp_order, vote_order, mps, events, rm)
        if df is None:
            continue
        n_fit = int(df["in_fit"].sum())
        n_votes = int((events.loc[vote_order, "rm"] == rm).sum())
        print(f"  {rm}: {n_votes} vote events, {n_fit} MPs in fit")
        yearly.append(df)

    aligned = [yearly[0].copy()]
    for t in range(1, len(yearly)):
        a = align_to_reference(aligned[t - 1], yearly[t])
        aligned.append(a)

    long = pd.concat(aligned, ignore_index=True)
    long["rm_idx"] = long["rm"].map({rm: i for i, rm in enumerate(RIKSMOTEN)})
    long.to_parquet(IN / "embedding_yearly_18_22.parquet", index=False)
    print(f"\nwrote embedding_yearly_18_22.parquet ({len(long):,} rows)")

    pc_cols = [f"PC{k+1}" for k in range(N_COMPONENTS)]
    fit_only = long[long["in_fit"]]
    cents = (fit_only.groupby(["party_modal", "rm"])[pc_cols].mean()
             .reset_index())
    print("\nParty centroid trajectories on PC1, PC2:")
    for party in ["S", "V", "MP", "C", "L", "M", "KD", "SD"]:
        sub = cents[cents["party_modal"] == party].sort_values("rm")
        if sub.empty:
            continue
        pts = [f"{r['rm']}: ({r['PC1']:+6.1f}, {r['PC2']:+6.1f})"
               for _, r in sub.iterrows()]
        print(f"  {party:3s}  " + "  ->  ".join(pts))


if __name__ == "__main__":
    main()
