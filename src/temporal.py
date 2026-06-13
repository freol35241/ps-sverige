"""Per-riksmöte PCA embeddings, Procrustes-aligned across years.

Strategy
--------
For each riksmöte t:
  1. Slice the matrix to that year's vote events.
  2. Subset to MPs with >= MIN_VOTES present votes in that year.
  3. PCA into K dims.

Then for each consecutive pair (t-1, t):
  - Find MPs present in both years (overlap set).
  - Orthogonal Procrustes (with reflection allowed) to rotate year-t coords
    onto year-(t-1) coords on the overlap set.
  - Apply the rotation to all year-t coordinates.

Year-0 reference frame is the first riksmöte's PCA. All subsequent years are
rotated/reflected into that frame. This preserves within-year structure while
making cross-year coordinates comparable.

The price of refit-per-year (vs projecting through a fixed basis) is that
"PC1" can mean slightly different things each year. Procrustes handles
rigid alignment (rotation+reflection) but not scale, so we also rescale
each year's coords to match year-0 RMS for plotting consistency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

RIKSMOTEN = ["2022/23", "2023/24", "2024/25", "2025/26"]
N_COMPONENTS = 4
MIN_VOTES_PER_YEAR = 80      # MP must be present this many times that year to be embedded
MIN_OVERLAP_FOR_PROCRUSTES = 30


def load():
    data = np.load(IN / "votes_matrix.npz", allow_pickle=True)
    M = data["M"]
    mp_order = list(data["mp_order"])
    vote_order = list(data["vote_order"])
    events = pd.read_parquet(IN / "vote_events.parquet").set_index("votering_id")
    mps = pd.read_parquet(IN / "mps.parquet")
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


def align_to_reference(ref_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Procrustes-align df's PCs to ref_df's PCs using the overlap MPs (in_fit in both)."""
    pc_cols = [f"PC{k+1}" for k in range(N_COMPONENTS)]
    ref_fit = ref_df[ref_df["in_fit"]].set_index("intressent_id")[pc_cols]
    cur_fit = df[df["in_fit"]].set_index("intressent_id")[pc_cols]
    overlap = ref_fit.index.intersection(cur_fit.index)
    if len(overlap) < MIN_OVERLAP_FOR_PROCRUSTES:
        raise RuntimeError(f"insufficient overlap ({len(overlap)}) for Procrustes")
    A = cur_fit.loc[overlap].to_numpy()
    B = ref_fit.loc[overlap].to_numpy()

    # Centre both.
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    # Orthogonal Procrustes: find R such that Ac @ R ≈ Bc.
    R, scale = orthogonal_procrustes(Ac, Bc)

    # Rescale so RMS of the aligned overlap matches the reference's RMS
    # (otherwise different years use different PC magnitudes).
    A_rot = Ac @ R
    rms_A = np.sqrt((A_rot ** 2).sum() / A_rot.size)
    rms_B = np.sqrt((Bc ** 2).sum() / Bc.size)
    s = rms_B / rms_A if rms_A else 1.0

    # Apply to all rows (not just overlap).
    X = df[pc_cols].to_numpy() - A.mean(axis=0)
    X_aligned = (X @ R) * s + B.mean(axis=0)
    aligned = df.copy()
    for k, col in enumerate(pc_cols):
        aligned[col] = X_aligned[:, k]
    return aligned


def main() -> None:
    M, mp_order, vote_order, mps, events = load()

    yearly = []
    for rm in RIKSMOTEN:
        df, _ = embed_year(M, mp_order, vote_order, mps, events, rm)
        n_fit = int(df["in_fit"].sum())
        n_votes = int((events.loc[vote_order, "rm"] == rm).sum())
        print(f"  {rm}: {n_votes} vote events, {n_fit} MPs in fit")
        yearly.append(df)

    # Use first year as anchor; align each subsequent year to its predecessor,
    # then re-express in year-0 frame by composing alignments transitively.
    aligned = [yearly[0].copy()]
    for t in range(1, len(yearly)):
        a = align_to_reference(aligned[t - 1], yearly[t])
        aligned.append(a)

    long = pd.concat(aligned, ignore_index=True)
    long["rm_idx"] = long["rm"].map({rm: i for i, rm in enumerate(RIKSMOTEN)})
    long.to_parquet(OUT / "embedding_yearly.parquet", index=False)
    print(f"\nwrote {OUT}/embedding_yearly.parquet  ({len(long):,} rows)")

    # Quick centroid drift report.
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
        print(f"  {party:3s}  " + "  →  ".join(pts))

    # L↔KD centroid distance per year (2D, then full 4D).
    print("\nL↔KD centroid distance per year:")
    for rm in RIKSMOTEN:
        sub = fit_only[fit_only["rm"] == rm]
        L = sub[sub["party_modal"] == "L"][pc_cols].mean().to_numpy()
        K = sub[sub["party_modal"] == "KD"][pc_cols].mean().to_numpy()
        if np.isnan(L).any() or np.isnan(K).any():
            print(f"  {rm}: insufficient data")
            continue
        d2 = np.linalg.norm(L[:2] - K[:2])
        d4 = np.linalg.norm(L - K)
        print(f"  {rm}: 2D={d2:5.2f}   4D={d4:5.2f}")


if __name__ == "__main__":
    main()
