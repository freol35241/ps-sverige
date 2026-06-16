"""Per-mandate analyses against unified topic IDs.

Runs for both mandates and produces per-mandate parquets that can be
compared across mandates without re-running compute. Inputs:

  votes_matrix.npz / mps.parquet / vote_events.parquet      (2022-26)
  votes_matrix_18_22.npz / mps_18_22.parquet / vote_events_18_22.parquet  (2018-22)
  topics_unified.parquet (both mandates, unified IDs)

For each mandate, produces:

  mp_embedding_{tag}.parquet     PCA embedding of MPs
  pca_{tag}.npz                  PCA components + var ratios
  defectors_{tag}.parquet        per-MP defector index
  per_topic_stats_{tag}.parquet  per (party, topic_unified) mean stance
  agreement_{tag}.parquet        per-year Tidö-bloc agreement share

Where {tag} is "18_22" or "22_26".

Designed to be re-run idempotently; existing outputs are overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src"))

MIN_VOTES = 200  # MPs with at least this many votes participate in PCA fit
PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
TIDÖ_PARTIES = ["M", "KD", "L", "SD"]
PC_COLS = ["PC1", "PC2", "PC3", "PC4"]


def load_mandate(tag: str):
    """Return matrix, mps, events for the given mandate tag (18_22 or 22_26)."""
    if tag == "22_26":
        data = np.load(IN / "votes_matrix.npz", allow_pickle=True)
        mps = pd.read_parquet(IN / "mps.parquet")
        events = pd.read_parquet(IN / "vote_events.parquet")
    elif tag == "18_22":
        data = np.load(IN / "votes_matrix_18_22.npz", allow_pickle=True)
        mps = pd.read_parquet(IN / "mps_18_22.parquet")
        events = pd.read_parquet(IN / "vote_events_18_22.parquet")
    else:
        raise ValueError(tag)
    M = data["M"]
    mp_order = list(data["mp_order"])
    vote_order = list(data["vote_order"])
    mps = mps.set_index("intressent_id").loc[mp_order].reset_index()
    events = events.set_index("votering_id").loc[vote_order].reset_index()
    return M, mps, events


# ---------------------------------------------------------------------------
# PCA / behavioural map
# ---------------------------------------------------------------------------

def run_pca(tag: str):
    print(f"\n== PCA: {tag} ==")
    M, mps, events = load_mandate(tag)
    present = np.isfinite(M).sum(axis=1)
    fit_mask = present >= MIN_VOTES
    col_mean = np.nanmean(M[fit_mask], axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    M_imp = np.where(np.isnan(M), col_mean, M).astype(np.float32)
    M_imp -= col_mean
    pca = PCA(n_components=4, random_state=0)
    pca.fit(M_imp[fit_mask])
    coords = pca.transform(M_imp)

    out = mps.copy()
    for k in range(4):
        out[f"PC{k+1}"] = coords[:, k]
    out["n_present_votes"] = present
    out["in_fit"] = fit_mask

    out.to_parquet(IN / f"mp_embedding_{tag}.parquet", index=False)
    np.savez(IN / f"pca_{tag}.npz",
             components=pca.components_,
             var_ratio=pca.explained_variance_ratio_,
             col_mean=col_mean,
             vote_order=np.array(events["votering_id"].tolist()))
    var = pca.explained_variance_ratio_
    print("  variance explained: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(var)))
    print(f"  fit-set: {fit_mask.sum()} of {len(out)} MPs (>= {MIN_VOTES} votes)")
    print("  party centroids on PC1 / PC2:")
    fit = out[out["in_fit"]]
    cents = fit.groupby("party_modal")[["PC1", "PC2"]].mean().sort_values("PC1")
    print(cents.round(3).to_string())
    return out


# ---------------------------------------------------------------------------
# Defectors
# ---------------------------------------------------------------------------

def run_defectors(tag: str):
    print(f"\n== defectors: {tag} ==")
    emb = pd.read_parquet(IN / f"mp_embedding_{tag}.parquet")
    fit = emb[emb["in_fit"]].copy()
    centroids = fit.groupby("party_modal")[PC_COLS].mean()
    scatter = (fit.groupby("party_modal")[PC_COLS]
                  .apply(lambda g: np.linalg.norm(
                      g.to_numpy() - g.to_numpy().mean(axis=0), axis=1).mean()))

    def per_mp(row):
        x = row[PC_COLS].to_numpy(dtype=float)
        own = row["party_modal"]
        if own not in centroids.index:
            return pd.Series({"d_own": np.nan, "d_near_other": np.nan,
                              "nearest_other": "", "ratio_own_over_near": np.nan,
                              "z_own": np.nan})
        d_own = float(np.linalg.norm(x - centroids.loc[own].to_numpy()))
        others = centroids.drop(index=own)
        d_each = np.linalg.norm(others.to_numpy() - x, axis=1)
        i_near = int(np.argmin(d_each))
        d_near = float(d_each[i_near])
        nearest_other = others.index[i_near]
        return pd.Series({
            "d_own": d_own,
            "d_near_other": d_near,
            "nearest_other": nearest_other,
            "ratio_own_over_near": d_own / d_near if d_near else np.nan,
            "z_own": d_own / scatter[own] if scatter[own] else np.nan,
        })

    extra = fit.apply(per_mp, axis=1)
    out = pd.concat([fit.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    out = out.sort_values("ratio_own_over_near", ascending=False)
    out.to_parquet(IN / f"defectors_{tag}.parquet", index=False)
    print(f"  top 12 by ratio_own_over_near:")
    print(out.head(12)[["namn", "party_modal", "nearest_other",
                        "ratio_own_over_near"]].round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Per (party, unified topic) mean stance
# ---------------------------------------------------------------------------

def run_per_topic_stats(tag: str):
    """Compute per (party, topic) mean stance using unified topic IDs."""
    print(f"\n== per-topic stance: {tag} ==")
    M, mps, events = load_mandate(tag)

    # Load unified topics filtered to this mandate
    topics = pd.read_parquet(IN / "topics_unified.parquet")
    mandate_key = "2018-22" if tag == "18_22" else "2022-26"
    topics = topics[topics["mandate"] == mandate_key][
        ["votering_id", "topic_id_unified"]]
    topics["votering_id"] = topics["votering_id"].astype(str)
    events["votering_id"] = events["votering_id"].astype(str)

    # Map each vote event (column of M) to a topic
    event_to_topic = dict(zip(topics["votering_id"], topics["topic_id_unified"]))
    vote_order = events["votering_id"].tolist()
    col_topic = np.array([event_to_topic.get(v, -1) for v in vote_order])
    print(f"  vote events mapped to a unified topic: "
          f"{(col_topic >= 0).sum():,} / {len(col_topic):,}")

    # Per party: MPs of that party_modal
    rows = []
    for party in PARTIES:
        mp_mask = (mps["party_modal"] == party).to_numpy()
        if mp_mask.sum() == 0:
            continue
        sub = M[mp_mask]
        for t in range(28):
            col_mask = col_topic == t
            if col_mask.sum() == 0:
                continue
            block = sub[:, col_mask]
            obs = np.isfinite(block).sum()
            if obs == 0:
                continue
            mean_stance = np.nanmean(block)
            rows.append({
                "party": party,
                "topic_id_unified": int(t),
                "mean_stance": float(mean_stance),
                "n_obs": int(obs),
                "n_votes": int(col_mask.sum()),
                "n_mps": int(mp_mask.sum()),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(IN / f"per_topic_stats_{tag}.parquet", index=False)
    print(f"  rows written: {len(df)}")


# ---------------------------------------------------------------------------
# Bloc agreement / collapse trend
# ---------------------------------------------------------------------------

def run_agreement(tag: str):
    """Per-year Tidö-bloc agreement share."""
    print(f"\n== Tidö-bloc agreement trend: {tag} ==")
    M, mps, events = load_mandate(tag)
    # Vote date → riksmöte year
    events["datum"] = pd.to_datetime(events["datum"], errors="coerce")
    # Riksmöte runs Sept→Aug; assign year = year of vote if month >= 9 else year-1
    rm_year = events["datum"].apply(
        lambda d: f"{d.year}-{(d.year+1)%100:02d}" if pd.notna(d) and d.month >= 9
                  else f"{d.year-1}-{d.year%100:02d}" if pd.notna(d)
                  else "")
    events["rm_label"] = rm_year

    # Party-of-MP map (for column-level operations we go MP-by-MP)
    mp_party = dict(zip(mps["intressent_id"].tolist(), mps["party_modal"].tolist()))
    mp_order = mps["intressent_id"].tolist()
    party_of_row = np.array([mp_party.get(m, "") for m in mp_order])

    rows = []
    for rm_label, group in events.groupby("rm_label"):
        if not rm_label:
            continue
        col_mask = events["rm_label"].to_numpy() == rm_label
        sub = M[:, col_mask]
        # Per vote, what fraction of Tidö-party MPs voted the same way?
        n_unanimous = 0
        n_total = 0
        for j in range(sub.shape[1]):
            col = sub[:, j]
            party_voted = {}
            for p in TIDÖ_PARTIES:
                mp_mask = party_of_row == p
                cell = col[mp_mask]
                # Take the mode of finite values
                finite = cell[np.isfinite(cell)]
                if len(finite) == 0:
                    party_voted[p] = np.nan
                else:
                    party_voted[p] = float(np.median(finite))
            vals = [v for v in party_voted.values() if not np.isnan(v)]
            if len(vals) < 2:
                continue
            n_total += 1
            if all(v == vals[0] for v in vals):
                n_unanimous += 1
        rows.append({
            "rm_label": rm_label,
            "n_votes": int(n_total),
            "n_unanimous": int(n_unanimous),
            "unanimity_share": n_unanimous / n_total if n_total else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("rm_label")
    df.to_parquet(IN / f"agreement_{tag}.parquet", index=False)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for tag in ("18_22", "22_26"):
        # Skip if base data not present
        if tag == "18_22" and not (IN / "votes_matrix_18_22.npz").exists():
            print(f"  skip {tag}: votes_matrix_18_22.npz not built yet")
            continue
        run_pca(tag)
        run_defectors(tag)
        run_per_topic_stats(tag)
        run_agreement(tag)
    print("\n== done ==")


if __name__ == "__main__":
    main()
