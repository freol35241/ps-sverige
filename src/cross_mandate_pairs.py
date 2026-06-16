"""Cross-mandate party-pair vote similarity.

For each riksmöte across both mandates, compute the vote-stance similarity
between every party pair (Pearson correlation of party mean vote vectors).
Output a long-form parquet plus a figure that traces selected pairs across
all 8 parliamentary years.

The Pearson approach handles the mandate-PCA-mismatch issue automatically:
party vote vectors live in the (events) basis of each year, but pairwise
correlations are scale-invariant and comparable across years.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "web"
PUB = ROOT / "site" / "public" / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
INK = "#222222"
BG = "#FAFAFA"


ROST_CODE = {"Ja": 1.0, "Nej": -1.0, "Avstår": 0.0, "Frånvarande": np.nan}


def load_long_with_party(tag: str) -> pd.DataFrame:
    if tag == "18_22":
        long = pd.read_parquet(IN / "votes_long_18_22.parquet")
        mps = pd.read_parquet(IN / "mps_18_22.parquet")
        events = pd.read_parquet(IN / "vote_events_18_22.parquet")
    else:
        long = pd.read_parquet(IN / "votes_long.parquet")
        mps = pd.read_parquet(IN / "mps.parquet")
        events = pd.read_parquet(IN / "vote_events.parquet")
    # 2022-26 lacks pre-computed rost_code; derive consistently
    if "rost_code" not in long.columns:
        long["rost_code"] = long["rost"].map(ROST_CODE)
    # Use party_modal from mps so defectors don't drag the party mean
    long = long.merge(mps[["intressent_id", "party_modal"]], on="intressent_id", how="left")
    events["datum"] = pd.to_datetime(events["datum"], errors="coerce")
    rm_year = events["datum"].apply(
        lambda d: f"{d.year}-{(d.year+1)%100:02d}" if pd.notna(d) and d.month >= 9
                  else f"{d.year-1}-{d.year%100:02d}" if pd.notna(d) else "")
    long = long.merge(
        events[["votering_id"]].assign(rm_label=rm_year.values),
        on="votering_id", how="left")
    return long


def per_year_party_vectors(long: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """For each riksmöte, return a dict {party: mean vote vector across vote_events of that year}.
    Uses rost_code if present (Yes=+1, No=-1, Avstår=0). Frånvarande dropped."""
    out: dict[str, dict[str, np.ndarray]] = {}
    long_valid = long.dropna(subset=["rost_code", "party_modal"])
    for rm, year_data in long_valid.groupby("rm_label"):
        if not rm:
            continue
        # Vote events of this year, in fixed order
        event_order = sorted(year_data["votering_id"].dropna().unique())
        event_idx = {e: i for i, e in enumerate(event_order)}
        n_events = len(event_order)
        party_vecs = {}
        for party in PARTIES:
            sub = year_data[year_data["party_modal"] == party]
            if sub.empty:
                continue
            # Per-event mean rost_code across all MPs of this party
            mean_per_event = (sub.groupby("votering_id")["rost_code"].mean()
                                 .reindex(event_order).to_numpy())
            party_vecs[party] = mean_per_event
        out[rm] = party_vecs
    return out


def pairwise_correlations(party_vecs: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    parties_present = [p for p in PARTIES if p in party_vecs]
    for i, p1 in enumerate(parties_present):
        for p2 in parties_present[i+1:]:
            v1, v2 = party_vecs[p1], party_vecs[p2]
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() < 50:
                continue
            r = np.corrcoef(v1[mask], v2[mask])[0, 1]
            rows.append({"pair": f"{p1}-{p2}", "r": float(r),
                         "n_overlap": int(mask.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    print("== loading 2018-22 ==")
    long18 = load_long_with_party("18_22")
    vec18 = per_year_party_vectors(long18)
    print("== loading 2022-26 ==")
    long22 = load_long_with_party("22_26")
    vec22 = per_year_party_vectors(long22)

    all_rows = []
    for rm in sorted(vec18.keys()):
        df = pairwise_correlations(vec18[rm])
        df["rm_label"] = rm
        all_rows.append(df)
    for rm in sorted(vec22.keys()):
        df = pairwise_correlations(vec22[rm])
        df["rm_label"] = rm
        all_rows.append(df)
    longp = pd.concat(all_rows, ignore_index=True)
    longp = longp[~longp["rm_label"].isin(["", "2022-23"]) | longp["rm_label"].ge("2022-23")]
    longp.to_parquet(IN / "pair_corr_yearly.parquet", index=False)
    print(f"  wrote pair_corr_yearly.parquet ({len(longp)} rows)")

    # Figure: trace selected pairs across both mandates
    focus_pairs = ["M-SD", "L-M", "L-KD", "L-SD", "S-C", "V-MP", "S-V", "M-KD"]
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    pair_color = {
        "M-SD": "#000000", "L-M": "#1F4E79", "L-KD": "#5B9BD5",
        "L-SD": "#FFC000", "S-C": "#E8112D", "V-MP": "#5B8C32",
        "S-V": "#AF0000", "M-KD": "#211F70",
    }
    for pair in focus_pairs:
        sub = longp[longp["pair"] == pair].sort_values("rm_label")
        if sub.empty:
            continue
        ax.plot(sub["rm_label"], sub["r"], marker="o",
                lw=2.0, label=pair,
                color=pair_color.get(pair, "#666"), markersize=6)
    # Mandate divider
    rm_labels_sorted = sorted(longp["rm_label"].unique())
    if "2022-23" in rm_labels_sorted:
        i = rm_labels_sorted.index("2022-23")
        ax.axvline(i - 0.5, color="#888", lw=1.0, linestyle="--")
        ax.text(i - 0.55, 1.0, "← 2018-22 mandate", ha="right",
                fontsize=10, color="#555", va="top")
        ax.text(i - 0.45, 1.0, "2022-26 mandate →", ha="left",
                fontsize=10, color="#555", va="top")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="#aaa", lw=0.5, alpha=0.6)
    ax.set_ylabel("Pearson correlation of party mean vote vectors", fontsize=11)
    ax.set_xlabel("parliamentary year", fontsize=11)
    ax.set_title("Did party pairs already start converging before 2022?\n"
                 "Each line is the per-year vote-vector correlation for a pair. "
                 "+1 = identical voting, -1 = mirrored, 0 = unrelated.",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=10, ncols=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    out = FIG / "55_pair_vote_correlation_longitudinal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    import shutil
    shutil.copyfile(out, PUB / out.name)
    print(f"  wrote {out.name}")

    # Print headline pair trajectories
    print("\n== M-SD across years ==")
    print(longp[longp["pair"] == "M-SD"].sort_values("rm_label").to_string(index=False))
    print("\n== L-KD across years ==")
    print(longp[longp["pair"] == "L-KD"].sort_values("rm_label").to_string(index=False))
    print("\n== L-M across years ==")
    print(longp[longp["pair"] == "L-M"].sort_values("rm_label").to_string(index=False))


if __name__ == "__main__":
    main()
