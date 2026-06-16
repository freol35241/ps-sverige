"""Longitudinal figures for the two-mandate article.

Produces:
  50_manifesto_element_trajectories.png — per-party element shares 2018→2022(→2026 for L)
  51_tido_agreement_longitudinal.png — Tidö unanimity per riksmöte across both mandates
  52_defectors_comparison.png — top defectors per mandate side by side
  53_manifesto_value_trajectories.png — signature value shifts per party 2018→2022
  54_per_mandate_centroid_maps.png — side-by-side party centroids on PC1/PC2 per mandate
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

FIG.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)

PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
PARTY_ORDER = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
INK = "#222222"
BG = "#FAFAFA"


def save(fig, name: str) -> None:
    p = FIG / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    import shutil
    shutil.copyfile(p, PUB / name)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# 50. Manifesto element trajectories (diagnosis, end-state, mechanism)
# ---------------------------------------------------------------------------

def fig_element_trajectories():
    df = pd.read_parquet(IN / "manifesto_trajectories.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), dpi=200, sharey=False)
    fig.patch.set_facecolor(BG)
    elements = [("share_diagnosis", "Diagnosis"),
                ("share_end_state", "End state"),
                ("share_mechanism", "Mechanism")]
    for ax, (col, title) in zip(axes, elements):
        ax.set_facecolor(BG)
        for party in PARTY_ORDER:
            sub = df[df["party"] == party].sort_values("year")
            if len(sub) < 2:
                continue
            ax.plot(sub["year"], sub[col] * 100,
                    marker="o", lw=2.0, color=PARTY_COLOR[party],
                    label=party, markersize=7)
            # Label the rightmost point with party initials
            x = sub["year"].iat[-1]
            y = sub[col].iat[-1] * 100
            ax.text(x + 0.3, y, party, color=PARTY_COLOR[party],
                    fontsize=10, va="center", fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
        ax.set_xlabel("election manifesto year", fontsize=10)
        ax.set_ylabel("share of substantive sentences (%)", fontsize=10)
        ax.set_xticks([2018, 2022, 2026])
        ax.set_xlim(2017, 2027.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Manifesto register, per party, across two election cycles\n"
                 "Lines connect each party's 2018 and 2022 manifestos. "
                 "L 2026 included where available.",
                 y=1.04, fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "50_manifesto_element_trajectories.png")


# ---------------------------------------------------------------------------
# 51. Tidö-bloc unanimity across both mandates
# ---------------------------------------------------------------------------

def fig_tido_longitudinal():
    a18 = pd.read_parquet(IN / "agreement_18_22.parquet")
    a22 = pd.read_parquet(IN / "agreement_22_26.parquet")
    a = pd.concat([a18, a22], ignore_index=True)
    a = a[a["n_votes"] >= 100]  # drop tiny n entries that span mandate boundaries
    a = a.sort_values("rm_label")

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(a))
    colors = ["#6D8FAA" if r["rm_label"].startswith(("2018", "2019", "2020", "2021"))
              else "#4A7CA0" for _, r in a.iterrows()]
    bars = ax.bar(x, a["unanimity_share"] * 100, color=colors,
                  edgecolor="white", linewidth=1.5)
    for b, share, n in zip(bars, a["unanimity_share"], a["n_votes"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{share*100:.0f} %\n(n={n})",
                ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(a["rm_label"], fontsize=10)
    ax.set_ylabel("share of chamber votes where M, KD, L and SD voted together (%)",
                  fontsize=11)
    ax.set_ylim(0, 110)
    # Mandate divider
    boundary_x = (a["rm_label"].tolist().index("2022-23")
                  if "2022-23" in a["rm_label"].tolist() else 4) - 0.5
    ax.axvline(boundary_x, color="#888", lw=1.0, linestyle="--")
    ax.text(boundary_x - 0.1, 105, "← 2018-22 mandate",
            ha="right", fontsize=10, color="#555")
    ax.text(boundary_x + 0.1, 105, "2022-26 mandate →",
            ha="left", fontsize=10, color="#555")
    ax.set_title("Did the four Tidö parties already vote together before 2022?\n"
                 "Share of contested chamber votes where M, KD, L and SD all voted the same way.",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "51_tido_agreement_longitudinal.png")


# ---------------------------------------------------------------------------
# 52. Defectors comparison
# ---------------------------------------------------------------------------

def fig_defectors_comparison():
    d18 = pd.read_parquet(IN / "defectors_18_22.parquet")
    d22 = pd.read_parquet(IN / "defectors_22_26.parquet")
    d18 = d18[d18["party_modal"].isin(PARTY_ORDER)].head(12)
    d22 = d22[d22["party_modal"].isin(PARTY_ORDER)].head(12)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), dpi=200, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, df, title in zip(axes,
                              [d18, d22],
                              ["2018-22 mandate", "2022-26 mandate"]):
        ax.set_facecolor(BG)
        head = df.iloc[::-1]
        y = np.arange(len(head))
        ax.barh(y, head["ratio_own_over_near"],
                color=[PARTY_COLOR.get(p, "#888") for p in head["party_modal"]],
                edgecolor="white")
        labels = [f"{n[:32]:<32}  ({p}→{q})"
                  for n, p, q in zip(head["namn"], head["party_modal"],
                                     head["nearest_other"])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5, fontfamily="monospace")
        ax.set_xscale("log")
        ax.axvline(1.0, color="#444", lw=0.8, linestyle="--")
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK,
                     loc="left", pad=6)
        ax.set_xlabel("d(own party) / d(nearest other party), log scale")
        for yi, val in zip(y, head["ratio_own_over_near"].to_numpy()):
            ax.text(val * 1.07, yi, f"{val:.1f}×", va="center",
                    fontsize=7.5, color="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Top behavioural defectors per mandate\n"
                 "Did the L-defectors phenomenon exist in 2018-22, "
                 "or did it emerge with Tidö?",
                 y=1.02, fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "52_defectors_comparison.png")


# ---------------------------------------------------------------------------
# 53. Manifesto value trajectories (per-party signature values 2018 → 2022)
# ---------------------------------------------------------------------------

def fig_value_trajectories():
    df = pd.read_parquet(IN / "manifesto_trajectories.parquet")
    value_cols = [c for c in df.columns if c.startswith("v_") and c.endswith("_per_kw")]
    # For each party, find the values that shifted most between 2018 and 2022
    rows = []
    for party in PARTY_ORDER:
        sub = df[(df["party"] == party) & (df["year"].isin([2018, 2022]))]
        if len(sub) < 2:
            continue
        p18 = sub[sub["year"] == 2018].iloc[0]
        p22 = sub[sub["year"] == 2022].iloc[0]
        for c in value_cols:
            v = c.replace("v_", "").replace("_per_kw", "")
            rows.append({"party": party, "value": v,
                         "v_2018": float(p18[c]), "v_2022": float(p22[c]),
                         "delta": float(p22[c]) - float(p18[c])})
    shifts = pd.DataFrame(rows)

    # Top 8 values by mean absolute shift across parties
    top_values = (shifts.groupby("value")["delta"].apply(lambda s: s.abs().mean())
                        .sort_values(ascending=False).head(8).index.tolist())

    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(top_values))
    w = 0.10
    for i, party in enumerate(PARTY_ORDER):
        sub = shifts[(shifts["party"] == party) & (shifts["value"].isin(top_values))]
        sub = sub.set_index("value").reindex(top_values).reset_index()
        deltas = sub["delta"].fillna(0).to_numpy()
        ax.bar(x + (i - 3.5) * w, deltas, w, label=party,
               color=PARTY_COLOR[party], edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_values, fontsize=10, rotation=20)
    ax.set_ylabel("Δ per-1,000-word frequency, 2018 to 2022", fontsize=11)
    ax.set_title("Signature values: where each party's manifesto shifted, 2018 to 2022\n"
                 "Top 8 values by mean absolute change. Positive = mentioned "
                 "more in 2022 than 2018.",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncols=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), fontsize=10)
    fig.tight_layout()
    save(fig, "53_value_shifts_2018_2022.png")


# ---------------------------------------------------------------------------
# 54. Side-by-side party centroid maps per mandate
# ---------------------------------------------------------------------------

def fig_centroid_maps():
    e18 = pd.read_parquet(IN / "mp_embedding_18_22.parquet")
    e22 = pd.read_parquet(IN / "mp_embedding_22_26.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=200)
    fig.patch.set_facecolor(BG)
    for ax, emb, title, var_text in zip(
            axes,
            [e18, e22],
            ["2018-22 mandate", "2022-26 mandate"],
            ["PC1 vs PC2  (variance ratios vary by mandate)",
             "PC1 vs PC2  (variance ratios vary by mandate)"]):
        ax.set_facecolor(BG)
        fit = emb[emb["in_fit"]]
        for party in PARTY_ORDER:
            sub = fit[fit["party_modal"] == party]
            if sub.empty:
                continue
            ax.scatter(sub["PC1"], sub["PC2"], s=18,
                       color=PARTY_COLOR[party], alpha=0.65,
                       edgecolor="white", linewidth=0.5)
            cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
            ax.scatter([cx], [cy], s=480, color=PARTY_COLOR[party],
                       edgecolor="white", linewidth=2, zorder=10)
            ax.text(cx, cy, party, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK,
                     loc="left", pad=6)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.axhline(0, color="#aaa", lw=0.5, alpha=0.5)
        ax.axvline(0, color="#aaa", lw=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("The chamber map, per mandate\n"
                 "In 2018-22, SD sits on its own arm. In 2022-26, "
                 "SD is inside the Tidö cluster on the right of PC1.",
                 y=1.02, fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "54_per_mandate_centroid_maps.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("== rendering longitudinal figures ==")
    fig_element_trajectories()
    fig_tido_longitudinal()
    fig_defectors_comparison()
    fig_value_trajectories()
    fig_centroid_maps()


if __name__ == "__main__":
    main()
