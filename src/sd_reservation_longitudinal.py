"""8-year reservations-filed trajectory per party, both mandates.

Builds figure 21_sd_reservation_collapse_longitudinal showing each
party's reservation count per riksmöte across the full 2018-26 window,
with SD highlighted.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures" / "web"
PUB = ROOT / "site" / "public" / "figures"

PARTY_COLOR = {
    "V": "#AF0000", "S": "#E8112D", "MP": "#83CF39", "C": "#009933",
    "L": "#006AB3", "KD": "#211F70", "M": "#52BDEC", "SD": "#DDDD00",
}
INK = "#222222"
BG = "#FAFAFA"

RM_PREFIX = {
    "H601": "2018-19", "H701": "2019-20", "H801": "2020-21", "H901": "2021-22",
    "HA01": "2022-23", "HB01": "2023-24", "HC01": "2024-25", "HD01": "2025-26",
}


def rm_from_dokid(s: str) -> str:
    return RM_PREFIX.get(s[:4] if isinstance(s, str) else "", "")


def build_counts() -> pd.DataFrame:
    r18 = pd.read_parquet(IN / "reservations_18_22.parquet")
    r22 = pd.read_parquet(IN / "reservations.parquet")
    r18["rm"] = r18["dok_id"].apply(rm_from_dokid)
    r22["rm"] = r22["dok_id"].apply(rm_from_dokid)
    parties = {"V", "S", "MP", "C", "L", "KD", "M", "SD"}
    rows = []
    for df in (r18, r22):
        for _, row in df.iterrows():
            for p in str(row.get("partier", "")).split(";"):
                p = p.strip()
                if p in parties:
                    rows.append({"party": p, "rm": row["rm"]})
    expanded = pd.DataFrame(rows)
    expanded = expanded[expanded["rm"] != ""]
    counts = expanded.groupby(["party", "rm"]).size().unstack(fill_value=0)
    counts = counts.reindex(["V", "S", "MP", "C", "L", "KD", "M", "SD"])
    counts = counts[sorted(counts.columns)]
    return counts


def main() -> None:
    counts = build_counts()
    counts.to_parquet(IN / "reservations_per_party_per_year.parquet")
    print(counts.to_string())

    rms = list(counts.columns)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(rms))
    for party in ["V", "S", "MP", "C", "L", "KD", "M", "SD"]:
        y = counts.loc[party].to_numpy()
        is_sd = party == "SD"
        lw = 4.0 if is_sd else 1.5
        alpha = 1.0 if is_sd else 0.7
        ax.plot(x, y, marker="o", lw=lw, color=PARTY_COLOR[party],
                label=party, markersize=6 if is_sd else 4, alpha=alpha)
        # Label the rightmost point
        ax.text(len(rms) - 0.95, y[-1], f" {party}",
                color=PARTY_COLOR[party], fontsize=10,
                fontweight="bold" if is_sd else "normal",
                va="center")
    # Mandate boundary
    boundary = rms.index("2022-23") - 0.5
    ax.axvline(boundary, color="#888", lw=1.0, linestyle="--")
    ax.text(boundary - 0.1, ax.get_ylim()[1] * 0.95,
            "← 2018-22 opposition →",
            ha="right", fontsize=10, color="#555")
    ax.text(boundary + 0.1, ax.get_ylim()[1] * 0.95,
            "← 2022-26 Tidö (M,KD,L,SD in cabinet/support) →",
            ha="left", fontsize=10, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(rms, fontsize=10)
    ax.set_xlim(-0.5, len(rms))
    ax.set_ylabel("Reservations filed (signed by party)", fontsize=11)
    ax.set_title("SD's reservation collapse in context\n"
                 "Reservations filed per party per riksmöte. SD highlighted. M, KD, L "
                 "drop to near zero in 2022-26 because they are in cabinet.",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "57_reservations_longitudinal.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(p, PUB / p.name)
    print(f"\nwrote {p.name}")


if __name__ == "__main__":
    main()
