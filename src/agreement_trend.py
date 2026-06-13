"""Direct test of bloc cohesion using raw votes (not PCA).

For each riksmöte we compute: share of vote events where all parties in
a given set vote the same majority. This is the model-free counterpart
to the Procrustes-PCA "Tidö collapse" figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

RIKSMOTEN = ["2022/23", "2023/24", "2024/25", "2025/26"]


def majority_excl_absent(s: pd.Series):
    s = s[s != "Frånvarande"]
    if s.empty:
        return None
    return s.value_counts().idxmax()


def agreement_per_year(long: pd.DataFrame, parties: list[str]) -> pd.Series:
    out = {}
    for rm in RIKSMOTEN:
        sub = long[long["rm"] == rm]
        pm = (sub.groupby(["votering_id", "parti"])["rost"]
                 .agg(majority_excl_absent)
                 .unstack("parti"))
        block = pm[parties].dropna()
        agree = (block.nunique(axis=1) == 1).mean() if len(block) else np.nan
        out[rm] = agree
    return pd.Series(out)


def pairwise_agreement(long: pd.DataFrame, a: str, b: str) -> pd.Series:
    out = {}
    for rm in RIKSMOTEN:
        sub = long[long["rm"] == rm]
        pm = (sub.groupby(["votering_id", "parti"])["rost"]
                 .agg(majority_excl_absent)
                 .unstack("parti"))
        block = pm[[a, b]].dropna()
        out[rm] = (block[a] == block[b]).mean() if len(block) else np.nan
    return pd.Series(out)


def main() -> None:
    long = pd.read_parquet(IN / "votes_long.parquet")

    cabinet = agreement_per_year(long, ["M", "KD", "L"])
    tido = agreement_per_year(long, ["M", "KD", "L", "SD"])
    opp = agreement_per_year(long, ["S", "V", "MP", "C"])
    sd_cab = pairwise_agreement(long, "SD", "M")  # SD agreement with cabinet largest party
    sd_l = pairwise_agreement(long, "SD", "L")    # SD vs L (the loosest cabinet party)
    c_s = pairwise_agreement(long, "C", "S")      # C-S coupling

    print("Agreement rates per riksmöte:")
    print(pd.DataFrame({
        "M+KD+L (cabinet)": cabinet,
        "M+KD+L+SD (Tidö)": tido,
        "Opposition (S+V+MP+C)": opp,
        "SD ↔ M": sd_cab,
        "SD ↔ L": sd_l,
        "C ↔ S": c_s,
    }).round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=140)
    fig.patch.set_facecolor("#FAFAFA")
    x = np.arange(len(RIKSMOTEN))

    # Panel A: bloc-level
    ax = axes[0]; ax.set_facecolor("#FAFAFA")
    ax.plot(x, tido * 100, "-o", lw=2.5, color="#DD6E0F",
            label="Tidö bloc (M+KD+L+SD)", markersize=10)
    ax.plot(x, cabinet * 100, "--s", lw=1.8, color="#A04C0A", alpha=0.85,
            label="Cabinet only (M+KD+L)", markersize=7)
    ax.plot(x, opp * 100, "-o", lw=2.5, color="#666",
            label="Opposition (S+V+MP+C)", markersize=10)
    for xi, v in zip(x, tido):
        ax.annotate(f"{v:.0%}", (xi, v * 100), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10,
                    color="#DD6E0F", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN)
    ax.set_ylabel("Share of votes with unanimous bloc majority")
    ax.set_ylim(0, 105)
    ax.set_title("a) Block-level unanimity",
                 fontsize=12, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", framealpha=0.9, fontsize=10)
    ax.axvspan(2.5, 3.5, color="#FFDDAA", alpha=0.2, zorder=0)
    ax.annotate("election year", xy=(3, 3), ha="center",
                fontsize=9, color="#A04C0A", style="italic")

    # Panel B: where the convergence came from — SD
    ax = axes[1]; ax.set_facecolor("#FAFAFA")
    ax.plot(x, sd_cab * 100, "-o", lw=2.5, color="#DDDD00",
            markeredgecolor="#888", markeredgewidth=1.5,
            label="SD ↔ M", markersize=10)
    ax.plot(x, sd_l * 100, "-^", lw=2.0, color="#006AB3",
            label="SD ↔ L", markersize=9)
    ax.plot(x, c_s * 100, "-s", lw=2.0, color="#009933",
            label="C ↔ S (for contrast)", markersize=9, alpha=0.7)
    for xi, v in zip(x, sd_cab):
        ax.annotate(f"{v:.0%}", (xi, v * 100), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10,
                    color="#A07A00", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN)
    ax.set_ylabel("Pairwise majority-vote agreement")
    ax.set_ylim(0, 105)
    ax.set_title("b) The convergence is driven by SD adopting cabinet positions",
                 fontsize=12, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", framealpha=0.9, fontsize=10)
    ax.axvspan(2.5, 3.5, color="#FFDDAA", alpha=0.2, zorder=0)

    fig.tight_layout()
    out = FIG / "08_agreement_trend.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
