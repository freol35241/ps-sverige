"""Web-publication renders of the five headline figures plus a new
"SD reservation collapse" figure.

Outputs land in `figures/web/` and are mirrored to `site/public/figures/`
under the same filenames as the analytical originals so existing
<img> tags in the article keep working.

These figures are tuned for ~1200 px width, displayed at ~720 px container.
Fonts and margins are sized so labels remain legible on mobile.
"""
from __future__ import annotations

import shutil
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import PARTY_COLOR, PARTY_ORDER

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
WEB = ROOT / "figures" / "web"
SITE = ROOT / "site" / "public" / "figures"

RIKSMOTEN = ["2022/23", "2023/24", "2024/25", "2025/26"]
BG = "#FAFAFA"
INK = "#222222"
MUTED = "#888888"
GRID = "#DDDDDD"

# Global rcParams – bigger fonts, sans-serif, light grid.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.titlepad": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": BG,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})


def _save(fig: plt.Figure, name: str) -> Path:
    WEB.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    out = WEB / name
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    shutil.copyfile(out, SITE / name)
    return out


# ---------------------------------------------------------------------------
# 1. Behavioural map
# ---------------------------------------------------------------------------
def fig_behavioural_map() -> Path:
    emb = pd.read_parquet(IN / "mp_embedding.parquet")
    pca = np.load(IN / "pca.npz", allow_pickle=True)
    var = pca["var_ratio"]
    fit = emb[emb["in_fit"]].copy()

    fig, ax = plt.subplots(figsize=(12, 7.5))

    for party in PARTY_ORDER:
        sub = fit[fit["party_modal"] == party]
        if sub.empty:
            continue
        ax.scatter(sub["PC1"], sub["PC2"],
                   s=45, c=PARTY_COLOR[party],
                   edgecolor="white", linewidth=0.6,
                   alpha=0.85, zorder=3)

    cents = fit.groupby("party_modal")[["PC1", "PC2"]].mean()
    # KD and M nearly coincide – nudge KD label down so both badges are visible.
    label_offset = {"KD": (0, -3.2), "M": (0, 1.5)}
    for party, row in cents.iterrows():
        if party == "-":
            continue
        dx, dy = label_offset.get(party, (0, 0))
        bx, by = row["PC1"] + dx, row["PC2"] + dy
        if (dx, dy) != (0, 0):
            ax.plot([row["PC1"], bx], [row["PC2"], by],
                    color=PARTY_COLOR.get(party, "#444"),
                    lw=1.0, alpha=0.6, zorder=4)
        ax.annotate(party, (bx, by),
                    fontsize=14, fontweight="bold",
                    ha="center", va="center",
                    color="white",
                    bbox=dict(boxstyle="circle,pad=0.45",
                              fc=PARTY_COLOR.get(party, "#444"),
                              ec="white", lw=2),
                    zorder=5)

    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1 – regering mot opposition  ({var[0]:.0%} av variansen)")
    ax.set_ylabel(f"PC2 – struktur inom blocken  ({var[1]:.0%} av variansen)")
    ax.set_title("Riksdagens beteendekarta, 2022–2026",
                 loc="left")

    # Subtle directional hints, no legend needed – party badges are the legend.
    ax.annotate("Tidöblocket", xy=(0.985, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=11, color=MUTED, style="italic")
    ax.annotate("opposition", xy=(0.015, 0.03), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=11, color=MUTED, style="italic")

    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "01_behavioural_map.png")


# ---------------------------------------------------------------------------
# 7. Tidö collapse – four small multiples
# ---------------------------------------------------------------------------
def fig_tido_collapse() -> Path:
    long = pd.read_parquet(IN / "embedding_yearly.parquet")
    fit = long[long["in_fit"]]
    cents = (fit.groupby(["party_modal", "rm"])[["PC1", "PC2"]]
                .mean().reset_index())
    parties = ["M", "KD", "L", "SD"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), sharex=True, sharey=True)

    tido = fit[fit["party_modal"].isin(parties)]
    x_lo, x_hi = tido["PC1"].min() - 1, tido["PC1"].max() + 1
    y_lo, y_hi = tido["PC2"].min() - 1, tido["PC2"].max() + 1

    for ax, rm in zip(axes, RIKSMOTEN):
        sub = fit[(fit["rm"] == rm) & (fit["party_modal"].isin(parties))]
        for party in parties:
            ps = sub[sub["party_modal"] == party]
            ax.scatter(ps["PC1"], ps["PC2"], s=28,
                       c=PARTY_COLOR[party], edgecolor="white", linewidth=0.5,
                       alpha=0.85, zorder=3)
        cs = cents[(cents["rm"] == rm) & (cents["party_modal"].isin(parties))]
        for _, row in cs.iterrows():
            ax.annotate(row["party_modal"], (row["PC1"], row["PC2"]),
                        fontsize=11, fontweight="bold", ha="center", va="center",
                        color="white",
                        bbox=dict(boxstyle="circle,pad=0.30",
                                  fc=PARTY_COLOR.get(row["party_modal"], "#444"),
                                  ec="white", lw=1.2),
                        zorder=5)

        cs_xy = {p: (cs[cs["party_modal"] == p][["PC1", "PC2"]].iloc[0].to_numpy()
                     if (cs["party_modal"] == p).any() else None)
                 for p in parties}
        dists = [np.linalg.norm(cs_xy[a] - cs_xy[b])
                 for a, b in combinations(parties, 2)
                 if cs_xy[a] is not None and cs_xy[b] is not None]
        mean_d = float(np.mean(dists)) if dists else float("nan")

        ax.set_title(rm, loc="left", fontsize=13)
        ax.text(0.97, 0.97, f"d = {mean_d:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold",
                color="#A04C0A" if mean_d < 2 else INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#DDD", lw=0.8))
        ax.axhline(0, color=GRID, lw=0.5, zorder=1)
        ax.axvline(0, color=GRID, lw=0.5, zorder=1)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.tick_params(length=0, labelsize=10)

    axes[0].set_ylabel("PC2")
    for ax in axes:
        ax.set_xlabel("PC1")

    fig.suptitle("Tidöblocket smälter samman",
                 x=0.02, ha="left", y=1.02, fontsize=15, fontweight="bold")
    fig.text(0.02, 0.96,
             "Genomsnittligt parvis avstånd (d) mellan M, KD, L, SD",
             ha="left", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "07_tido_collapse.png")


# ---------------------------------------------------------------------------
# 8. Agreement trend – two-panel line chart
# ---------------------------------------------------------------------------
def _majority_excl_absent(s: pd.Series):
    s = s[s != "Frånvarande"]
    if s.empty:
        return None
    return s.value_counts().idxmax()


def _agreement_per_year(long: pd.DataFrame, parties: list[str]) -> pd.Series:
    out = {}
    for rm in RIKSMOTEN:
        sub = long[long["rm"] == rm]
        pm = (sub.groupby(["votering_id", "parti"])["rost"]
                 .agg(_majority_excl_absent)
                 .unstack("parti"))
        block = pm[parties].dropna()
        out[rm] = (block.nunique(axis=1) == 1).mean() if len(block) else np.nan
    return pd.Series(out)


def _pairwise(long: pd.DataFrame, a: str, b: str) -> pd.Series:
    out = {}
    for rm in RIKSMOTEN:
        sub = long[long["rm"] == rm]
        pm = (sub.groupby(["votering_id", "parti"])["rost"]
                 .agg(_majority_excl_absent)
                 .unstack("parti"))
        block = pm[[a, b]].dropna()
        out[rm] = (block[a] == block[b]).mean() if len(block) else np.nan
    return pd.Series(out)


def fig_agreement_trend() -> Path:
    long = pd.read_parquet(IN / "votes_long.parquet")
    cabinet = _agreement_per_year(long, ["M", "KD", "L"])
    tido = _agreement_per_year(long, ["M", "KD", "L", "SD"])
    opp = _agreement_per_year(long, ["S", "V", "MP", "C"])
    sd_m = _pairwise(long, "SD", "M")
    s_v = _pairwise(long, "S", "V")
    v_mp = _pairwise(long, "V", "MP")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(RIKSMOTEN))

    # --- Panel A: bloc-level
    ax = axes[0]
    ax.axvspan(2.5, 3.5, color="#FFE9C7", alpha=0.55, zorder=0)
    ax.plot(x, tido * 100, "-o", lw=3, color="#DD6E0F",
            label="Tidöblocket (M+KD+L+SD)", markersize=10, zorder=4)
    ax.plot(x, cabinet * 100, "--s", lw=1.8, color="#A04C0A", alpha=0.85,
            label="Endast kabinettet (M+KD+L)", markersize=7, zorder=3)
    ax.plot(x, opp * 100, "-o", lw=2.2, color="#666",
            label="Oppositionen (S+V+MP+C)", markersize=8, zorder=3)
    for xi, v in zip(x, tido):
        ax.annotate(f"{v:.0%}", (xi, v * 100), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11,
                    color="#A04C0A", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN)
    ax.set_ylabel("Andel omröstningar med enad majoritet")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Blocksamsyn", loc="left")
    ax.grid(axis="y", alpha=0.3, color=GRID)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.text(3, 6, "election year", ha="center", fontsize=10,
            color="#A04C0A", style="italic")
    ax.tick_params(length=0)

    # --- Panel B: SD ↔ cabinet pair
    ax = axes[1]
    ax.axvspan(2.5, 3.5, color="#FFE9C7", alpha=0.55, zorder=0)
    ax.plot(x, sd_m * 100, "-o", lw=3, color="#C9B500",
            markeredgecolor="#666", markeredgewidth=1.2,
            label="SD ↔ M (cabinet)", markersize=10, zorder=4)
    ax.plot(x, v_mp * 100, "-^", lw=2.0, color="#7A7A7A",
            label="V ↔ MP (oppositionspar)", markersize=8, zorder=3)
    ax.plot(x, s_v * 100, "-s", lw=2.0, color="#AAAAAA",
            label="S ↔ V (oppositionspar)", markersize=8, zorder=3)
    for xi, v in zip(x, sd_m):
        ax.annotate(f"{v:.0%}", (xi, v * 100), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11,
                    color="#8A7E00", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN)
    ax.set_ylabel("Parvis röstöverensstämmelse")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("SD hoppar, oppositionsparen gör det inte", loc="left")
    ax.grid(axis="y", alpha=0.3, color=GRID)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.tick_params(length=0)

    fig.suptitle("Hur Tidöblocket samlades på en gemensam röstlinje",
                 x=0.02, ha="left", y=1.01, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "08_agreement_trend.png")


# ---------------------------------------------------------------------------
# 15. Pair scatter: WHAT vs WHY
# ---------------------------------------------------------------------------
PARTIES_8 = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]


def _what_why_matrices():
    grid = pd.read_parquet(IN / "party_topic.parquet")
    pivot = grid.pivot(index="parti", columns="topic_id",
                       values="score").loc[PARTIES_8]
    X = pivot.to_numpy()
    what = np.zeros((len(PARTIES_8), len(PARTIES_8)))
    for i in range(len(PARTIES_8)):
        for j in range(len(PARTIES_8)):
            what[i, j] = 1 - np.nanmean(np.abs(X[i] - X[j])) / 2.0

    rv = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    V, N = rv["V"], rv["N"]
    V_c = V.copy()
    for t in range(V.shape[1]):
        present = N[:, t] > 0
        if present.sum() >= 2:
            V_c[present, t] -= V_c[present, t].mean(axis=0, keepdims=True)

    why = np.zeros((len(PARTIES_8), len(PARTIES_8)))
    for i in range(len(PARTIES_8)):
        for j in range(len(PARTIES_8)):
            if i == j:
                why[i, j] = 1.0
                continue
            sims = []
            for t in range(V.shape[1]):
                if N[i, t] == 0 or N[j, t] == 0:
                    continue
                vi, vj = V_c[i, t], V_c[j, t]
                ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
                if ni == 0 or nj == 0:
                    continue
                sims.append(float(vi @ vj / (ni * nj)))
            why[i, j] = np.mean(sims) if sims else np.nan
    return what, why


def fig_pair_what_vs_why() -> Path:
    what, why = _what_why_matrices()
    rows = []
    for i, a in enumerate(PARTIES_8):
        for j in range(i + 1, len(PARTIES_8)):
            b = PARTIES_8[j]
            rows.append({"a": a, "b": b,
                         "what_sim": what[i, j], "why_sim": why[i, j]})
    pairs = pd.DataFrame(rows)

    # Trend line.
    xs = pairs["what_sim"].to_numpy()
    ys = pairs["why_sim"].to_numpy()
    a_fit, b_fit = np.polyfit(xs, ys, 1)

    # Story pairs.
    NARRATIVE = {
        ("S", "C"): ("marriage of convenience", "below"),
        ("M", "SD"): ("rhetorical alignment", "below"),
        ("V", "MP"): ("genuine pair", "above"),
    }

    fig, ax = plt.subplots(figsize=(12, 7.5))

    xline = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
    ax.plot(xline, a_fit * xline + b_fit, "--", color="#BBB", lw=1.2,
            label="trend (what predicts why)", zorder=1)

    for _, r in pairs.iterrows():
        key = (r["a"], r["b"]) if (r["a"], r["b"]) in NARRATIVE else \
              ((r["b"], r["a"]) if (r["b"], r["a"]) in NARRATIVE else None)
        residual = r["why_sim"] - (a_fit * r["what_sim"] + b_fit)

        if key is not None:
            color = "#1B9E77" if NARRATIVE[key][1] == "above" else "#D95F02"
            ax.scatter(r["what_sim"], r["why_sim"], s=240, alpha=0.95,
                       edgecolor=color, linewidth=2.2, c="white", zorder=5)
            ax.annotate(f"{r['a']}–{r['b']}",
                        (r["what_sim"], r["why_sim"]),
                        fontsize=13, fontweight="bold", color=color,
                        xytext=(10, 10), textcoords="offset points",
                        zorder=6)
        else:
            ax.scatter(r["what_sim"], r["why_sim"], s=110, alpha=0.55,
                       edgecolor="#555", linewidth=0.5, c="#9CB7D4",
                       zorder=3)
            # Light labels for the remaining pairs – small + grey.
            ax.annotate(f"{r['a']}–{r['b']}",
                        (r["what_sim"], r["why_sim"]),
                        fontsize=9, color="#777",
                        xytext=(5, 4), textcoords="offset points",
                        zorder=4)

    # Inline annotation boxes for the three story pairs.
    ax.text(0.985, 0.04,
            "Över trendlinjen (grön): närmare i retorik än i röster, delade värden men olika politik\n"
            "Under trendlinjen (orange): närmare i röster än i retorik, ett bekvämlighetsförbund",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#444",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#DDD", lw=0.8))

    ax.set_xlabel("WHAT – vote-stance similarity")
    ax.set_ylabel("WHY – speech-reasoning similarity")
    ax.set_title("Vad partierna gör mot hur de motiverar det", loc="left")
    ax.grid(alpha=0.3, color=GRID)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "15_pair_what_vs_why.png")


# ---------------------------------------------------------------------------
# 16. Three-axis parallel coordinates
# ---------------------------------------------------------------------------
def fig_three_axis() -> Path:
    pairs = pd.read_parquet(IN / "three_axis_pairs.parquet")

    cat_color = {"aligned": "#1B9E77",
                 "fragmented": "#D95F02",
                 "apart": "#7570B3",
                 "middle": "#BBBBBB"}
    cat_alpha = {"aligned": 1.0, "fragmented": 1.0, "apart": 0.85, "middle": 0.35}
    cat_lw = {"aligned": 2.6, "fragmented": 2.6, "apart": 1.8, "middle": 1.0}

    fig, ax = plt.subplots(figsize=(12, 7.5))
    x = np.array([0, 1, 2])

    # Background bands at each axis.
    for xi in x:
        ax.axvline(xi, color="#EAEAEA", lw=1.0, zorder=0)

    # Layer: middle first (background), then aligned/fragmented/apart on top.
    order = ["middle", "apart", "aligned", "fragmented"]
    for cat in order:
        sub = pairs[pairs["category"] == cat]
        for _, row in sub.iterrows():
            y = [row["what_sim_n"], row["why_sim_n"], row["how_sim_n"]]
            ax.plot(x, y, "-o",
                    color=cat_color[cat], alpha=cat_alpha[cat],
                    lw=cat_lw[cat], markersize=6,
                    zorder=2 if cat == "middle" else 4)
            if cat in ("aligned", "fragmented", "apart"):
                ax.annotate(row["pair"], (x[2], y[2]),
                            xytext=(8, 0), textcoords="offset points",
                            fontsize=11, va="center",
                            color=cat_color[cat], fontweight="bold",
                            zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(["WHAT\nvote stance", "WHY\nspeech reasoning",
                        "HOW\nco-authorship + lexicon"], fontsize=12)
    ax.set_ylabel("Parlikhet (rangnormaliserad)")
    ax.set_ylim(-0.06, 1.10)
    ax.set_xlim(-0.15, 2.55)
    ax.grid(axis="y", alpha=0.3, color=GRID)
    ax.tick_params(length=0)

    ax.set_title("Tre axlar av samsyn, 28 partipar", loc="left")

    handles = [plt.Line2D([0], [0], color=cat_color[c], lw=2.6, marker="o",
                          label=lbl)
               for c, lbl in [("aligned", "aligned on all three"),
                              ("fragmented", "axes disagree"),
                              ("apart", "apart on all three"),
                              ("middle", "middle")]]
    ax.legend(handles=handles, loc="upper left", framealpha=0.95,
              fontsize=10)

    fig.tight_layout()
    return _save(fig, "16_three_axis.png")


# ---------------------------------------------------------------------------
# 21. NEW – SD reservation collapse
# ---------------------------------------------------------------------------
def fig_sd_reservation_collapse() -> Path:
    r = pd.read_parquet(IN / "reservations.parquet")

    parties = ["V", "S", "MP", "C", "SD"]  # opposition + SD; cabinet writes ~0
    counts = {p: [] for p in parties}
    for rm in RIKSMOTEN:
        sub = r[r["rm"] == rm]
        for p in parties:
            n = sub["partier"].apply(lambda s: p in s.split(";")).sum()
            counts[p].append(int(n))

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(RIKSMOTEN))

    # Background band for election year.
    ax.axvspan(2.5, 3.5, color="#FFE9C7", alpha=0.55, zorder=0)
    ax.text(3, 1080, "valåret", ha="center", fontsize=11,
            color="#A04C0A", style="italic")

    # Draw non-SD parties in muted party colors first.
    for p in ["V", "S", "MP", "C"]:
        ax.plot(x, counts[p], "-o", color=PARTY_COLOR[p],
                lw=2.0, markersize=7, alpha=0.65, zorder=3, label=p)
        # Label end of line.
        ax.annotate(p, (x[-1], counts[p][-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=12, fontweight="bold",
                    color=PARTY_COLOR[p], va="center", zorder=5)

    # SD: thick yellow line, darker stroke for visibility on light background.
    sd_y = counts["SD"]
    ax.plot(x, sd_y, "-o", color=PARTY_COLOR["SD"],
            lw=4.5, markersize=13,
            markeredgecolor="#666", markeredgewidth=1.5,
            label="SD", zorder=6)

    # Annotate every SD point.
    for xi, v in zip(x, sd_y):
        offset = (0, 16) if v < 100 else (0, -22)
        ax.annotate(f"{v}", (xi, v), textcoords="offset points",
                    xytext=offset, ha="center", fontsize=12,
                    fontweight="bold", color="#8A7E00", zorder=7)
    # SD end-of-line label.
    ax.annotate("SD", (x[-1], sd_y[-1]),
                xytext=(12, 6), textcoords="offset points",
                fontsize=14, fontweight="bold",
                color="#8A7E00", va="center", zorder=7)

    # Drop arrow + caption (placed to the left of the SD line so it doesn't cover).
    ax.annotate("",
                xy=(2.95, 60), xytext=(2.45, 60),
                arrowprops=dict(arrowstyle="->", lw=2.4, color="#8A7E00"),
                zorder=6)
    ax.text(2.42, 60,
            "SD slutar skriva\nreservationer\ni valåret",
            ha="right", va="center", fontsize=12,
            color="#8A7E00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#E8DD9A", lw=1))

    ax.set_xticks(x); ax.set_xticklabels(RIKSMOTEN)
    ax.set_ylabel("Antal reservationer per parti")
    ax.set_ylim(-30, 1150)
    ax.set_title("SD:s reservationer försvinner i valåret", loc="left",
                 pad=28)
    ax.text(0.0, 1.02,
            "Reservationer är formella avvikelser från utskottets förslag. "
            "Regeringspartierna (M, KD, L) skriver nästan inga.",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color=MUTED)

    ax.grid(axis="y", alpha=0.3, color=GRID)
    ax.tick_params(length=0)
    fig.tight_layout()
    return _save(fig, "21_sd_reservation_collapse.png")


# ---------------------------------------------------------------------------
def main() -> None:
    for f in (fig_behavioural_map,
              fig_tido_collapse,
              fig_agreement_trend,
              fig_pair_what_vs_why,
              fig_three_axis,
              fig_sd_reservation_collapse):
        p = f()
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
