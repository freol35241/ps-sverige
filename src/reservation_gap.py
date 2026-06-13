"""Compare speech-based vs reservation-based WHY signal, and update gap analysis.

Reservations are concentrated party-reasoning text — much higher signal density
than chamber speeches, but cabinet parties (L, KD, M) have very few because they
write the majority opinion, not the dissent.

For pair similarity:
- opposition-opposition pairs: use reservation vectors (concentrated)
- pairs involving a cabinet party: fall back to speech vectors

We also output a side-by-side comparison so the differences are visible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"]
OPPOSITION = {"V", "S", "MP", "C"}
CABINET = {"L", "KD", "M"}


def centered_pair_cosines(V, N) -> np.ndarray:
    """Per-topic centred pair cosines, averaged over filled topics."""
    Vc = V.copy()
    for t in range(V.shape[1]):
        present = N[:, t] > 0
        if present.sum() >= 2:
            Vc[present, t] -= Vc[present, t].mean(axis=0, keepdims=True)
    M = np.full((V.shape[0], V.shape[0]), np.nan)
    for i in range(V.shape[0]):
        for j in range(V.shape[0]):
            if i == j:
                M[i, j] = 1.0; continue
            sims = []
            for t in range(V.shape[1]):
                if N[i, t] == 0 or N[j, t] == 0:
                    continue
                vi, vj = Vc[i, t], Vc[j, t]
                ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
                if ni == 0 or nj == 0:
                    continue
                sims.append(float(vi @ vj / (ni * nj)))
            if sims:
                M[i, j] = float(np.mean(sims))
    return M


def main() -> None:
    speech = np.load(IN / "reasoning_vectors.npz", allow_pickle=True)
    res = np.load(IN / "reservation_vectors.npz", allow_pickle=True)
    speech_M = centered_pair_cosines(speech["V"], speech["N"])
    res_M = centered_pair_cosines(res["V"], res["N"])

    speech_df = pd.DataFrame(speech_M, index=PARTIES, columns=PARTIES)
    res_df = pd.DataFrame(res_M, index=PARTIES, columns=PARTIES)
    print("Speech-based WHY (per-topic centred cosine):")
    print(speech_df.round(2).to_string())
    print("\nReservation-based WHY:")
    print(res_df.round(2).to_string())

    # Coverage check
    res_cov = np.isfinite(res_M).astype(int)
    speech_cov = np.isfinite(speech_M).astype(int)
    print(f"\nreservation matrix coverage: {res_cov.sum()} cells")
    print(f"speech matrix coverage: {speech_cov.sum()} cells")
    print(f"missing-from-reservation cells (cabinet pairs):")
    for i, a in enumerate(PARTIES):
        for j, b in enumerate(PARTIES):
            if i < j and not np.isfinite(res_M[i, j]):
                print(f"  {a}-{b}")

    # Combined: prefer reservation where available, fall back to speech.
    combined = np.where(np.isfinite(res_M), res_M, speech_M)
    combined_df = pd.DataFrame(combined, index=PARTIES, columns=PARTIES)
    print("\nCombined WHY (reservation > speech fallback):")
    print(combined_df.round(2).to_string())

    # Compare findings: which pairs change most between the two sources?
    print("\nPairs where reservation WHY differs most from speech WHY (opposition only):")
    rows = []
    for i, a in enumerate(PARTIES):
        for j in range(i + 1, len(PARTIES)):
            b = PARTIES[j]
            if not (a in OPPOSITION and b in OPPOSITION):
                continue
            sv, rv = speech_M[i, j], res_M[i, j]
            if not (np.isfinite(sv) and np.isfinite(rv)):
                continue
            rows.append({"pair": f"{a}-{b}",
                         "speech_why": sv, "reservation_why": rv,
                         "delta": rv - sv})
    if rows:
        delta_df = pd.DataFrame(rows).sort_values("delta")
        print(delta_df.round(3).to_string(index=False))

    # === Figure: side-by-side WHY matrices ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=140,
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.patch.set_facecolor("#FAFAFA")
    for ax, mat, title in [
        (axes[0], speech_df.to_numpy(), "WHY — from debate speeches\n(broader source, mixed signal)"),
        (axes[1], res_df.to_numpy(), "WHY — from reservation reasoning\n(concentrated; cabinets blank)"),
        (axes[2], combined_df.to_numpy(), "WHY — combined\n(reservation where available)"),
    ]:
        ax.set_facecolor("#FAFAFA")
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="equal")
        ax.set_xticks(range(len(PARTIES))); ax.set_xticklabels(PARTIES, fontsize=11)
        ax.set_yticks(range(len(PARTIES))); ax.set_yticklabels(PARTIES, fontsize=11)
        for i in range(len(PARTIES)):
            for j in range(len(PARTIES)):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=10, color="#999")
                    continue
                c = "white" if abs(v) > 0.22 else "#333"
                ax.text(j, i, f"{v:+.2f}".replace("+", ""),
                        ha="center", va="center", fontsize=8, color=c)
        ax.set_title(title, fontsize=11, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("WHY refined — reservation reasoning has higher signal density",
                 fontsize=13, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "18_why_speech_vs_reservation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Save the combined matrix for downstream use.
    np.savez_compressed(IN / "why_combined.npz",
                        M=combined,
                        parties=np.array(PARTIES, dtype=object))


if __name__ == "__main__":
    main()
