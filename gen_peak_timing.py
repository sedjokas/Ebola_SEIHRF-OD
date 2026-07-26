"""
gen_peak_timing.py
===================
Estimates when the SEIHRF-OD epidemic curve is projected to peak (maximum
daily incidence), under the calibrated posterior and two assumptions:
  (a) Baseline: current dynamics continue unchanged (C(t) held at its last
      documented anchor level from day 46 onward; no new interventions).
  (b) S1+S3 combined intervention (enhanced communication from day 14 +
      zero body reclamation from day 0), for comparison.

This requires a much longer simulation horizon than the day-90 window used
elsewhere in the manuscript, because the catchment population (8.3M) is
large relative to current case counts, so classical susceptible-depletion
does not occur until much later than day 90 unless the opinion-driven
shift toward compliance (mu_NB) meaningfully outpaces new infections.

Output: imgs/fig_peak_timing.pdf / .png, and printed summary statistics
for the manuscript text (median peak day, peak daily incidence, cumulative
deaths at peak, 95% CrI from posterior draws).

CAVEAT (see manuscript Limitations): this is a long-horizon extrapolation
far beyond the validated 71-day calibration window and assumes no change
in policy, security conditions, or population behavior beyond what is
already encoded in the fitted opinion-dynamics parameters. It should be
read as illustrating the trajectory implied by current dynamics, not as
an operational forecast.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

LANCET = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"
OUTDIR = os.path.join(LANCET, "imgs")

FIXED = dict(
    beta_H   = 0.06,
    beta_FS  = 0.002,
    kappa    = 1.0 / 9.0,
    theta_B  = 0.28,
    delta_I  = 0.18,
    delta_H  = 0.12,
    gamma_I  = 0.09,
    gamma_H  = 0.10,
    psi_I    = 0.45,
    psi_H    = 0.15,
    omega_FR = 0.80,
    omega_FS = 3.00,
    beta_D   = 8.00,
    N_pop    = 10_877_533.0,
    seed_total = 8.0,
)

# Twelve documented anchors (declaration-day units, day 0 = 15 May 2026), the
# last one (day 65, C=0.85) is held constant indefinitely for the long-run
# projection -- there is no documented basis for assuming further
# escalation or de-escalation beyond the 23 July 2026 data freeze.
CT_ANCHORS = [
    (0.0,  0.55), (3.0, 0.65), (6.0, 1.00), (9.0, 0.60),
    (18.0, 0.75), (19.0, 1.00), (23.0, 0.70), (31.0, 0.60), (44.0, 0.70),
    (53.0, 0.65), (56.0, 0.75), (65.0, 0.85),
]


def C_func(t, scale=1.0):
    c = CT_ANCHORS[0][1]
    for start, level in CT_ANCHORS:
        if t >= start:
            c = level
    return c * scale


def rhs(t, y, p, gc_scale=1.0, bFR_scale=1.0, gc_start=None):
    SB, EB, IB, HB, RB, SN, EN, IN, HN, RN, FR, FS, Dcum = y
    N = SB + EB + IB + HB + RB + SN + EN + IN + HN + RN
    if N <= 0:
        return [0.0] * 13
    gc_mult = gc_scale if (gc_start is None or t >= gc_start) else 1.0
    gc  = p["gamma_comm"] * gc_mult
    bFR = p["beta_FR"] * bFR_scale
    Ct  = C_func(t)

    lam_B = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + p["beta_FS"]*FS) / N
    lam_N = (p["beta_I"]*(IB+IN) + p["beta_H"]*(HB+HN) + bFR*FR) / N
    Dvis  = (p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)) / N
    phi   = SN / (SB + SN) if (SB + SN) > 0 else p["phi0"]
    mu_BN = p["alpha"]*phi + p["delta_C"]*Ct
    mu_NB = gc + p["beta_D"]*Dvis

    dSB = -lam_B*SB - mu_BN*SB + mu_NB*SN
    dEB =  lam_B*SB - p["kappa"]*EB
    dIB =  p["kappa"]*EB - (p["theta_B"] + p["delta_I"] + p["gamma_I"])*IB
    dHB =  p["theta_B"]*IB - (p["delta_H"] + p["gamma_H"])*HB
    dRB =  p["gamma_I"]*IB + p["gamma_H"]*HB
    dSN = -lam_N*SN + mu_BN*SB - mu_NB*SN
    dEN =  lam_N*SN - p["kappa"]*EN
    dIN =  p["kappa"]*EN - (p["theta_N"] + p["delta_I"] + p["gamma_I"])*IN
    dHN =  p["theta_N"]*IN - (p["delta_H"] + p["gamma_H"])*HN
    dRN =  p["gamma_I"]*IN + p["gamma_H"]*HN
    dFR = (p["psi_I"]*p["delta_I"]*IN + p["psi_H"]*p["delta_H"]*HN - p["omega_FR"]*FR)
    dFS = (p["delta_I"]*IB + p["delta_H"]*HB
           + (1-p["psi_I"])*p["delta_I"]*IN
           + (1-p["psi_H"])*p["delta_H"]*HN
           - p["omega_FS"]*FS)
    dDcum = p["delta_I"]*(IB+IN) + p["delta_H"]*(HB+HN)
    return [dSB, dEB, dIB, dHB, dRB, dSN, dEN, dIN, dHN, dRN, dFR, dFS, dDcum]


def make_y0(phi0):
    N, seed_total = FIXED["N_pop"], FIXED["seed_total"]
    IB0, IN0 = (1-phi0)*seed_total, phi0*seed_total
    return [(1-phi0)*N - IB0, 0, IB0, 0, 0,
            phi0*N - IN0,     0, IN0, 0, 0,
            0, 0, 0]


def run(row, T_MAX, gc_scale=1.0, bFR_scale=1.0, gc_start=None, n_eval=2000):
    p = {**FIXED, **row}
    t_eval = np.linspace(0, T_MAX, n_eval)
    sol = solve_ivp(
        rhs, (0, T_MAX), make_y0(row["phi0"]),
        args=(p, gc_scale, bFR_scale, gc_start),
        t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
    )
    EB, EN = sol.y[1], sol.y[6]
    incidence = p["kappa"] * (EB + EN)
    return t_eval, incidence, sol.y[12]


def find_peak(t_eval, incidence):
    i = np.argmax(incidence)
    return t_eval[i], incidence[i]


if __name__ == "__main__":
    draws = pd.read_csv(os.path.join(LANCET, "posterior_draws.csv"))
    T_MAX = 730   # 2-year horizon to ensure the peak is captured

    print("Running baseline long-horizon simulation at posterior median...")
    med = draws.median(numeric_only=True).to_dict()
    t_eval, inc_med, deaths_med = run(med, T_MAX)
    peak_day_med, peak_inc_med = find_peak(t_eval, inc_med)
    print(f"  Posterior-median peak day = {peak_day_med:.0f}  "
          f"peak daily incidence = {peak_inc_med:.0f}  "
          f"cumulative deaths at peak = {np.interp(peak_day_med, t_eval, deaths_med):.0f}")

    print("\nRunning posterior draws for uncertainty band (n=150)...")
    rng = np.random.default_rng(42)
    sample = draws.iloc[rng.choice(len(draws), size=150, replace=False)]
    peak_days, peak_incs, peak_deaths = [], [], []
    inc_curves = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if (i + 1) % 30 == 0:
            print(f"  draw {i+1}/150")
        t_e, inc, deaths = run(dict(row), T_MAX, n_eval=1500)
        pd_, pi_ = find_peak(t_e, inc)
        peak_days.append(pd_)
        peak_incs.append(pi_)
        peak_deaths.append(np.interp(pd_, t_e, deaths))
        inc_curves.append(np.interp(t_eval, t_e, inc))

    peak_days = np.array(peak_days)
    peak_incs = np.array(peak_incs)
    peak_deaths = np.array(peak_deaths)
    inc_curves = np.array(inc_curves)

    print(f"\n=== Baseline peak-timing summary (posterior draws) ===")
    print(f"Peak day: median={np.median(peak_days):.0f}  "
          f"95% CrI [{np.percentile(peak_days,2.5):.0f}, {np.percentile(peak_days,97.5):.0f}]")
    print(f"Peak daily incidence: median={np.median(peak_incs):.0f}  "
          f"95% CrI [{np.percentile(peak_incs,2.5):.0f}, {np.percentile(peak_incs,97.5):.0f}]")
    print(f"Cumulative deaths at peak: median={np.median(peak_deaths):.0f}  "
          f"95% CrI [{np.percentile(peak_deaths,2.5):.0f}, {np.percentile(peak_deaths,97.5):.0f}]")

    decl = pd.Timestamp("2026-05-15")
    peak_date_med = decl + pd.Timedelta(days=float(np.median(peak_days)))
    print(f"Calendar date at median peak day: {peak_date_med:%d %B %Y}")

    # ── S1+S3 combined intervention comparison ────────────────────────────────
    print("\nRunning S1+S3 combined-intervention long-horizon simulation...")
    t_e13, inc13_med, deaths13_med = run(med, T_MAX, gc_scale=2.0, bFR_scale=0.0, gc_start=14.0)
    peak_day13, peak_inc13 = find_peak(t_e13, inc13_med)
    print(f"  S1+S3 peak day = {peak_day13:.0f}  peak daily incidence = {peak_inc13:.0f}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    lo95 = np.percentile(inc_curves, 2.5, axis=0)
    hi95 = np.percentile(inc_curves, 97.5, axis=0)
    med_curve = np.median(inc_curves, axis=0)

    LANCET_BLUE = "#004E7D"
    CORAL       = "#E8735A"
    TEAL        = "#1D9E75"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: full long-horizon trajectory with peak marked
    ax1.fill_between(t_eval, lo95, hi95, color=LANCET_BLUE, alpha=0.18,
                      label="95% CrI (posterior)")
    ax1.plot(t_eval, inc_med, color=LANCET_BLUE, lw=2.0, label="Baseline (posterior median)")
    ax1.plot(t_e13, inc13_med, color=TEAL, lw=1.8, ls="--", label="S1+S3 combined intervention")
    ax1.axvline(np.median(peak_days), color=CORAL, lw=1.2, ls=":")
    ax1.plot(np.median(peak_days), np.median(peak_incs), "o", color=CORAL, ms=7, zorder=5)
    ax1.annotate(
        f"Peak: day {np.median(peak_days):.0f}\n({peak_date_med:%d %b %Y})",
        xy=(np.median(peak_days), np.median(peak_incs)),
        xytext=(np.median(peak_days) + 40, np.median(peak_incs) * 0.85),
        fontsize=8.5, color=CORAL,
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))
    ax1.axvspan(0, 71, color="gray", alpha=0.08, lw=0)
    ax1.text(26, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1, "",
             fontsize=1)  # placeholder to keep autoscale stable before text below
    ax1.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax1.set_ylabel("Daily incidence (model-predicted new infectious)")
    ax1.set_title("(A)  Long-horizon projection and estimated epidemic peak")
    ax1.legend(fontsize=8, loc="upper right")

    # Panel B: zoom on the observed/near-term window for context
    zoom_days = 150
    mask = t_eval <= zoom_days
    ax2.fill_between(t_eval[mask], lo95[mask], hi95[mask], color=LANCET_BLUE, alpha=0.18)
    ax2.plot(t_eval[mask], inc_med[mask], color=LANCET_BLUE, lw=2.0, label="Baseline")
    mask13 = t_e13 <= zoom_days
    ax2.plot(t_e13[mask13], inc13_med[mask13], color=TEAL, lw=1.8, ls="--",
              label="S1+S3 combined")
    ax2.axvspan(0, 71, color="gray", alpha=0.12, lw=0, label="Calibration window (observed)")
    ax2.set_xlabel("Days since outbreak declaration (15 May 2026)")
    ax2.set_ylabel("Daily incidence")
    ax2.set_title(f"(B)  Near-term window (days 0-{zoom_days})")
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Projected epidemic peak under current dynamics (extrapolation beyond the\n"
        "71-day calibration window; assumes no new interventions or security change)",
        fontsize=9.5, y=1.03)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(OUTDIR, f"fig_peak_timing.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    plt.close(fig)

    # Save summary CSV for manuscript reference
    summary = pd.DataFrame({
        "quantity": ["peak_day", "peak_daily_incidence", "cum_deaths_at_peak"],
        "median": [np.median(peak_days), np.median(peak_incs), np.median(peak_deaths)],
        "lo95": [np.percentile(peak_days, 2.5), np.percentile(peak_incs, 2.5), np.percentile(peak_deaths, 2.5)],
        "hi95": [np.percentile(peak_days, 97.5), np.percentile(peak_incs, 97.5), np.percentile(peak_deaths, 97.5)],
    })
    summary.to_csv(os.path.join(LANCET, "peak_timing_results.csv"), index=False)
    print("\nSaved peak_timing_results.csv")
    print("\nDone.")
