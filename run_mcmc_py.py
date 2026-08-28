"""
run_mcmc_py.py
==============
Python equivalent of run_mcmc.R using CmdStanPy.
Calibrates the SEIHRF-OD model on the latest INRB-UMIE data
(data freeze 25 August 2026, build 1819da2).

Requirements (already installed):
    cmdstanpy == 1.3.0
    numpy, scipy, pandas
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# ── Paths ──────────────────────────────────────────────────────────────────────
LANCET = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(LANCET, "data")
STAN   = os.path.join(LANCET, "seihrf_od.stan")

# ── 1. Load and prepare case data ─────────────────────────────────────────────
print("Loading data …")

# National daily new-confirmed-cases series, derived from the cumulative INSP
# SitRep series (data/update_2026_08_25/daily_new_cases_deaths_derived.csv):
# the zone-level "new confirmed cases" file is not maintained past 1 June 2026
# in the source repository, so daily increments are recovered from the
# cumulative national series, distributing multi-day reporting gaps evenly
# across the missing calendar days.
derived = pd.read_csv(
    os.path.join(DATA, "update_2026_08_25", "daily_new_cases_deaths_derived.csv"),
    parse_dates=["date"],
)

y_cases = derived["new_confirmed_cases_est"].round().astype(int).tolist()
T = len(y_cases)

print(f"T = {T} days | Total cases = {sum(y_cases)}")
print(f"Date range: {derived['date'].iloc[0].date()} → {derived['date'].iloc[-1].date()}")

# ── 2. phi0 seed from contact-tracing proxy ───────────────────────────────────
# phi0_obs = 1 - mean(r_c) over first 5 reporting days (unchanged across
# freezes: phi0 is the *initial* scepticism fraction at outbreak onset,
# not re-estimated as the series lengthens).
phi0_obs    = 0.38
phi0_obs_sd = 0.05

# ── 3. Build Stan data dictionary ─────────────────────────────────────────────
stan_data = {
    "T":           T,
    "y_cases":     y_cases,
    # Catchment population: sum of WorldPop GRID3 v4.4 counts across the 58
    # health zones with confirmed cases as of this freeze (build 1819da2),
    # dedup'd via the repository's own data/aliases.csv reconciliation table.
    "N_pop":       12_996_531.0,
    "phi0_obs":    phi0_obs,
    "phi0_obs_sd": phi0_obs_sd,
    # Twelve conflict anchors: [start_day, level] × 12. Day 1 = 14 May 2026.
    # Unchanged since the 71-day freeze: no new anchors have been formally
    # calibrated for the 24 Jul-25 Aug window (see README "What changed" /
    # manuscript Limitations). C(t) holds at the last anchor's level (0.85)
    # for all t beyond day 67 -- see seihrf_od.stan's conflict_C().
    "x_r_conflict": [
         1.0, 0.55,   # anchor 1:  window opens post-Nyankunde exposure, 11 May
         5.0, 0.65,   # anchor 2:  CDC announcement + Berlin evacuation, 18 May
         8.0, 1.00,   # anchor 3:  peak cluster I, Rwampara/Mongbwalu, 21-23 May
        11.0, 0.60,   # anchor 4:  persistent insecurity I, 24 May onward
        20.0, 0.75,   # anchor 5:  renewed escalation, Bunia/Katana attacks, 2 Jun
        21.0, 1.00,   # anchor 6:  peak cluster II, Oicha/Mbau massacre, 3 Jun
        25.0, 0.70,   # anchor 7:  persistent insecurity II, Nyamurongo attack, 7 Jun
        33.0, 0.60,   # anchor 8:  partial de-escalation, 15 Jun
        46.0, 0.70,   # anchor 9:  renewed disruption, PoC burned + attacks, late Jun-early Jul
        55.0, 0.65,   # anchor 10: healthcare-provider strike, Bunia/Rwampara, 7-9 Jul
        58.0, 0.75,   # anchor 11: repeat PoC vandalism + provider threats, 10 Jul
        67.0, 0.85,   # anchor 12: Muchanga bridge attack + multi-zone resistance, 19 Jul
    ],
    "rel_tol":   1e-6,
    "abs_tol":   1e-8,
    "max_steps": 10_000,
}

# Save for reproducibility
with open(os.path.join(LANCET, "stan_data.json"), "w") as f:
    json.dump(stan_data, f, indent=2)
print("Stan data saved → stan_data.json")

# ── 4. Compile Stan model ─────────────────────────────────────────────────────
print("\nCompiling Stan model …")
model = CmdStanModel(stan_file=STAN)
print("Compilation complete.")

# ── 5. Run MCMC ───────────────────────────────────────────────────────────────
print("\nRunning MCMC (4 chains × 2000 warmup + 2000 sampling) …")
print("This will take 30–90 minutes depending on CPU speed.\n")

fit = model.sample(
    data            = stan_data,
    chains          = 4,
    parallel_chains = 4,
    iter_warmup     = 2000,
    iter_sampling   = 2000,
    adapt_delta     = 0.95,
    max_treedepth   = 12,
    seed            = 42,
    show_progress   = True,
    output_dir      = LANCET,
)

# ── 6. Convergence diagnostics ────────────────────────────────────────────────
print("\n=== Convergence summary ===")
params = ["beta_I", "beta_FR", "phi0", "theta_N", "alpha",
          "gamma_comm", "delta_C", "phi_obs", "R0"]
summary = fit.summary(sig_figs=4)

# CmdStanPy >= 1.1 uses ESS_bulk; older versions use N_Eff
ess_col = "ESS_bulk" if "ESS_bulk" in summary.columns else "N_Eff"
display_cols = [c for c in ["Mean", "StdDev", "5%", "50%", "95%", "R_hat", ess_col]
                if c in summary.columns]

key_rows = summary[summary.index.isin(params)]
print(key_rows[display_cols].to_string())

rhat_vals = summary["R_hat"].dropna()
ess_vals  = summary[ess_col].dropna()
print(f"\nMax R-hat: {rhat_vals.max():.4f}  (should be < 1.02)")
print(f"Min ESS:   {ess_vals.min():.0f}  (should be > 400)")

# ── 7. Export posterior draws ─────────────────────────────────────────────────
draws_df = fit.draws_pd(vars=params)
out_csv  = os.path.join(LANCET, "posterior_draws.csv")
draws_df.to_csv(out_csv, index=False)
print(f"\nPosterior draws saved → {out_csv}")
print(f"Shape: {draws_df.shape[0]} draws × {draws_df.shape[1]} columns")

# ── 8. Quick posterior summaries for manuscript ───────────────────────────────
print("\n=== Posterior medians and 95% CrI ===")
for p in params:
    if p in draws_df.columns:
        vals = draws_df[p]
        print(f"  {p:<15} median={vals.median():.3f}  "
              f"95%CrI [{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")

print("\nDone.")
