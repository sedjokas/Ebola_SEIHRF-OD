"""
homogeneous_r0.py
==================
Fits a homogeneous (non-stratified, no opinion dynamics, no F_R compartment)
SEIRH model to the same 52-day confirmed-case series via NegBin maximum
likelihood, then computes its implied R0 for comparison against the
SEIHRF-OD posterior. This is the "homogeneous model fitted to the same
data" comparator quoted in the manuscript (Summary, Results, Discussion).

Only beta_I_hom is estimated; all other rates are fixed at the same
values used elsewhere in the model. theta_hom is the population-weighted
average of theta_B and theta_N at the new posterior-median phi0, so the
homogeneous model has the same aggregate hospitalisation behaviour as the
stratified model (apples-to-apples comparison).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

LANCET = "/Users/selainkaserekakabunga/Documents/Lancet_Paper"

draws = pd.read_csv(os.path.join(LANCET, "posterior_draws.csv"))
phi0_med    = draws["phi0"].median()
theta_N_med = draws["theta_N"].median()
phi_obs_med = draws["phi_obs"].median()

THETA_B = 0.28
THETA_HOM = (1 - phi0_med) * THETA_B + phi0_med * theta_N_med

FIXED = dict(
    beta_H   = 0.06,
    kappa    = 1.0 / 9.0,
    theta    = THETA_HOM,
    delta_I  = 0.18,
    delta_H  = 0.12,
    gamma_I  = 0.09,
    gamma_H  = 0.10,
    N_pop    = 10_877_533.0,
    seed_total = 8.0,
)

derived = pd.read_csv(os.path.join(LANCET, "data", "update_2026_07_25",
                                    "daily_new_cases_deaths_derived.csv"),
                       parse_dates=["date"])
y_obs = derived["new_confirmed_cases_est"].round().astype(int).values
T = len(y_obs)


def rhs(t, y, beta_I):
    S, E, I, H, R = y
    N = S + E + I + H + R
    if N <= 0:
        return [0.0] * 5
    lam = (beta_I * I + FIXED["beta_H"] * H) / N
    dS = -lam * S
    dE = lam * S - FIXED["kappa"] * E
    dI = FIXED["kappa"] * E - (FIXED["theta"] + FIXED["delta_I"] + FIXED["gamma_I"]) * I
    dH = FIXED["theta"] * I - (FIXED["delta_H"] + FIXED["gamma_H"]) * H
    dR = FIXED["gamma_I"] * I + FIXED["gamma_H"] * H
    return [dS, dE, dI, dH, dR]


def run_mu(beta_I):
    N, seed_total = FIXED["N_pop"], FIXED["seed_total"]
    y0 = [N - seed_total, 0.0, seed_total, 0.0, 0.0]
    sol = solve_ivp(rhs, (0.0, float(T)), y0, args=(beta_I,),
                     t_eval=np.arange(1, T + 1, dtype=float),
                     method="RK45", rtol=1e-7, atol=1e-9, max_step=0.5)
    return np.maximum(FIXED["kappa"] * sol.y[1], 1e-9)


def neg_log_lik(beta_I):
    mu = run_mu(beta_I)
    p_nb = phi_obs_med / (mu + phi_obs_med)
    ll = nbinom.logpmf(y_obs, phi_obs_med, p_nb).sum()
    return -ll


res = minimize_scalar(neg_log_lik, bounds=(0.05, 3.0), method="bounded",
                       options={"xatol": 1e-5})
beta_I_hom = res.x

k_hom = FIXED["theta"] + FIXED["delta_I"] + FIXED["gamma_I"]
gh_sum = FIXED["delta_H"] + FIXED["gamma_H"]
R0_hom = (beta_I_hom + FIXED["beta_H"] * FIXED["theta"] / gh_sum) / k_hom

print(f"phi0 (posterior median)      = {phi0_med:.4f}")
print(f"theta_N (posterior median)   = {theta_N_med:.4f}")
print(f"theta_hom (weighted average) = {THETA_HOM:.4f}")
print(f"phi_obs (posterior median)   = {phi_obs_med:.4f}")
print(f"\nFitted beta_I_hom = {beta_I_hom:.4f}")
print(f"R0_hom = {R0_hom:.3f}")
