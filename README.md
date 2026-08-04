# Ebola SEIHRF-OD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20453103.svg)](https://doi.org/10.5281/zenodo.20453103)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the code, data, and figures for:

> Selain K. Kasereka, Kyandoghere Kyamakya, Jean-Jacques T. Muyembe,
> "Political distrust, armed conflict, and body reclamation drive Ebola
> transmission dynamics in eastern Democratic Republic of Congo: a
> coupled epidemic and opinion dynamics model (SEIHRF-OD)"
> — under review, 2026
>
> University of Klagenfurt · INRB/INOHA Kinshasa · INSP Kinshasa

---

## What this is

When families in eastern DRC started reclaiming the bodies of relatives who
died of Ebola from treatment centres — sometimes violently — and carrying
them home for traditional burial, existing epidemic models had no way to
represent that. Those models assume everyone follows public-health guidance.
Here they clearly did not.

SEIHRF-OD splits the population into two groups: people who accept the EVD
diagnosis (Believers, *B*) and people who deny or doubt it (Sceptics, *N*).
These are analytical categories representing levels of institutional trust;
they carry no moral judgment about affected communities, whose distrust is a
structural product of conflict, prior exploitation, and systemic exclusion.

Sceptics avoid hospitals and their deceased relatives' bodies enter a separate
compartment **F_R** (reclaimed bodies), with transmission coefficient
β_FR that substantially exceeds community and hospital rates. A third layer
tracks how the sceptic proportion φ(t) changes over time, driven by social
contagion, visible community deaths, health communication, and armed-conflict
intensity C(t).

Calibration uses full Bayesian MCMC (CmdStan/CmdStanPy) on daily INSP
situation-report data (2,973 confirmed cases, SitReps 001–070). The basic
reproduction number R₀ is derived analytically via the Next Generation Matrix
and evaluated across all posterior draws.

> **Note on reproducibility across data freezes:** this repository has now been
> recalibrated twice as the outbreak progressed (127 → 1,561 → 2,973 cases).
> Several headline quantities — the homogeneous-model R₀ gap, the dominant
> Sobol-sensitivity parameter, and the relative ranking of the S2/S3
> counterfactual scenarios — have **not** been stable across these freezes; see
> "Calibration results" and "Sensitivity analysis" below for the honest,
> freeze-by-freeze picture rather than a single cherry-picked number.

---

## Repo layout

```
Ebola_SEIHRF-OD/
├── seihrf_od.stan               # Stan ODE model — 12-compartment SEIHRF-OD
├── seihrf_od_profile.stan       # Profile-likelihood wrapper (fixes one param at a time)
├── run_mcmc.R                   # CmdStanR MCMC calibration script
├── run_mcmc_py.py               # CmdStanPy MCMC calibration script (Python)
├── figures_replot.py            # Figs 1, 3, S1, S4 — correct Table-1 parameters
├── gen_fig2.py                  # Fig 2 — R₀ vs φ₀ and p_c vs φ₀
├── gen_fig4.py                  # Fig 4 — Sobol tornado + prior/posterior panel
├── gen_ppc.py                   # Posterior predictive check → figS_ppc
├── holdout_validation.py        # 8/3 hold-out validation → figS_holdout
├── bfr_robustness.py            # Supp Table S1 — β_FR prior-support sweep
├── compute_scenarios.py         # Counterfactual S1/S2/S3/S1+S3 with MCMC CrI
├── profile_likelihood.py        # Profile-likelihood identifiability → figS2
├── sensitivity_Ct.py            # C(t) sensitivity — superseded by figures_replot.py
├── acled_pipeline.py            # C(t) reconstruction from documented security events
├── homogeneous_r0.py            # Independent homogeneous-model R0 comparator (MLE fit)
├── gen_peak_timing.py           # Long-horizon peak-timing + final-size projection
├── stan_data.json               # Prepared Stan input (2,973 cases, SitReps 001–070)
├── posterior_draws.csv          # 8 000 MCMC draws (4 chains × 2 000 samples)
├── profile_likelihood_results.csv   # From the original 127-case freeze; not rerun
├── sensitivity_Ct_results.csv       # From the original 127-case freeze; not rerun
├── requirements.txt
├── data/
│   ├── insp_sitrep__new_confirmed_cases__daily.csv      # zone-level; not maintained upstream past ~1 Jun
│   ├── insp_sitrep__cumulative_confirmed_cases__daily.csv   # national, through 23 Jul 2026
│   ├── insp_sitrep__cumulative_confirmed_deaths__daily.csv  # national, through 23 Jul 2026
│   ├── insp_sitrep__cumulative_contacts_isolated__daily.csv
│   ├── insp_sitrep__new_contacts_listed__daily.csv
│   └── update_2026_07_25/
│       └── daily_new_cases_deaths_derived.csv   # Daily new cases/deaths recovered from the
│                                                  # cumulative series (gaps spread evenly across
│                                                  # missing calendar days); this is the actual
│                                                  # model input, T=71 days
├── imgs/
│   ├── fig1_epidemic_opinion.{pdf,png}   # Epidemic curve + opinion dynamics
│   ├── fig2_R0_analysis.{pdf,png}        # R₀ vs φ₀ and p_c panels
│   ├── fig3_scenarios.{pdf,png}          # Counterfactual scenarios
│   ├── fig4_sensitivity.{pdf,png}        # Sobol tornado + prior/posterior
│   ├── fig5_spatial.{pdf,png}            # Spatial covariates (metapopulation; structural, not refit)
│   ├── figS1_Rt.{pdf,png}               # Time-varying Rt
│   ├── figS2_profile_likelihood.pdf      # Profile-likelihood identifiability (127-case freeze)
│   ├── figS4_sensitivity_Ct.pdf          # C(t) sensitivity (12 anchors)
│   ├── figS_ppc.{pdf,png}               # Posterior predictive check (all 71 days)
│   ├── figS_holdout.{pdf,png}           # Hold-out validation (58/13 split)
│   └── fig_peak_timing.{pdf,png}        # Long-horizon peak-timing projection (illustrative)
└── src/
    ├── seihrf_od_model.py       # Python ODE implementation (exploratory; stale build reference)
    └── seird_od_model.py
```

---

## Getting started

### Requirements

```bash
pip install -r requirements.txt
```

Python dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `cmdstanpy`,
`SALib` (Sobol sensitivity), `sympy`.

### MCMC calibration (Python — recommended)

```bash
python run_mcmc_py.py
# → posterior_draws.csv  (8 000 draws)
# → stan_data.json
```

4 chains × 2 000 warm-up + 2 000 sampling, seed=42, CmdStanPy.
Convergence: max R̂ = 1.003, min ESS = 2 923 (excellent).

### MCMC calibration (R)

```r
install.packages(c("cmdstanr", "posterior", "bayesplot", "loo",
                   "dplyr", "readr", "lubridate"))
cmdstanr::install_cmdstan()
Rscript run_mcmc.R
```

### Generate figures

```bash
python figures_replot.py   # figs 1, 3, S1, S4
python gen_fig2.py         # fig 2
python gen_fig4.py         # fig 4
```

### Counterfactual scenarios

```bash
python compute_scenarios.py   # prints S1/S2/S3/S1+S3 deaths-averted table
```

### β_FR robustness (Supp Table S1)

```bash
python bfr_robustness.py   # sweeps β_FR across prior 5th–95th pct range
```

### Identifiability and sensitivity

```bash
python profile_likelihood.py   # → imgs/figS2_profile_likelihood.pdf
python sensitivity_Ct.py       # → imgs/figS4_sensitivity_Ct.pdf
```

### Validation

```bash
# Posterior predictive check (uses existing posterior_draws.csv — no new MCMC)
python gen_ppc.py              # → imgs/figS_ppc.pdf/.png

# 58/13 hold-out validation (runs new MCMC on a 58-day calibration set, ~20 min)
# Result cached in posterior_calib.csv after first run — delete that file first
# if you change T_CALIB, or it will silently reuse a stale cached posterior.
python holdout_validation.py   # → imgs/figS_holdout.pdf/.png
```

### Long-horizon peak-timing / final-size projection

```bash
python gen_peak_timing.py   # → imgs/fig_peak_timing.pdf/.png
```

Extends the calibrated model to a 730-day horizon holding C(t) at its last
documented anchor level indefinitely, to estimate when daily incidence peaks
under current dynamics absent further intervention. This is a strong
extrapolation (see manuscript Limitations) — read as illustrative, not a
forecast.

---

## The model

### State variables (12 compartments)

|  | Believers (B) | Sceptics (N) |
|---|---|---|
| Susceptible | S_B | S_N |
| Exposed | E_B | E_N |
| Infectious | I_B | I_N |
| Hospitalised | H_B | H_N |
| Recovered | R_B | R_N |

Plus two post-mortem compartments: **F_R** (reclaimed body — high
transmission, β_FR) and **F_S** (safe burial — negligible transmission,
β_FS ≈ 0).

### Forces of infection

```
λ_B = [β_I(I_B+I_N) + β_H(H_B+H_N) + β_FS·F_S] / N
λ_N = [β_I(I_B+I_N) + β_H(H_B+H_N) + β_FR·F_R] / N
```

The only structural difference between the two groups is the last term:
Sceptics interact with reclaimed bodies; Believers do not.

### Opinion-dynamics layer

B↔N conversion rates:

```
μ_BN(t) = α·φ(t) + δ_C·C(t)       — social contagion + conflict amplification
μ_NB(t) = γ_comm + β_D·D_vis(t)   — health communication + visible deaths
```

The sceptic proportion φ(t) satisfies:

```
dφ/dt =  α·φ(1−φ)              — scepticism spreads person-to-person
        − β_D·D_vis(t)·φ       — visible deaths erode denial
        − γ_comm·φ             — health communication shifts opinion
        + δ_C·C(t)·(1−φ)      — conflict recruits Believers into Scepticism
```

C(t) is a piecewise step function anchored to twelve documented security events
in Ituri, North Kivu, Haut-Uele, and Tshopo (see `acled_pipeline.py` and the
Data section below).

### Analytical reproduction number

Using the Next Generation Matrix (van den Driessche & Watmough, 2002),
R₀ is the dominant eigenvalue of a 2×2 effective NGM matrix **M**:

```
R₀ = [tr(M) + √(tr(M)² − 4·det(M))] / 2

tr(M)  = (1−φ₀)·R₀_B + φ₀·R₀_N
det(M) = φ₀(1−φ₀)·R₀_B · [β_FR/ω_FR · burial term] ≥ 0
```

The weighted average tr(M) equals R₀ only when β_FR = 0 (no body
reclamation). When β_FR > 0 the exact R₀ falls below tr(M) by
det(M)/tr(M), which is under 6% across the posterior parameter range.

Group-specific reproduction numbers at posterior medians:

```
R₀_B ≈ 1.829   (β_I=0.929, θ_B=0.28 fixed)
R₀_N ≈ 3.623   (β_FR=1.659, θ_N=0.038)
```

Posterior-median R₀ ≈ **2.55** (MCMC median across all 8 000 draws).
Plug-in estimate from parameter medians: 2.552 — negligible Jensen gap
(<0.1%) with 2,973 confirmed cases.

A homogeneous (non-stratified) model independently fitted to the same case
series by NegBin maximum likelihood (`homogeneous_r0.py`) yields R₀ ≈ 2.32,
about 10% below the stratified estimate. **This gap is not a stable
quantity** — it was 21% on the original 127-case/13-day series, closed to
~0% on the 1,561-case/52-day series, and reopened to ~10% here. The
takeaway is not "homogeneous models underestimate R₀ by X%" for a fixed X;
it's that aggregate R₀ from case-count growth alone is largely insensitive
to stratification, so the stratified model's real value is decomposing R₀
into R₀_B vs R₀_N — information no homogeneous model can recover, regardless
of how close its aggregate estimate happens to land.

---

## Calibration results

Data: 2,973 confirmed cases, INRB-UMIE/Ebola_DRC_2026 build `fe2c943`
(SitReps 001–070, data freeze 23 July 2026).

| Parameter | Posterior median | 95% CrI | Status |
|---|---|---|---|
| β_I (community transmission, day⁻¹) | 0.929 | 0.86–1.00 | Data-informed |
| β_FR (reclaimed-body transmission, day⁻¹) | 1.659 | 1.16–2.14 | Prior-dominated |
| φ₀ (initial scepticism) | 0.459 | 0.36–0.56 | Data-informed |
| γ_comm (communication rate, day⁻¹) | 0.073 | 0.039–0.098 | Data-informed |
| θ_N (sceptic hospitalisation rate, day⁻¹) | prior | — | Prior-dominated |
| α, δ_C | prior | — | Prior-dominated |
| **R₀** | **2.55** | **2.36–2.76** | Derived |

Convergence: max R̂ = 1.003, min ESS = 2 923 (4 chains × 2 000 draws each).

β_FR, α, and δ_C yield flat profile-likelihood curves (non-identifiable
at the 95% level; this was established on the original 127-case freeze and
not rerun since — see `profile_likelihood.py`). β_I, φ₀, and (as of this
freeze) γ_comm are the parameters that update substantially from their
priors. All headline conclusions hold across the full
prior support of β_FR (see `bfr_robustness.py` and Supp Table S1).

### Results across all three data freezes (127 → 1,561 → 2,973 cases)

Reported here so nobody mistakes a single freeze's number for a stable
property of the outbreak:

| Quantity | 13-day freeze | 52-day freeze | 71-day freeze (current) |
|---|---|---|---|
| Confirmed cases | 127 | 1,561 | 2,973 |
| R₀ (posterior median) | 2.17 | 2.67 | 2.55 |
| Homogeneous-model R₀ gap | +21% | ~0% | +10% |
| Dominant Sobol parameter | β_I (S_i=0.60) | γ_comm (S_i=0.38) | β_I (S_i=0.34) |
| S3 (body reclamation) rank among S1/S2/S3 | largest | 2nd (barely) | smallest |
| S1 (communication) rank among S1/S2/S3 | 2nd | largest | largest |

The homogeneous-gap and Sobol-dominance columns track how far the day-90
scenario-projection horizon extrapolates beyond whatever the calibration
window happens to end at — not a fixed property of the transmission
dynamics. S1 (enhanced communication) is the one ranking that has held at
every freeze but the first.

---

## Counterfactual scenarios

From the calibrated posterior (2,973 cases; cumulative deaths at day 90):

| | Intervention | Deaths averted (median; 95% CrI) |
|---|---|---|
| S1 | Double communication rate from day 14 (γ_comm × 2) | **32%** (21–49%) |
| S2 | Halve conflict intensity throughout (C(t) × 0.5) | **23%** (16–39%) |
| S3 | Eliminate body reclamation (β_FR = 0) | **18%** (10–36%) |
| S1+S3 | Combined | **41%** (27–61%) |

**Primary inference:** S1 (enhanced communication) has been the largest
single-intervention effect at every data freeze but the first; the S2/S3
ranking has flipped between freezes (see the cross-freeze table above), so
we report it as unstable rather than asserting a fixed ordering. Absolute
death projections depend on extrapolation beyond the calibration window —
now a much smaller extrapolation (day 90 is only 21 days past the day-69
end of calibration) than at the original 13-day freeze, hence the smaller
percentages here despite the larger population now known to be at risk.
Sweeping β_FR across its prior 5th–95th percentile [1.19, 2.01] day⁻¹
keeps S3 deaths averted in the range 12–20% and R₀_N > R₀_B at all
values tested.

---

## Sensitivity analysis

Global first-order Sobol sensitivity indices for cumulative deaths at day 90:

| Parameter | S_i | Driver role |
|---|---|---|
| β_I (community transmission) | **0.34** | Primary |
| γ_comm (communication rate) | 0.27 | Primary (narrowly behind β_I) |
| δ_C (conflict amplification) | 0.10 | Secondary |
| α (social contagion) | 0.09 | Secondary |
| β_FR, φ₀, θ_N | ≤ 0.02 each | Minor |

β_I and γ_comm have swapped the #1/#2 ranking at every successive data
freeze (β_I dominant at 13 days, γ_comm dominant at 52 days, β_I dominant
again at 71 days) — see the cross-freeze table above. β_FR remains
prior-dominated regardless; its low Sobol index reflects the current
identifiability limit, not a low physical importance.

C(t) sensitivity: perturbing each of the twelve conflict-intensity anchors
independently by ±30% changes cumulative 90-day deaths by at most **2.7%**
(Anchor 8, partial de-escalation, days 31–44 — the longest-duration anchor
prior to the renewed-disruption period). All anchors are well inside the
±10% materiality threshold.

---

## Validation

### Posterior predictive check (full dataset)

Using all 8 000 posterior draws on the 71-day observed series:

| | 50% CrI | 95% CrI |
|---|---|---|
| Coverage (71 days) | 45/71 (63%) | 63/71 (89%) |

### Hold-out validation — 58/13 split

Calibration on days 1–58 (14 May–10 Jul 2026, 1,875 cases) with days
59–71 (11–23 Jul, 1,099 cases) held out as the validation set. The split
places both documented conflict peaks and the mid-July strike/vandalism
anchors in calibration and tests forecast accuracy on the
Muchanga-attack / multi-zone-resistance regime.

Calibration posterior (T=58): R₀=2.63 [2.40, 2.87], max R̂=1.002,
min ESS=2 784 — consistent with full-data posterior.

| | 50% CrI | 95% CrI |
|---|---|---|
| Calibration coverage (58 days) | 30/58 (52%) | 50/58 (86%) |
| **Forecast coverage (13 held-out days)** | 2/13 (15%) | **13/13 (100%)**  |

All 13 held-out observations (11–23 Jul) fall within the 95% credible
interval of the out-of-sample forecast, though only 2/13 fall within the
tighter 50% interval — the model gets the epidemic regime right but not
the precise day-to-day fluctuation, including a large multi-day reporting
catch-up around 21–22 July that no smooth deterministic trajectory would
be expected to hit exactly.

---

## Data

Epidemiological input: INSP daily situation reports, sourced from
[INRB-UMIE/Ebola\_DRC\_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)
(repository since renamed `INRB-UMIE/BDBV2026-Data`; old URLs redirect),
build `fe2c943` (data freeze 23 July 2026; accessed 26 July 2026).

```
SitReps included: 001–070 (SitReps 003, 029, 043, 045, 059, 061, 063, 068 missing)
Confirmed cases at calibration: 2,973
Confirmed deaths at calibration: 1,309
Health zones with a confirmed case: 48 (51 under active surveillance)
Provinces affected: Ituri, North Kivu, Haut-Uele, Tshopo, South Kivu
Catchment population (N_pop, sum of WorldPop across the 48 zones): 10,877,533
Files:
  insp_sitrep__new_confirmed_cases__daily.csv        (zone-level; stale past ~1 Jun upstream)
  insp_sitrep__cumulative_confirmed_cases__daily.csv  (national, current)
  insp_sitrep__cumulative_confirmed_deaths__daily.csv (national, current)
  insp_sitrep__cumulative_contacts_isolated__daily.csv
  insp_sitrep__new_contacts_listed__daily.csv
  update_2026_07_25/daily_new_cases_deaths_derived.csv  (actual T=71 model input)
```

Conflict-intensity function C(t): piecewise step function anchored to
twelve documented security events (see `acled_pipeline.py` for the
original five; anchors 6–12 added across the two recalibrations from the
INRB-UMIE security-incident log):

| Anchor | Model day | C value | Event |
|---|---|---|---|
| 1 | 1 (from 11 May 2026) | 0.55 | US health worker exposed at Nyankunde |
| 2 | 5 (18 May 2026) | 0.65 | CDC announcement + Berlin evacuation |
| 3 | 8 (21–23 May 2026) | 1.00 | Rwampara/Mongbwalu tent burnings (peak I) |
| 4 | 11 (24 May+) | 0.60 | Persistent insecurity I; >100 000 displaced |
| 5 | 20 (2 Jun 2026) | 0.75 | Bunia/Katana safe-burial team attacks |
| 6 | 21 (3 Jun 2026) | 1.00 | Oicha/Mbau massacre, ~24 civilians (peak II) |
| 7 | 25 (7 Jun 2026) | 0.70 | Nyamurongo cemetery attack, Bunia |
| 8 | 33 (15 Jun 2026) | 0.60 | Partial de-escalation, Mongbwalu |
| 9 | 46 (28 Jun–2 Jul) | 0.70 | PoC burned (Miala) + renewed attacks |
| 10 | 55 (7–9 Jul 2026) | 0.65 | Healthcare-provider strike, Bunia/Rwampara |
| 11 | 58 (10 Jul 2026) | 0.75 | Repeat PoC vandalism + provider threats |
| 12 | 67 (19 Jul 2026) | 0.85 | Muchanga bridge attack + multi-zone resistance |

WHO situation reports:
[DON602](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON602) ·
[DON603](https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON603)

---

## Citation

```bibtex
@article{kasereka2026seihrf,
  author  = {Kasereka, Selain K. and Kyamakya, Kyandoghere
             and Muyembe, Jean-Jacques T.},
  title   = {Political distrust, armed conflict, and body reclamation
             drive {Ebola} transmission dynamics in eastern {DRC}:
             a coupled epidemic and opinion dynamics model ({SEIHRF-OD})},
  year    = {2026},
  note    = {Under review}
}
```

Data: INRB-UMIE, *Ebola\_DRC\_2026*, build `fe2c943`, GitHub, 2026.
[https://github.com/INRB-UMIE/Ebola\_DRC\_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)

Code archive: Kasereka SK, Kyamakya K, Muyembe JJT. *SEIHRF-OD v1.0.0*.
Zenodo, 2026. [https://doi.org/10.5281/zenodo.20453103](https://doi.org/10.5281/zenodo.20453103)

---

## Licence

Code: MIT. Data files are from INRB, INSP, and WHO; see
[INRB-UMIE/Ebola\_DRC\_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026)
for their respective licences.
