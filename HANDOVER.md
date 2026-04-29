# HANDOVER.md — Estimation-of-Games-under-No-Regret_Code

## Purpose

MATLAB + Stata + Python codebase for "Estimation of Games under No Regret: Structural Econometrics for AI" (Lomys & Magnolfi, JPE revision). Refactored from the RA's original `Replication_2025` package into a clean, reproducible layout. Simulates no-regret learning dynamics (regret matching, proxy-regret matching), solves BCCE identification problems via SOCP, and generates figures/tables for the paper.

## Architecture

```
stata/              Stata orchestration, cleaning, figures, appendix tables
python/             Scraping scripts (Swappa, Decluttr, Gazelle)
matlab/src/         MATLAB entry scripts + df.* namespace (refactored core)
matlab/data/        MATLAB input data (tracked in git)
matlab/test/        Fixture validation
data/               Raw scraped data + intermediates (gitignored, local only)
docs/               Provenance, audit notes, DAS, scraping protocols
output/             Generated figures/tables (gitignored)
```

### MATLAB `+df/` Namespace

| Package | Contents |
|---------|----------|
| `+df/+setup/` | Game configuration (simulation DGP, Swappa application) |
| `+df/+sim/` | Learning algorithms (regret matching, PRM) |
| `+df/+solvers/` | SOCP/LP solvers (CVX+SeDuMi batch mode, native linprog for polytope) |
| `+df/+stages/` | Stage orchestrators (I–IV pipeline) |
| `+df/+report/` | Figures, tables, SVM classification |
| `+df/+io/`, `+df/+util/` | Paths, grid builders, LaTeX export |

### Key Entry Points

| Script | Stage | Purpose |
|--------|-------|---------|
| `I_MAIN_Simul_2acts.m` | I | Polytope + learning trajectory |
| `II_MAIN_simul.m` | II | Parametric identification |
| `II_RUN_nonparam_revision.m` | II | Nonparametric identification (R1.1.b) |
| `II_RUN_demand_identification.m` | II | Joint η + (μ,σ) identification (R1.1.d) |
| `II_RUN_prm_comparison.m` | II | PRM vs RM comparison (R1.1.c, not yet run) |
| `III_MAIN_Estim_Application_PrefSpec.m` | III | Swappa application |
| `IV_MAIN_Emp_Distrib_Regrets.m` | IV | Bootstrap regrets |

## Data Pipeline

1. **Raw scrape** → `data/Swappa_Daily_Data/`, `data/Decluttr_Daily_Data/`, `data/gazelle_data/`
2. **Stata cleaning** → `stata/build_intermediates.do` (or fallback to shipped `.dta` files)
3. **Stata merge + export** → `stata/generate_matlab_data.do` → `matlab/data/`
4. **MATLAB estimation** → Stages I–IV → `matlab/output/`
5. **Stata figures/tables** → `stata/Figure_master.do`, `stata/AppendixTables.do` → `output/stata/`

## Data Provenance

Full provenance documentation:
- `docs/DATA_AVAILABILITY_STATEMENT.md` — JPE-style DAS (all sources, methods, rights)
- `docs/Scraping_Protocol_Summary_v2.md` — field-level technical reconstruction
- `data/gazelle_data/notes_gazelle_data.txt` — Gazelle collection timeline

### Data Sources Summary

| Source | Type | Period | Files |
|--------|------|--------|-------|
| Swappa | Listing scrape (daily) | Jul 11 – Dec 23, 2023 | 166 xlsx + 3 supplemental |
| Decluttr | API scrape (daily) | Jul 11 – Dec 23, 2023 | 166 xlsx |
| Gazelle sell (old) | Web scrape | Jul 1 – Jul 21, 2024 | 20 csv |
| Gazelle sell (new) | Web scrape | Jul 27 – Sep 8, 2024 | 44 csv |
| Gazelle buy | Web scrape | Jul 29 – Sep 8, 2024 | 39 csv |
| Apple MSRP | Manual lookup | Static | 1 xlsx |

## Solver Stack

- Primary: CVX 2.2 + SeDuMi (batch mode via `solve_grid_cvx.m`)
- Polytope: native `linprog` (AMPL dependency eliminated)
- Environment: MATLAB R2024a+, Windows 11

## Linked Repos

- **Paper repo**: `Estimation-of-Games-under-No-Regret/` (Overleaf-synced LaTeX)
- **RA's original package**: `Dropbox/Replication Package/` (unrefactored; used for provenance audit)

## Known Limitations

- Grid construction for large s (≥20): fewer global Dirichlet draws (scaled by 10/s)
- CVX+SeDuMi hangs with >~2,600 candidates per batch; keep fast-pass at ~256
- SVM classification requires ≥3 identified points
- `learn_prm.m` / `regret_matching_prm.m`: early implementations, not yet validated
- Figure 12 requires Stata `heatplot` package (not yet installed)
- `df_repo_paths` is a **function** — call as `paths = df_repo_paths()`, not as a script

## Replication Package Status (for eventual JPE submission)

| Component | Status |
|-----------|--------|
| Raw data (all sources) | Complete in `data/` |
| Scraping code | Complete in `python/` |
| Stata pipeline | Complete in `stata/`; verified through Figure 11 + appendix tables |
| MATLAB code | Refactored; fast-pass results generated; production runs pending |
| DAS | Draft complete (`docs/DATA_AVAILABILITY_STATEMENT.md`) |
| Full JPE README | Not yet written (DAS + run manifest + software versions needed) |
| `requirements.txt` | Not yet written |
