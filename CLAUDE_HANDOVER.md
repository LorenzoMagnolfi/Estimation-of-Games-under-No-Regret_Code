# CLAUDE_HANDOVER.md — Estimation-of-Games-under-No-Regret_Code

<!--
  This file is for Claude session continuity. Update at the END of each session.
  Read this FIRST when resuming work on a project.
-->

## Last Session
- **Date**: 2026-03-29
- **Summary**: Replication package audit. Cross-checked RA's Dropbox "Replication Package" folder against code repo, copied missing data files, drafted Data Availability Statement, and drafted follow-up email to RA (Bo Feng) requesting remaining loose ends.

## Current State

### What Was Done
- Audited RA's `Dropbox/Replication Package/` folder against this code repo
- Copied missing files into repo:
  - 20 old Gazelle sell CSVs → `data/gazelle_data/240701_240721/`
  - `DEVICES.xlsx` → `data/Raw_Data_Clean/`
  - `Swappa_0711_1223_raw.xlsx` → `data/Raw_Data_Clean/`
  - `Scraping_Protocol_Summary.md` and `_v2.md` → `docs/`
  - `notes_gazelle_data.txt` → `data/gazelle_data/`
- Confirmed AMPL/NEOS files NOT needed (refactored code uses native `linprog` via `solve_polytope_lp.m`)
- Drafted `docs/DATA_AVAILABILITY_STATEMENT.md` covering all 5 data sources (Swappa, Decluttr, Gazelle, Apple MSRP, crosswalks), collection methods, access, rights, and "cannot re-collect" section
- Drafted email to Bo Feng requesting: (1) `device_gazelle.xlsx` + `gazelle_device_links.csv`, (2) old Gazelle sell script, (3) Python/browser versions, (4) Swappa auth clarification

### Files Touched
- `docs/DATA_AVAILABILITY_STATEMENT.md` — NEW (DAS draft for JPE)
- `docs/Scraping_Protocol_Summary.md` — copied from RA package
- `docs/Scraping_Protocol_Summary_v2.md` — copied from RA package
- `data/gazelle_data/240701_240721/*.csv` — 20 files copied from RA package
- `data/gazelle_data/notes_gazelle_data.txt` — copied from RA package
- `data/Raw_Data_Clean/DEVICES.xlsx` — copied from RA package
- `data/Raw_Data_Clean/Swappa_0711_1223_raw.xlsx` — copied from RA package

### Git Status
- Branch: `main`, 3 commits ahead of `origin/main` (from prior sessions)
- 1 modified file: `matlab/src/+df/+stages/run_stage_ii_nonparam.m`
- 16 untracked files (handover docs, new MATLAB scripts from prior sessions, DAS, scraping docs)
- Data files are gitignored (under `/data/`)
- **Needs commit + push** (accumulated across multiple sessions)

## Pending / Next Steps
- [x] **Bo Feng follow-up resolved** (via Jason): all 4 items answered
  - `device_gazelle.xlsx` + `gazelle_device_links.csv` → copied to `data/gazelle_data/`
  - Old Gazelle sell script (`scrape_gazelle (V1_bf_240721).py`) → copied to `python/`
  - Python 3.11.5, pandas 2.2.1, requests 2.32.2, BS4 4.12.2, openpyxl 3.1.2, selenium 4.9.0, Chrome/ChromeDriver 129.0.6668.70
  - Swappa login NOT required for listing browsing (cookie was convenience)
  - RA's README PDF (`EstimationUnderNoRegret_Application.pdf`) → copied to `docs/`
  - DAS updated with all version info and corrections
- [ ] **Commit and push** — accumulated untracked files from multiple sessions
- [ ] **Production MATLAB runs** (10k grid nonparametric, finer demand ID) — blocked on Theorem 3 resolution
- [ ] **PRM comparison** (`II_RUN_prm_comparison.m`) — blocked on Theorem 3
- [ ] **Figure 12**: needs `heatplot` Stata package
- [ ] **Final README**: DAS is drafted; full JPE-style README still needs software versions, run times, manifest
- [ ] **`requirements.txt`** for Python scraping environment (awaiting Bo's version info)

## Open Questions
- **Theorem 3 feedback assumption**: full-info or bandit rate? NL must resolve. Blocks PRM exercise and R1.1.c response.
- ~~Swappa authentication~~: confirmed no login needed (cookie was convenience)
- ~~Gazelle scraper inputs~~: recovered and copied
- **Data redistribution rights**: DAS states no ToS restrictions; confirm with NL whether any additional statement is needed for JPE.

## Tricky Bits
- `data/` is gitignored — raw data lives locally only. The copied files are present on disk but not tracked by git. This is intentional (large files).
- `matlab/data/` IS tracked (whitelisted in `.gitignore`) — this is the MATLAB input layer shipped with the package.
- `df_repo_paths` is a **function** returning a struct, not a script. Call as `paths = df_repo_paths()`.
- CVX+SeDuMi hangs with >~2,600 candidates per batch. Keep fast-pass grids at ~256.
- The RA package folder (`Dropbox/Replication Package/`) is separate from this repo — it's the RA's original unrefactored package. This repo is the refactored version.
