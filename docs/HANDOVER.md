# Handover

## Status
- The active code home is [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code).
- The project is being handled in two tracks:
  - baseline replication of the current paper
  - later revision work built on top of a frozen baseline
- The Stata pipeline has been repaired enough to rebuild the curated intermediates and MATLAB inputs from the current replication package data.
- MATLAB has not been rerun end-to-end in the refactored code repo yet.
- Scraping/provenance documentation is still incomplete and is waiting on replies from former RAs.
- The current direction for the numerical layer is to keep MATLAB, but do a deep architectural refactor rather than a language migration.

## What Changed
- Repo structure and replication notes were established in:
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\README.md](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\README.md)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\FIRST_PORT_NOTES.md](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\FIRST_PORT_NOTES.md)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\REVISION_TRACK_NOTES.md](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\REVISION_TRACK_NOTES.md)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\REPLICATION_README_AUDIT.md](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\REPLICATION_README_AUDIT.md)
- Stata drivers and pathing were cleaned in:
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\build_intermediates.do](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\build_intermediates.do)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\run_stata_outputs.do](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\run_stata_outputs.do)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\run_appendix_tables.do](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\run_appendix_tables.do)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\gazelle_data.do](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\gazelle_data.do)
- MATLAB entry scripts were patched earlier for repo-relative pathing, centered on:
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\df_repo_paths.m](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\df_repo_paths.m)
- Non-proprietary data mirrors were created by:
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\export_nonproprietary_mirrors.do](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\stata\export_nonproprietary_mirrors.do)
  - output tree: [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\open_data](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\open_data)
- Provenance recovery notes were written in:
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\PROVENANCE_RECOVERY_NOTES.md](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\docs\PROVENANCE_RECOVERY_NOTES.md)

## Key Context
- Verified earlier in this thread:
  - the current replication package is a sufficient raw starting point for baseline replication
  - literal [C:\Users\lorem\Dropbox\Data](C:\Users\lorem\Dropbox\Data) is useful as an archive/convenience layer, but it should not be the official package baseline unless a missing dependency is discovered
- Stata rebuilds succeeded earlier in this thread:
  - curated intermediates were rebuilt
  - MATLAB input files were regenerated
  - appendix tables and most Stata figures were generated
- One known Stata blocker remains:
  - Figure 12 depends on `heatplot`, which is not installed on this machine
- Current MATLAB architecture assessment:
  - the biggest hotspot is repeated CVX solves inside [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\ComputeBCCE_eps.m](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\ComputeBCCE_eps.m)
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\II_MAIN_simul.m](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\II_MAIN_simul.m) and [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\III_MAIN_Estim_Application_PrefSpec.m](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\III_MAIN_Estim_Application_PrefSpec.m) mix compute, I/O, plotting, and reporting
  - [C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\learn_mod.m](C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code\matlab\src\learn_mod.m) is another major speed target
- Bo Feng's repo should be treated as revision-track input only:
  - it contains meaningful referee-response simulation work
  - it does not provide a full architectural refactor of the MATLAB stack
- Current refactor preference:
  - preserve the current paper outputs to numerical tolerance
  - keep thin wrapper mains for replication
  - move the numerical core to a modular pipeline
  - likely use `coneprog` as the future default backend, with `CVX` preserved as a validation/fallback backend during transition

## Open Items
- MATLAB has not yet been run end-to-end from the new repo; timing improvements are estimated, not benchmarked.
- Former RAs have not yet replied with the missing scraping/provenance details.
- A proper `SCRAPING_PROTOCOL.md` has not been written yet.
- Fixture infrastructure is in place but has not yet been run on a MATLAB machine (pending baseline capture).

## MATLAB Refactor
- The full refactor plan is in [docs/MATLAB_REFACTOR_PLAN.md](MATLAB_REFACTOR_PLAN.md).
- Supersedes earlier chat-only notes. Includes:
  - deep audit of all three solver variants (including previously missed `ComputeBCCE_eps_pass.m`)
  - 5-phase implementation plan + lightweight baseline capture
  - decision to kill AMPL dependency (replace with `linprog`)
  - paper outputs as artifact oracle + targeted fixture runs for module-level validation
  - `+df/` package namespace layout
  - performance targets, risk assessment, decision log

## Next Step
1. ~~Instrument the four MAIN scripts with `save` calls~~ (DONE)
2. ~~Create fixture runner, verify script, timing template~~ (DONE — `matlab/test/`)
3. **Run `run_fixtures.m` on a MATLAB machine** to capture baseline (~15-25 min)
4. Copy `fixture_*.mat` → `matlab/test/fixtures/baseline/`
5. Begin Phase 1: eliminate globals + extract game setup (mechanical, low risk)
6. Phase 3b (kill AMPL / `linprog`) can proceed independently in parallel
