# Figure scripts for the V05 simulation evidence (2 to 3 November 2026)

Each script reads the pulled result tables of the two laboratories and writes a PDF and a PNG. They were run on Lorenzo's
machine against `C:/Users/magnolfi.1/AuctionNet/results_unity/` (lab tables under `lab_v05/tables/part_ii/`, auction
summaries under `designA/{tailF_v1,exp3ix_v1,paired_v1}/`), which are the artifacts of the Unity jobs recorded in the
paper repository's note `JPE Revision/two-laboratories-results-note-2026-11-03.tex` (Appendix A). The figures used in that
note are committed to the paper repository under `JPE Revision/figures/two-laboratories-2026-11-03/`.

| Script | Input | Figure in the note |
|---|---|---|
| `fig_frontier.py` | `v05_frontier.mat` | lab frontier at 4M |
| `fig_rate.py` | `v05_rate.mat`, `v05_truth8M.mat` | lab rate landscape |
| `fig_sweeps.py` | `v05_sweeps.csv` | action-grid and cost-support sweeps |
| `fig_nonparam.py` | `v05_nonparam.csv` | nonparametric clouds |
| `fig_demand_c3.py` | `v05_demand_c3.csv` | demand ridge and C3 projection |
| `fig_sparse.py` | `v05_sparse.csv` | sparse outcomes |
| `fig_regimeP_band.py` | `tailF_v1/role28/hedge_1e8_regimeP.csv` | auction Exercise P band, role 28 |
| `fig_regimeO_removal.py` | `tailF_v1/summary_O.csv`, `exp3ix_v1/summary_O.csv` | auction Exercise O removal |
| `fig_feedback_price.py` | `tailF_v1/role28/...`, `exp3ix_v1/role28/...` | auction price of bandit feedback, role 28 |

Palette and layout follow the dataviz conventions used throughout (categorical slots `#2a78d6`, `#eb6834`, `#1baf7a`;
surface `#fcfcfb`); matplotlib defaults from `~/.matplotlib/matplotlibrc` (serif, usetex) apply.
