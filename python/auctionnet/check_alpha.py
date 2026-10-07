"""Run 2026-10-07 as ~/NoRegret_runs/auctionnet/checks_20261007/alpha_max.py (Slurm job 11252763); this file adds only this docstring.
Usage: python check_alpha.py  (reads ~/NoRegret_data/auctionnet/parquet, writes results/verify/alpha_max.csv)
"""
# Max within (advertiser, step) relative spread of bid/pValue over rows with bid > 0 and pValue > 0, every period.
import glob, os, numpy as np, pyarrow.parquet as pq, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/NoRegret_data/auctionnet/parquet/period-*.parquet")), key=lambda x: int(x.split("-")[-1].split(".")[0])):
    d = pq.read_table(f, columns=["advertiserNumber", "timeStepIndex", "bid", "pValue"]).to_pandas()
    d = d[(d.bid > 0) & (d.pValue > 0)]
    a = d.bid / d.pValue
    g = a.groupby([d.advertiserNumber, d.timeStepIndex])
    spread = (g.max() - g.min()) / g.mean()
    rows.append(dict(period=os.path.basename(f), groups=len(spread), max_rel_spread=float(spread.max()), n_gt_1e6=int((spread > 1e-6).sum())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.expanduser("~/NoRegret_data/auctionnet/results/verify/alpha_max.csv"), index=False)
print("ALPHA_DONE")
