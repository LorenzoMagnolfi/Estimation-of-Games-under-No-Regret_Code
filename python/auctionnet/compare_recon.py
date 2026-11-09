"""Compare the Regime O masks of two reconciliation batches cell by cell (recon_v1 bool masks against recon_v2 int8 masks with
1 feasible, 0 certified infeasible, 2 unresolved). Writes a CSV with, per run x rule x N, the feasible counts in both batches, the
unresolved count, the cells lost (feasible in v1, certified infeasible in v2) and gained (the reverse), and prints one line each.
Usage: python compare_recon.py <v1_dir> <v2_dir> <out_csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

v1d, v2d, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f2 in sorted(glob.glob(os.path.join(v2d, "*_O_*.npy"))):
    b = os.path.basename(f2)[:-4]
    run, rest = b.split("_O_", 1); rule, N = rest.rsplit("_", 1)
    m2 = np.load(f2).astype(np.int8)
    f1 = os.path.join(v1d, os.path.basename(f2))
    rec = dict(run=run, rule=rule, N=int(N), n_grid=int(m2.size), v2_feasible=int((m2 == 1).sum()), v2_unresolved=int((m2 == 2).sum()), v2_outer=int((m2 != 0).sum()))
    if os.path.exists(f1):
        m1 = np.load(f1).astype(bool)
        rec.update(v1_feasible=int(m1.sum()), lost=int((m1 & (m2 == 0)).sum()), gained=int((~m1 & (m2 == 1)).sum()), unresolved_were_feasible=int((m1 & (m2 == 2)).sum()))
    else:
        rec.update(v1_feasible=np.nan, lost=np.nan, gained=np.nan, unresolved_were_feasible=np.nan)
    rows.append(rec)
T = pd.DataFrame(rows).sort_values(["run", "rule", "N"])
T.to_csv(out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(T.to_string(index=False))
chg = T[(T.lost.fillna(0) > 0) | (T.gained.fillna(0) > 0) | (T.v2_unresolved > 0)]
print(f"COMPARE_DONE: {len(T)} masks compared; {len(chg)} with a change or an unresolved cell")
