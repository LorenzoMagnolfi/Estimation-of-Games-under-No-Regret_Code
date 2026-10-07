"""Summarize the Design A robustness array (run_robust.py) against the definitive run.

Reads results/designA/inf_role<r>_hedge_1e8_regime{P,O}.csv (the definitive run, labelled 'base') and
results/designA/robust/role<r>/<variant>/inf_hedge_1e8_regime{P,O}.csv, plus robust/role<r>/stz_stz.csv.
Writes three CSVs next to the robust folder: summary_regimeP.csv, summary_regimeO.csv, summary_stz.csv.
Usage: python summarize_robust.py <designA_results_dir>
"""
import glob
import json
import os
import sys

import pandas as pd

da = sys.argv[1]
rob = os.path.join(da, "robust")
VARIANTS = ["base", "seed2", "seed3", "lam075", "lam150", "lamcat", "nm8", "nm16", "nv5", "nv9", "wide"]
rowsP, rowsO, rowsS = [], [], []
for rdir in sorted(glob.glob(os.path.join(rob, "role*"))):
    role = int(os.path.basename(rdir)[4:])
    man = json.load(open(os.path.join(rdir, "MANIFEST.json")))
    for v in VARIANTS:
        pref = os.path.join(da, f"inf_role{role}_hedge_1e8") if v == "base" else os.path.join(rdir, v, "inf_hedge_1e8")
        if not os.path.exists(pref + "_regimeP.csv"):
            continue
        lam = man["lambda0"] if v in ("base", "seed2", "seed3") else man["variants"][v]["lambda0"]
        P = pd.read_csv(pref + "_regimeP.csv")
        P = P[P.rule.isin(["markov", "exact", "empirical"])]
        for _, r in P.iterrows():
            rowsP.append(dict(role=role, variant=v, lambda0=lam, N=int(r.N), rule=r.rule, lo=r.lam_lo, hi=r.lam_hi,
                              width=r.width, empty=r.lam_lo > r.lam_hi, contains=bool(r.contains_truth)))
        O = pd.read_csv(pref + "_regimeO.csv")
        for _, r in O.iterrows():
            rowsO.append(dict(role=role, variant=v, N=int(r.N), rule=r.rule, share=r.share, mu_lo=r.mu_lo, mu_hi=r.mu_hi,
                              sig_lo=r.sig_lo, sig_hi=r.sig_hi, truth_feasible=bool(r.truth_feasible)))
    sp = os.path.join(rdir, "stz_stz.csv")
    if os.path.exists(sp):
        S = pd.read_csv(sp)
        S.insert(0, "role", role)
        rowsS.append(S)

pd.DataFrame(rowsP).to_csv(os.path.join(rob, "summary_regimeP.csv"), index=False)
pd.DataFrame(rowsO).to_csv(os.path.join(rob, "summary_regimeO.csv"), index=False)
pd.concat(rowsS).to_csv(os.path.join(rob, "summary_stz.csv"), index=False)
print("wrote", len(rowsP), len(rowsO), sum(len(s) for s in rowsS))
