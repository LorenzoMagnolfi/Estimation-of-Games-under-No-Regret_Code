"""Six-role tables for the recon_v2 batch (Unity array 11390728, code d69e40f).

Reads results_unity/designA/recon_v2/role<r>/ (main runs and coverage) plus the earlier tailF_v1, paired_v1 and exp3ix_v1
summaries, writes sixrole_*.csv and the digest sixrole_summary.md next to the role folders.
Usage: python sixrole_tables.py [designA_results_dir]   (default: the local results_unity/designA mirror)
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

R = [0, 10, 19, 28, 35, 40]
D = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/magnolfi.1/AuctionNet/results_unity/designA"
V2 = f"{D}/recon_v2"
OUT_MD = f"{V2}/sixrole_summary.md"
NS = [100000, 480000, 1000000, 3000000, 10000000, 30000000, 100000000]
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
md = []


def say(s=""):
    md.append(s); print(s)


def load(run, kind):
    out = []
    for r in R:
        d = pd.read_csv(f"{V2}/role{r}/{run}_regime{kind}.csv"); d.insert(0, "role", r); out.append(d)
    return pd.concat(out, ignore_index=True)


def fmt_iv(lo, hi):
    return f"[{lo:.1f}, {hi:.1f}]"


def tf(v):
    s = str(v)
    return 1.0 if s == "True" else (0.0 if s == "False" else (2.0 if s == "unresolved" else np.nan))


# ---------------------------------------------------------------- Exercise P
PI, PP, PX = load("hedge_1e8_iid", "P"), load("hedge_1e8_paired", "P"), load("exp3ix_1e7_exact", "P")
truth = PI[PI.rule == "empirical"].groupby("role").lam_lo.first().round(1)
say("# Six-role recon_v2 digest (array 11390728, code d69e40f)\n")
say("Truth lambda_0 by role: " + ", ".join(f"{r}: {truth[r]}" for r in R) + "\n")


def ptable(P, rules, title, Ns=NS):
    say(f"## {title}\n")
    rows = []
    for N in Ns:
        for r in R:
            s = P[(P.N == N) & (P.role == r)].set_index("rule")
            row = dict(N=N, role=r)
            for ru in rules:
                if ru in s.index:
                    row[f"{ru}"] = fmt_iv(s.loc[ru, "lam_lo"], s.loc[ru, "lam_hi"]); row[f"w_{ru}"] = round(s.loc[ru, "width"], 1)
                    row[f"in_{ru}"] = bool(s.loc[ru, "contains_truth"])
            rows.append(row)
    t = pd.DataFrame(rows)
    say(t.to_string(index=False)); say("")
    # width ranges and truth counts
    for ru in rules:
        if f"w_{ru}" in t:
            g = t.groupby("N")[f"w_{ru}"].agg(["min", "max"]).round(1)
            say(f"{ru}: width range over roles by N: " + "; ".join(f"{int(N):.0e} [{a}, {b}]" for N, (a, b) in g.iterrows())
                + f"; truth inside {int(t[f'in_{ru}'].sum())} of {len(t)}")
    say("")
    return t


tI = ptable(PI, ["momentF", "momentF_matched", "jointF", "tailF"], "Exercise P, Hedge, i.i.d. environment (saved 1e8 run, exact full-pool tables)")
# ratios
for a, b in [("momentF", "jointF"), ("jointF", "tailF"), ("momentF", "tailF"), ("momentF_matched", "jointF"), ("momentF", "momentF_matched")]:
    rr = (tI[f"w_{a}"] / tI[f"w_{b}"])
    g = rr.groupby(tI.N).agg(["min", "max"]).round(2)
    say(f"width ratio {a}/{b} by N: " + "; ".join(f"{int(N):.0e} [{x}, {y}]" for N, (x, y) in g.iterrows()))
say("")

tP = ptable(PP, ["momentF", "momentF_matched", "jointF", "tailF"], "Exercise P, Hedge, paired environment (saved 1e8 run, context-conditional tables)")
m = tP.merge(tI, on=["N", "role"], suffixes=("_pair", "_iid"))
for ru in ["tailF", "jointF", "momentF"]:
    rr = m[f"w_{ru}_pair"] / m[f"w_{ru}_iid"]
    say(f"paired/iid width ratio {ru}: min {rr.min():.2f} max {rr.max():.2f} median {rr.median():.2f} over {len(rr)} cells")
say("")

tX = ptable(PX, ["tailB", "tailBc", "momentB", "momentB_matched", "jointB"], "Exercise P, EXP3-IX with the exact width (new 1e7 run; role 28 the 4 November trajectory)", Ns=NS[:5])
mx = tX.merge(tI, on=["N", "role"])
for ru in ["tailB", "tailBc"]:
    rr = mx[f"w_{ru}"] / mx["w_tailF"]
    g = rr.groupby(mx.N).agg(["min", "max", "median"]).round(2)
    say(f"bandit price {ru}/tailF by N: " + "; ".join(f"{int(N):.0e} [{x}, {y}] med {z}" for N, (x, y, z) in g.iterrows()))
say("")
price = mx[["N", "role", "w_tailB", "w_tailBc", "w_tailF"]].copy()
price["ratio_tailB"] = price.w_tailB / price.w_tailF; price["ratio_tailBc"] = price.w_tailBc / price.w_tailF
price.to_csv(f"{V2}/sixrole_bandit_price.csv", index=False)
tI.to_csv(f"{V2}/sixrole_P_hedge_iid.csv", index=False); tP.to_csv(f"{V2}/sixrole_P_hedge_paired.csv", index=False); tX.to_csv(f"{V2}/sixrole_P_exp3ix_exact.csv", index=False)

# comparisons with the earlier versions
say("## Consistency with the 2 November runs (same trajectories)\n")
tf1 = pd.read_csv(f"{D}/tailF_v1/summary_P.csv")
c = PI[PI.rule == "tailF"].merge(tf1[["role", "N", "lam_lo_tailF", "lam_hi_tailF"]], on=["role", "N"])
say(f"Hedge i.i.d. tailF endpoints, recon_v2 vs tailF_v1: max |diff| lo {np.abs(c.lam_lo - c.lam_lo_tailF).max():.2e}, hi {np.abs(c.lam_hi - c.lam_hi_tailF).max():.2e} over {len(c)} cells")
pp1 = pd.read_csv(f"{D}/paired_v1/summary_P.csv")
c = PP[PP.rule == "tailF"].merge(pp1[["role", "N", "lam_lo_tailF", "lam_hi_tailF"]], on=["role", "N"])
say(f"Hedge paired tailF endpoints, recon_v2 vs paired_v1: max |diff| lo {np.abs(c.lam_lo - c.lam_lo_tailF).max():.2e}, hi {np.abs(c.lam_hi - c.lam_hi_tailF).max():.2e} over {len(c)} cells")
x1 = pd.read_csv(f"{D}/exp3ix_v1/summary_P.csv")
c = PX[PX.rule == "tailB"].merge(x1[["role", "N", "width_tailB", "lam_lo_tailB", "lam_hi_tailB"]], on=["role", "N"])
c["ratio_new_over_old"] = c.width / c.width_tailB
say("EXP3-IX tailB width, exact width (new) over the withdrawn 2 November width, by role at 1e7: "
    + ", ".join(f"{int(r)}: {v:.2f}" for r, v in c[c.N == 10000000].set_index("role").ratio_new_over_old.items()))
say("  at 1e5: " + ", ".join(f"{int(r)}: {v:.2f}" for r, v in c[c.N == 100000].set_index("role").ratio_new_over_old.items()))
say("")

# ---------------------------------------------------------------- Exercise O
OI, OP, OX = load("hedge_1e8_iid", "O"), load("hedge_1e8_paired", "O"), load("exp3ix_1e7_exact", "O")
for O in (OI, OP, OX):
    O["truth_num"] = O.truth_feasible.map(tf); O["emp_num"] = O.emp_feasible.map(tf)


def otable(O, rules, title, truthcol, Ns=NS):
    say(f"## {title}\n")
    rows = []
    for N in Ns:
        for r in R:
            s = O[(O.N == N) & (O.role == r)].set_index("rule")
            row = dict(N=N, role=r, n_none=int(s.loc["none", "n_feasible"]))
            for ru in rules:
                if ru in s.index:
                    row[f"n_{ru}"] = int(s.loc[ru, "n_feasible"]); row[f"rem_{ru}"] = round(1 - s.loc[ru, "share"] / s.loc["none", "share"], 3) if s.loc["none", "share"] > 0 else np.nan
                    row[f"t_{ru}"] = s.loc[ru, truthcol]
            row["unres"] = int(s.n_unresolved.sum()); row["floor"] = int(s.n_floor_replaced.max())
            rows.append(row)
    t = pd.DataFrame(rows)
    say(t.to_string(index=False)); say("")
    for ru in rules:
        if f"rem_{ru}" in t:
            g = t.groupby("N")[f"rem_{ru}"].agg(["min", "max"])
            say(f"{ru}: removal range over roles by N: " + "; ".join(f"{int(N):.0e} [{a:.3f}, {b:.3f}]" for N, (a, b) in g.iterrows())
                + f"; truth feasible {int((t[f't_{ru}'] == 1.0).sum())} of {len(t)}; unresolved {int((t[f't_{ru}'] == 2.0).sum())}")
    say(f"unresolved LPs total {int(t.unres.sum())}; C1 floor replacements per run (max over rules) by role: "
        + ", ".join(f"{int(r)}: {int(v)}" for r, v in t.groupby('role').floor.max().items()))
    say("")
    return t


oI = otable(OI, ["momentF", "momentF_matched", "jointF", "tailF", "tailF_own"], "Exercise O, Hedge, i.i.d. environment (lognormal truth)", "truth_num")
# active-set decomposition: rectangle - matched (active set) vs matched - budget (aggregation)
oI["active_set_effect"] = oI.n_momentF - oI.n_momentF_matched; oI["aggregation_effect"] = oI.n_momentF_matched - oI.n_jointF
say("Candidates removed by the budget beyond the rectangle, split into active set (rectangle minus matched) and aggregation (matched minus budget), 1e7 / 3e7 / 1e8:")
for r in R:
    s = oI[oI.role == r].set_index("N")
    say(f"  role {r}: " + "; ".join(f"{int(N):.0e}: none {s.loc[N,'n_none']}, rect {s.loc[N,'n_momentF']}, matched {s.loc[N,'n_momentF_matched']}, budget {s.loc[N,'n_jointF']}, tail {s.loc[N,'n_tailF']}, own {s.loc[N,'n_tailF_own']} (active {s.loc[N,'active_set_effect']}, aggr {s.loc[N,'aggregation_effect']})" for N in [10000000, 30000000, 100000000]))
say("")
oI.to_csv(f"{V2}/sixrole_O_hedge_iid.csv", index=False)

oP = otable(OP, ["momentF", "momentF_matched", "jointF", "tailF", "tailF_own"], "Exercise O, Hedge, paired environment (empirical mass vector is the truth; lognormal grid is the candidate set)", "emp_num")
say("Specification test: first horizon at which the lognormal grid is empty (n_feasible = 0) per rule and role, and the lognormal fitted candidate's feasibility (truth_num):")
for ru in ["tailF", "tailF_own", "jointF", "momentF_matched", "momentF"]:
    parts = []
    for r in R:
        s = OP[(OP.role == r) & (OP.rule == ru)].sort_values("N")
        empty = s[s.n_feasible == 0].N
        first_empty = f"{int(empty.iloc[0]):.0e}" if len(empty) else "never"
        rej = s[s.truth_num == 0.0].N
        first_rej = f"{int(rej.iloc[0]):.0e}" if len(rej) else "never"
        emp_ok = int((s.emp_num == 1.0).sum())
        parts.append(f"{r}: empty from {first_empty}, fitted candidate rejected from {first_rej}, empirical feasible {emp_ok}/{len(s)}")
    say(f"  {ru}: " + " | ".join(parts))
say("")
say("Lognormal-grid counts under tailF by horizon (paired):")
say(OP[OP.rule == "tailF"].pivot(index="role", columns="N", values="n_feasible").to_string()); say("")
say("Bids-only (none) counts by horizon (paired):")
say(OP[OP.rule == "none"].pivot(index="role", columns="N", values="n_feasible").to_string()); say("")
oP.to_csv(f"{V2}/sixrole_O_hedge_paired.csv", index=False)

oX = otable(OX, ["momentB", "momentB_matched", "jointB", "tailB", "tailBc", "tailB_own"], "Exercise O, EXP3-IX exact width, i.i.d. environment (lognormal truth)", "truth_num", Ns=NS[:5])
oX.to_csv(f"{V2}/sixrole_O_exp3ix_exact.csv", index=False)

# ---------------------------------------------------------------- certificates
say("## C1 floor and C2 certificates by role and run (solver logs)\n")
rows = []
for r in R:
    for run in ["hedge_1e8_iid", "hedge_1e8_paired", "exp3ix_1e7_exact"]:
        m = json.load(open(f"{V2}/role{r}/{run}_solver_log.json"))
        cert = pd.DataFrame(m["supporting_line_certificates"])
        rows.append(dict(role=r, run=run, mode=m["mode"], tables=m["tables"], ixnorm=m["ixnorm"], n_replaced=m["floor_log"]["n_replaced"], n_bounds=m["floor_log"]["n_bounds"],
                         n_cert=len(cert), min_margin=cert.min_margin_rel.min() if len(cert) else np.nan, max_gap=cert.max_gap_rel.max() if len(cert) else np.nan,
                         K_full_zero=int(sum(1 for k in m["K_full"] if k == 0.0))))
cert = pd.DataFrame(rows)
say(cert.to_string(index=False)); say("")
cert.to_csv(f"{V2}/sixrole_certificates.csv", index=False)

# ---------------------------------------------------------------- coverage
say("## Coverage over 20 seeds (coarse grid, 1e7)\n")
rows = []
for r in R:
    for tag, truthcol in [("hedge_iid", "truth_feasible"), ("hedge_paired", "emp_feasible"), ("exp3ix_exact", "truth_feasible")]:
        P = pd.read_csv(f"{V2}/role{r}/cov_{tag}/coverage_{tag}_P.csv"); O = pd.read_csv(f"{V2}/role{r}/cov_{tag}/coverage_{tag}_O.csv")
        O["t"] = O[truthcol].map(tf)
        gP = P.groupby(["rule", "N"]).agg(seeds=("seed", "nunique"), inP=("contains_truth", "sum"), width_med=("width", "median")).reset_index()
        gO = O.groupby(["rule", "N"]).agg(inO=("t", lambda v: int((v == 1.0).sum())), unresO=("t", lambda v: int((v == 2.0).sum())), nfeas_med=("n_feasible", "median"), unres_lp=("n_unresolved", "sum")).reset_index()
        g = gP.merge(gO, on=["rule", "N"], how="outer"); g.insert(0, "tag", tag); g.insert(0, "role", r); rows.append(g)
cov = pd.concat(rows, ignore_index=True)
cov.to_csv(f"{V2}/sixrole_coverage.csv", index=False)
main_rules = ["momentF", "momentF_matched", "jointF", "tailF", "tailF_own", "momentB", "momentB_matched", "jointB", "tailB", "tailBc", "tailB_own"]
cm = cov[cov.rule.isin(main_rules)]
say("Truth retention (seed-checkpoints inside / total) by tag and rule, Regime P and Regime O, plus unresolved:")
agg = cm.groupby(["tag", "rule"]).agg(cells=("seeds", "size"), seeds=("seeds", "sum"), inP=("inP", "sum"), inO=("inO", "sum"), unresO=("unresO", "sum"), unres_lp=("unres_lp", "sum")).reset_index()
say(agg.to_string(index=False)); say("")
say("Median Regime P width by tag, rule and N (range over roles):")
w = cm.groupby(["tag", "rule", "N"]).width_med.agg(["min", "max"]).round(1).reset_index()
say(w[w.rule.isin(["tailF", "jointF", "momentF", "tailB", "tailBc"])].to_string(index=False)); say("")
say("Median Regime O feasible count by tag, rule and N (range over roles):")
nf = cm.groupby(["tag", "rule", "N"]).nfeas_med.agg(["min", "max"]).reset_index()
say(nf[nf.rule.isin(["tailF", "jointF", "momentF", "tailB", "tailBc"])].to_string(index=False)); say("")
# paired coverage: lognormal grid emptiness per seed under tailF
rows = []
for r in R:
    O = pd.read_csv(f"{V2}/role{r}/cov_hedge_paired/coverage_hedge_paired_O.csv")
    s = O[O.rule == "tailF"].groupby("N").agg(empty_seeds=("n_feasible", lambda v: int((v == 0).sum())), emp_ok=("emp_feasible", lambda v: int((v.astype(str) == "True").sum()))).reset_index(); s.insert(0, "role", r); rows.append(s)
pe = pd.concat(rows)
say("Paired coverage under tailF: seeds (of 20) with an empty lognormal grid, and seeds keeping the empirical vector, by role and N:")
say(pe.pivot(index="role", columns="N", values="empty_seeds").to_string()); say(pe.pivot(index="role", columns="N", values="emp_ok").to_string()); say("")

# timing
say("## Timing (MANIFEST.json log)\n")
for r in R:
    m = json.load(open(f"{V2}/role{r}/MANIFEST.json"))
    L = {l["tag"]: l["seconds"] for l in m["log"]}
    say(f"role {r}: commit {m['code_commit'][:7]} {'dirty' if m['code_dirty'] else 'clean'}; infer iid {L.get('infer hedge 1e8 iid')} s, paired {L.get('infer hedge 1e8 paired')} s, exp3ix {L.get('infer exp3ix 1e7 exact')} s; coverage iid {L.get('coverage hedge_iid')} s, paired {L.get('coverage hedge_paired')} s, exp3ix {L.get('coverage exp3ix_exact')} s")

io.open(OUT_MD, "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
print("SIXROLE_DONE")
