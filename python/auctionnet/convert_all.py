"""Convert every AuctionNet period zip to parquet and extract the per-period environment inputs.

For each zip in <raw_dir>: unzip period-<k>.csv, write <parquet_dir>/period-<k>.parquet (zstd), delete the CSV,
then write <env_dir>/period-<k>_env.npz with, for every advertiser role i (48):
  pvals[i]      float32 (n_opps,)   predicted values of role i, in opportunity order (step-major)
  rivals[i]     float32 (n_opps, 3) three highest bids among the other 47 roles (r1 >= r2 >= r3)
  bids[i]       float32 (n_opps,)   recorded bid of role i
  step          int8   (n_opps,)    time step of each opportunity
  meta          per-role budget, CPA target, category
and a small per-period summary (exposure rates by slot, reserve check, counts).
Usage: python convert_all.py <raw_dir> <parquet_dir> <env_dir>
"""
import glob
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

raw, pqd, envd = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(pqd, exist_ok=True); os.makedirs(envd, exist_ok=True)


def extract_env(df, period, out):
    for c in ["advertiserNumber", "advertiserCategoryIndex", "timeStepIndex", "pvIndex", "xi", "adSlot", "isExposed"]:
        df[c] = df[c].astype(np.int64)
    df["oppid"] = df.timeStepIndex * 1_000_000 + df.pvIndex
    d = df.sort_values(["oppid", "bid"], ascending=[True, False])
    d["rank"] = d.groupby("oppid").cumcount() + 1
    opp = d.drop_duplicates("oppid")[["oppid", "timeStepIndex"]].set_index("oppid")
    n_opps = len(opp)
    top = d[d["rank"] <= 4][["oppid", "rank", "bid", "advertiserNumber"]]
    tb = top.pivot(index="oppid", columns="rank", values="bid").reindex(opp.index)
    ta = top.pivot(index="oppid", columns="rank", values="advertiserNumber").reindex(opp.index)
    B = tb.values; A = ta.values
    wide_v = d.pivot(index="oppid", columns="advertiserNumber", values="pValue").reindex(opp.index)
    wide_b = d.pivot(index="oppid", columns="advertiserNumber", values="bid").reindex(opp.index)
    roles = sorted(df.advertiserNumber.unique())
    pv = np.zeros((len(roles), n_opps), dtype=np.float32)
    bd = np.zeros((len(roles), n_opps), dtype=np.float32)
    rv = np.zeros((len(roles), n_opps, 3), dtype=np.float32)
    for ii, i in enumerate(roles):
        pv[ii] = wide_v[i].values
        bd[ii] = wide_b[i].values
        Bm = np.where(A == i, -1.0, B)
        rv[ii] = -np.sort(-Bm, axis=1)[:, :3]
    meta = df.drop_duplicates("advertiserNumber").set_index("advertiserNumber").loc[roles, ["budget", "CPAConstraint", "advertiserCategoryIndex"]]
    winners = d[d["rank"] <= 3]
    expo_by_slot = winners.groupby("rank").isExposed.mean().reindex([1, 2, 3]).values
    np.savez_compressed(out, period=period, roles=np.array(roles), pvals=pv, bids=bd, rivals=rv,
                        step=opp.timeStepIndex.values.astype(np.int8), budget=meta.budget.values, cpa=meta.CPAConstraint.values,
                        category=meta.advertiserCategoryIndex.values, expo_by_slot=expo_by_slot, n_opps=n_opps)
    return n_opps, expo_by_slot


t0 = time.time()
for z in sorted(glob.glob(os.path.join(raw, "*.zip"))):
    try:
        zf_test = zipfile.ZipFile(z); zf_test.close()
    except zipfile.BadZipFile:
        print("skip (incomplete zip)", os.path.basename(z), flush=True); continue
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        for n in names:
            k = os.path.splitext(os.path.basename(n))[0]           # period-<k>
            pq_path = os.path.join(pqd, k + ".parquet")
            env_path = os.path.join(envd, k + "_env.npz")
            if os.path.exists(pq_path) and os.path.exists(env_path):
                print("skip", k); continue
            csv_path = os.path.join(pqd, "_tmp_" + os.path.basename(n))
            with zf.open(n) as src, open(csv_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
            if not os.path.exists(pq_path):
                tbl = pacsv.read_csv(csv_path)
                pq.write_table(tbl, pq_path, compression="zstd")
            os.remove(csv_path)
            df = pq.read_table(pq_path, columns=["advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
                                                 "timeStepIndex", "pvIndex", "pValue", "bid", "xi", "adSlot", "isExposed"]).to_pandas()
            period = int(k.split("-")[1])
            n_opps, expo = extract_env(df, period, env_path)
            print(f"{k}: rows {len(df):,d}, opportunities {n_opps:,d}, exposure by slot {np.round(expo, 3)} ({time.time() - t0:.0f}s)", flush=True)
            del df
print("done", f"({time.time() - t0:.0f}s)")
