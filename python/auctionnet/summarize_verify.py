"""Collect the per-period verify_<period>.json files of verify_auction.py into one table.

Writes <results_dir>/verify_summary.csv (one row per period) and prints the range of every check across periods.
Usage: python summarize_verify.py <results_dir>
"""
import glob
import json
import os
import sys

import pandas as pd

res = sys.argv[1]
rows = [json.load(open(f)) for f in glob.glob(os.path.join(res, "verify_period-*.json"))]
df = pd.DataFrame(rows)
df["k"] = df.period.str.split("-").str[1].astype(int)
df = df.sort_values("k").drop(columns="k")
df.to_csv(os.path.join(res, "verify_summary.csv"), index=False)
num = df.select_dtypes("number")
print(f"{len(df)} periods: {', '.join(df.period)}")
print(pd.DataFrame({"min": num.min(), "max": num.max()}).to_string())
