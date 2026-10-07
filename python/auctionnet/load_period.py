"""Unzip one AuctionNet period, inspect the CSV, and convert it to parquet.

Usage: python load_period.py <zip_path> <out_dir>
Writes <out_dir>/period-<k>.parquet and prints schema, row counts, and basic checks.
Raw data live outside Dropbox by design.
"""
import sys, os, zipfile, glob, time
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

zip_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    print("zip members:", names)
    for n in names:
        info = z.getinfo(n)
        print(f"  {n}: {info.file_size/1e9:.2f} GB uncompressed")
    extract_dir = os.path.join(out_dir, "_csv")
    os.makedirs(extract_dir, exist_ok=True)
    z.extractall(extract_dir)
csvs = sorted(glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True))
print("csv files:", csvs, f"({time.time()-t0:.0f}s)")
for c in csvs:
    head = pd.read_csv(c, nrows=5)
    print("\nHEAD of", os.path.basename(c)); print(head.to_string())
    print("dtypes:\n", head.dtypes)
    tbl = pacsv.read_csv(c)
    print("rows:", tbl.num_rows, "cols:", tbl.num_columns, f"({time.time()-t0:.0f}s)")
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(c))[0] + ".parquet")
    pq.write_table(tbl, out, compression="zstd")
    print("wrote", out, f"{os.path.getsize(out)/1e9:.2f} GB", f"({time.time()-t0:.0f}s)")
