"""Resumable, stall-tolerant download of the AuctionNet period zips from Aliyun OSS.

For each file: HEAD for Content-Length; if a partial file exists, resume with a Range request; abort a transfer that
delivers no bytes for STALL seconds and retry (up to RETRIES times); verify the final size.
Usage: python download_all.py <raw_dir> [periods: 7-8 9-10 ...]
"""
import os
import sys
import time
import urllib.request

BASE = "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/"
NAME = "autoBidding_general_track_final_data_period_{}.zip"
STALL, RETRIES, CHUNK = 90, 8, 1 << 20
raw = sys.argv[1]
periods = sys.argv[2:] or ["7-8", "9-10", "11-12", "14-15", "16-17", "18-19", "20-21", "22-23", "24-25", "26-27"]
os.makedirs(raw, exist_ok=True)


def content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(url, path, total):
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have >= total:
        return True
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req, timeout=STALL) as r, open(path, "ab") as f:
        last = time.time()
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk); have += len(chunk)
            now = time.time()
            if now - last > STALL:
                raise TimeoutError("stall")
            last = now
    return os.path.getsize(path) >= total


for p in periods:
    name = NAME.format(p); url = BASE + name; path = os.path.join(raw, name)
    total = content_length(url)
    t0 = time.time()
    ok = False
    for attempt in range(RETRIES):
        try:
            ok = fetch(url, path, total)
            if ok:
                break
        except Exception as e:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  {name}: attempt {attempt + 1} failed ({type(e).__name__}) at {have / 1e9:.2f} of {total / 1e9:.2f} GB; resuming", flush=True)
            time.sleep(5)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"{name}: {'OK' if ok and size == total else 'INCOMPLETE'} {size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
print("done")
