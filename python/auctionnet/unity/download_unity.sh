#!/usr/bin/env bash
# Resumable download of all 11 AuctionNet period zips on the Unity login node (transfers are allowed there).
# Usage: nohup bash download_unity.sh /home/magnolfi.1/NoRegret_data/auctionnet/raw > download.log 2>&1 &
set -u
RAW="${1:-$HOME/NoRegret_data/auctionnet/raw}"
BASE="https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final"
mkdir -p "$RAW"; cd "$RAW"
for p in 7-8 9-10 11-12 13 14-15 16-17 18-19 20-21 22-23 24-25 26-27; do
  f="autoBidding_general_track_final_data_period_${p}.zip"
  total=$(curl -sI "$BASE/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  for attempt in 1 2 3 4 5 6 7 8; do
    have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
    if [ -n "$total" ] && [ "$have" -ge "$total" ]; then break; fi
    # resume; abort if the transfer delivers < 20 KB/s for 90 s, then retry
    curl -sL -C - --speed-time 90 --speed-limit 20000 --max-time 3600 -o "$f" "$BASE/$f"
    sleep 5
  done
  have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
  echo "$f size=$have expected=$total $( [ "$have" = "$total" ] && echo OK || echo INCOMPLETE ) $(date +%H:%M:%S)"
done
echo ALLDONE
