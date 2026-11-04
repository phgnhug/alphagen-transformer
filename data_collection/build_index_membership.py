"""
Build qlib instrument files (csi300.txt / csi500.txt) from baostock's historical index constituents.

baostock's query_hs300_stocks(date) / query_zz500_stocks(date) return the constituents in effect
on a date. We sample every two weeks, then, wherever two consecutive samples differ, bisect over
the trading days in between until each change is pinned to the exact day. Every query is cached
in ../data/membership/<index>_queries.json, so an interrupted run resumes where it stopped.

Usage:
    python build_index_membership.py csi300 [--seed-from <existing csi300.txt>]
    python build_index_membership.py csi500

--seed-from turns an instrument file built from the same two-week samples into cached samples,
so only the bisection queries hit the network.
"""

import json
import os
import socket
import sys
import time
from typing import Dict, List, Optional, Tuple

import baostock as bs
import pandas as pd

QLIB_PATH = os.path.expanduser("~/.qlib/qlib_data/cn_data_2023ref")
CACHE_DIR = "../data/membership"
SAMPLE_START = "2007-01-01"
SAMPLE_FREQ = "14D"

QUERIES = {
    "csi300": bs.query_hs300_stocks,
    "csi500": bs.query_zz500_stocks,
}


class MembershipQuery:
    def __init__(self, index: str, calendar: pd.DatetimeIndex):
        self._query = QUERIES[index]
        self._calendar = calendar
        self._cache_path = f"{CACHE_DIR}/{index}_queries.json"
        self.cache: Dict[str, List[str]] = {}
        if os.path.exists(self._cache_path):
            with open(self._cache_path) as f:
                self.cache = json.load(f)
        self.n_network = 0

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self.cache, f)

    def members(self, day_idx: int, retries: int = 5) -> frozenset:
        date = self._calendar[day_idx].strftime("%Y-%m-%d")
        if date in self.cache:
            return frozenset(self.cache[date])
        # baostock sockets have no timeout by default, so a lost reply hangs forever
        for attempt in range(retries):
            try:
                rs = self._query(date=date)
                if rs.error_code != "0":
                    raise RuntimeError(rs.error_msg)
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                break
            except Exception as e:
                print(f"{date}: attempt {attempt + 1} failed ({e}), re-login", flush=True)
                time.sleep(2)
                bs.logout()
                bs.login()
        else:
            raise RuntimeError(f"{date}: all {retries} attempts failed")
        self.cache[date] = sorted(r[1] for r in rows)
        self.n_network += 1
        if self.n_network % 20 == 0:
            self.save()
            print(f"  {self.n_network} network queries, last {date}", flush=True)
        return frozenset(self.cache[date])


def seed_cache(q: MembershipQuery, path: str, sample_idx: List[int], calendar: pd.DatetimeIndex) -> None:
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "start", "end"])
    df["code"] = df["code"].str[:2].str.lower() + "." + df["code"].str[2:]
    for i in sample_idx:
        day = calendar[i]
        date = day.strftime("%Y-%m-%d")
        if date not in q.cache:
            q.cache[date] = sorted(df.loc[(pd.to_datetime(df["start"]) <= day) &
                                          (pd.to_datetime(df["end"]) >= day), "code"])
    q.save()


def bisect_changes(q: MembershipQuery, lo: int, hi: int) -> None:
    "Query trading days in (lo, hi) until every change between them falls on adjacent days."
    if hi - lo <= 1 or q.members(lo) == q.members(hi):
        return
    mid = (lo + hi) // 2
    q.members(mid)
    bisect_changes(q, lo, mid)
    bisect_changes(q, mid, hi)


def to_intervals(q: MembershipQuery, calendar: pd.DatetimeIndex) -> List[Tuple[str, str, str]]:
    points = sorted((calendar.get_loc(pd.Timestamp(d)), frozenset(m)) for d, m in q.cache.items()
                    if pd.Timestamp(d) in calendar)
    intervals = []
    open_since: Dict[str, int] = {}
    prev_idx: Optional[int] = None
    for idx, members in points:
        if prev_idx is not None:
            for code in [c for c in open_since if c not in members]:
                intervals.append((code, open_since.pop(code), idx - 1))
        for code in members:
            open_since.setdefault(code, idx)
        prev_idx = idx
    last = len(calendar) - 1
    intervals += [(code, start, last) for code, start in open_since.items()]

    def qlib_code(code: str) -> str:
        return code.replace(".", "").upper()

    return sorted(
        (qlib_code(c), calendar[s].strftime("%Y-%m-%d"), calendar[e].strftime("%Y-%m-%d"))
        for c, s, e in intervals
    )


def main(index: str, seed_from: Optional[str] = None):
    calendar = pd.DatetimeIndex(pd.read_csv(
        f"{QLIB_PATH}/calendars/day.txt", header=None)[0])
    sample_days = pd.date_range(SAMPLE_START, calendar[-1], freq=SAMPLE_FREQ)
    sample_idx = sorted({min(int(calendar.searchsorted(d)), len(calendar) - 1) for d in sample_days}
                        | {len(calendar) - 1})

    q = MembershipQuery(index, calendar)
    if seed_from is not None:
        seed_cache(q, seed_from, sample_idx, calendar)

    socket.setdefaulttimeout(30)
    bs.login()
    try:
        print(f"[{index}] sampling {len(sample_idx)} dates", flush=True)
        for i in sample_idx:
            q.members(i)
        print(f"[{index}] bisecting changes", flush=True)
        for lo, hi in zip(sample_idx[:-1], sample_idx[1:]):
            bisect_changes(q, lo, hi)
    finally:
        q.save()
        bs.logout()

    intervals = to_intervals(q, calendar)
    path = f"{QLIB_PATH}/instruments/{index}.txt"
    pd.DataFrame(intervals).to_csv(path, sep="\t", header=False, index=False)
    print(f"[{index}] {q.n_network} network queries; wrote {len(intervals)} intervals to {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = args[args.index("--seed-from") + 1] if "--seed-from" in args else None
    main(args[0], seed)
