"""
Download stocks that appear in the rebuilt index membership files but were never fetched.

The original download started from today's stock list, so constituents that were delisted or
merged before then (e.g. sh.600001, sh.600005) have no k_data. This fetches their basic info,
adjust factors and price history with the same DataManager code as the main pipeline, appends
them to the cached csvs, then re-references them to 2023-01-15 (see readjust_prices.py).

Run readjust_prices.py's dump step afterwards to rebuild the qlib directory.
"""

import os
import socket
from pathlib import Path

import pandas as pd

from baostock_utils import baostock_login, baostock_logout
from fetch_baostock_data import DataManager
from readjust_prices import QLIB_EXPORT_PATH, readjust_job

INDICES = ["csi300", "csi500"]


def missing_codes(dm: DataManager) -> list:
    codes = set()
    for index in INDICES:
        path = f"{QLIB_EXPORT_PATH}/instruments/{index}.txt"
        if os.path.exists(path):
            codes |= set(pd.read_csv(path, sep="\t", header=None)[0])
    codes = {f"{c[:2].lower()}.{c[2:]}" for c in codes}
    return sorted(c for c in codes if not os.path.exists(f"{dm._save_path}/k_data/{c}.pkl"))


def main():
    dm = DataManager(
        save_path="../data",
        qlib_export_path="~/.qlib/qlib_data/cn_data_2024h1",
        qlib_base_data_path="~/.qlib/qlib_data/cn_data",
        adjust_date="2009-01-01",
        max_workers=1,
        max_retries=10,
        retry_wait_seconds=3.
    )
    basic_info = pd.read_csv(f"{dm._save_path}/basic_info.csv", index_col=0)
    adjust_factors = pd.read_csv(f"{dm._save_path}/adjust_factors.csv", index_col=[0, 1])
    codes = missing_codes(dm)
    print(f"{len(codes)} members without data: {codes}", flush=True)

    socket.setdefaulttimeout(30)
    baostock_login()
    try:
        new_info = pd.concat([dm._fetch_basic_info_job(c) for c in codes]).set_index("code")
        new_info = new_info[~new_info.index.isin(basic_info.index)]
        basic_info = pd.concat([basic_info, new_info])
        basic_info.to_csv(f"{dm._save_path}/basic_info.csv")

        new_adj = [dm._fetch_adjust_factors_job(c, "1990-12-19") for c in codes]
        new_adj = [df for df in new_adj if not df.empty]
        if new_adj:
            new_adj = pd.concat(new_adj).set_index(["code", "dividOperateDate"])
            new_adj = new_adj[~new_adj.index.isin(adjust_factors.index)]
            adjust_factors = pd.concat([adjust_factors, new_adj]).sort_index()
            adjust_factors.to_csv(f"{dm._save_path}/adjust_factors.csv")

        dm._basic_info = basic_info
        dm._adjust_factors = pd.read_csv(f"{dm._save_path}/adjust_factors.csv", index_col=[0, 1])
        for c in codes:
            if c not in basic_info.index:
                print(f"  {c}: no basic info on baostock, skipped", flush=True)
                continue
            try:
                dm._download_stock_data_job(c, basic_info.loc[c])
            except IndexError:
                # No rows since 2008-01-01 (the download start), so nothing to adjust
                print(f"  {c}: no price data since 2008, skipped", flush=True)
                continue
            path = Path(f"{dm._save_path}/k_data/{c}.pkl")
            if path.exists():
                readjust_job(path)
                print(f"  {c}: {len(pd.read_pickle(path))} rows", flush=True)
            else:
                print(f"  {c}: no price data", flush=True)
    finally:
        baostock_logout()


if __name__ == "__main__":
    main()
