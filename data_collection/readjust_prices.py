"""
Re-reference the cached baostock prices to the adjustment factors of 2023-01-15,
as stated in the paper (footnote 3), and dump them into a new qlib directory.

The cached k_data pickles were built with adjust_date="2009-01-01", i.e.
    price = raw_price * f(t) / f(2009-01-01),  volume = raw_volume / (f(t) / f(2009-01-01))
with f the back-adjust factor. Changing the reference date only rescales each stock by the
constant f(ref) / f(2009-01-01), which equals the pickle's own factor column at the new
reference date, so no re-download is needed.

The reference row is picked with the same as_of_date rule as DataManager._download_stock_data_job.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from qlib_dump_bin import DumpDataAll

ADJUST_DATE = "2023-01-15"
ADJUST_TYPE = "backAdjustFactor"
PRICE_FIELDS = ["open", "high", "low", "close", "preclose"]

K_DATA_PATH = "../data/k_data"
EXPORT_PATH = "../data/export_2023ref"
QLIB_EXPORT_PATH = os.path.expanduser("~/.qlib/qlib_data/cn_data_2023ref")


def as_of_date(df: pd.DataFrame, date: str) -> pd.Series:
    index: int = df.index.searchsorted(date, side="right") - 1   # type: ignore
    return df.iloc[index]


def readjust_job(path: Path) -> None:
    df: pd.DataFrame = pd.read_pickle(path)
    ref_factor = as_of_date(df, ADJUST_DATE)[ADJUST_TYPE]
    df[PRICE_FIELDS + [ADJUST_TYPE]] /= ref_factor
    df["volume"] *= ref_factor
    df["vwap"] = df["amount"] / df["volume"]

    code = path.stem
    code = f"{code[:2].upper()}{code[-6:]}"
    df.rename(columns={ADJUST_TYPE: "factor"}, inplace=True)
    df["code"] = code
    df.to_csv(Path(EXPORT_PATH) / f"{code}.csv")


def main(dump_only: bool = False):
    if not dump_only:
        os.makedirs(EXPORT_PATH, exist_ok=True)
        paths = list(Path(K_DATA_PATH).iterdir())
        with ProcessPoolExecutor(8) as executor:
            list(tqdm(executor.map(readjust_job, paths, chunksize=32), total=len(paths)))

    DumpDataAll(
        csv_path=EXPORT_PATH,
        qlib_dir=QLIB_EXPORT_PATH,
        max_workers=8,
        exclude_fields="date,code",
        symbol_field_name="code"
    ).dump()
    calendars = f"{QLIB_EXPORT_PATH}/calendars"
    Path(f"{calendars}/day_future.txt").write_text(Path(f"{calendars}/day.txt").read_text())
    print(f"Dumped to {QLIB_EXPORT_PATH}")


if __name__ == "__main__":
    main(dump_only="--dump-only" in sys.argv)
