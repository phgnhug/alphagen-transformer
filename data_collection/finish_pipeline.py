import pandas as pd
from fetch_baostock_data import DataManager

if __name__ == "__main__":
    dm = DataManager(
        save_path="../data",
        qlib_export_path="~/.qlib/qlib_data/cn_data_2024h1",
        qlib_base_data_path="~/.qlib/qlib_data/cn_data",
        adjust_date="2009-01-01",
        max_workers=10,
        retry_wait_seconds=2.
    )

    dm._basic_info = pd.read_csv(f"{dm._save_path}/basic_info.csv", index_col=0)
    dm._adjust_factors = pd.read_csv(f"{dm._save_path}/adjust_factors.csv", index_col=[0, 1])

    print("Export to csv")
    dm._save_csv()

    print("Dump qlib data")
    dm._dump_qlib_data()

    print("Fix constituents")
    dm._fix_constituents()

    print("Done.")