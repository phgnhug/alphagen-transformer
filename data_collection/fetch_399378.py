import pandas as pd
from fetch_baostock_data import DataManager
from baostock_utils import baostock_login

dm = DataManager(
    save_path="../data",
    qlib_export_path="~/.qlib/qlib_data/cn_data_2024h1",
    qlib_base_data_path="~/.qlib/qlib_data/cn_data",
    adjust_date="2009-01-01",
    max_workers=1,
    max_retries=10,
    retry_wait_seconds=3.
)

dm._basic_info = pd.read_csv(f"{dm._save_path}/basic_info.csv", index_col=0)
dm._adjust_factors = pd.read_csv(f"{dm._save_path}/adjust_factors.csv", index_col=[0, 1])

baostock_login()

code = "sz.399378"
data = dm._basic_info.loc[code]
dm._download_stock_data_job(code, data)
print(f"Saved {code}")
