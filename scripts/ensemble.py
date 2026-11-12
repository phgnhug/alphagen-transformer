# Score final alpha pools on the test (2020-2021) and validation (2019) segments, one by one and
# as seed ensembles: each pool's prediction (weighted sum of day-normalised alphas, as in
# LinearAlphaPool.test_ensemble) is normalised by day and the seeds are averaged with equal weight.
# Usage: python ensemble.py <index> <group>=<glob>[,<glob>...] [<group>=... ...]
import glob, json, sys
import torch
from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib

PARSER = ExpressionParser(Operators, ignore_case=True, non_positive_time_deltas_allowed=False,
                          additional_operator_mapping={"Max": [Greater], "Min": [Less], "Delta": [Sub]})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEGMENTS = {"valid": ("2019-01-01", "2019-12-31"), "test": ("2020-01-01", "2021-12-31")}

index, groups = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
initialize_qlib("~/.qlib/qlib_data/cn_data_2023ref")
close = Feature(FeatureType.CLOSE)
target = Ref(close, -20) / close - 1
calcs = {k: QLibStockDataCalculator(StockData(index, s, e, device=DEVICE), target)
         for k, (s, e) in SEGMENTS.items()}


def pool_value(path: str, calc: QLibStockDataCalculator) -> torch.Tensor:
    p = json.load(open(path))
    exprs = [PARSER.parse(e) for e in p["exprs"]]
    with torch.no_grad():
        return normalize_by_day(calc.make_ensemble_alpha(exprs, p["weights"]))


def score(value: torch.Tensor, calc: QLibStockDataCalculator) -> tuple:
    return calc._calc_IC(value, calc.target), calc._calc_rIC(value, calc.target)


for name, pattern in groups:
    paths = sorted(p for pat in pattern.split(",") for p in glob.glob(pat))
    print(f"\n== {name}: {len(paths)} pools")
    for seg, calc in calcs.items():
        values = [pool_value(p, calc) for p in paths]
        singles = torch.tensor([score(v, calc) for v in values])
        ens = score(normalize_by_day(torch.stack(values).nanmean(0)), calc)
        for p, (ic, ric) in zip(paths, singles.tolist()):
            print(f"  {seg} {p.rsplit('/', 1)[-1]:42s} IC {ic:.4f}  RankIC {ric:.4f}")
        print(f"  {seg} MEAN OF SEEDS  IC {singles[:, 0].mean():.4f}  RankIC {singles[:, 1].mean():.4f}"
              f"   BEST SEED IC {singles[:, 0].max():.4f}")
        print(f"  {seg} ENSEMBLE       IC {ens[0]:.4f}  RankIC {ens[1]:.4f}")
