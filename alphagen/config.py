import os
from typing import Type
from alphagen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs,  # Sign,
    Log,
    # Binary
    Add, Sub, Mul, Div, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var,  # Skew, Kurt,
    Max, Min,
    Med, Mad,  # Rank,
    Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]

DELTA_TIMES = [10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
# Reward for an expression that hits MAX_EXPR_LENGTH unfinished: the paper text says -1,
# the paper's code (commit e3b1526) uses 0
INVALID_EXPR_REWARD = float(os.environ.get("ALPHAGEN_INVALID_REWARD", "-1"))
# Observations store action ids with 0 as padding, so action 0 (Abs) looks like padding to the
# policy nets (paper's code). 1 stores action + 1 instead, keeping 0 for padding only.
OBS_OFFSET = int(os.environ.get("ALPHAGEN_OBS_OFFSET", "0"))
