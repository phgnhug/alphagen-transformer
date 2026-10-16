# LSTM baseline: CSI300, seed 0

First full run of the original AlphaGen (LSTM policy), reproduced locally on baostock data.

## Setup

| Item | Value |
|---|---|
| Command | `python -m scripts.rl --instruments csi300 --pool_capacity 20 --random_seeds 0` |
| Policy | `LSTMSharedNet` (2 layers, d_model 128, dropout 0.1), MaskablePPO |
| PPO | gamma 1.0, ent_coef 0.01, batch_size 128 |
| Pool capacity | 20 (MSE alpha pool, l1_alpha 5e-3) |
| Steps | 251,904 (target 250,000; checkpoints every 2,048 steps) |
| Device | Apple M4, MPS backend |
| Wall-clock | ~35 hours |
| Data | baostock daily data (7,566 instruments), qlib format in `~/.qlib/qlib_data/cn_data_2024h1` |
| Universe | `csi300.txt` copied from qlib's default (Yahoo-sourced) data |
| Training segment | 2012-01-01 to 2019-12-31 |
| Test segments | 2020-01-01 to 2020-04-30, 2020-05-01 to 2020-07-31, 2020-08-01 to 2020-09-01 |

## Final metrics (step 251,904)

| Metric | Value |
|---|---|
| Pool IC on training segment (`best_ic_ret`) | 0.0989 |
| Pool size / significant alphas | 20 / 20 |
| Test IC, segments 1 / 2 / 3 | 0.192 / 0.0747 / 0.203 |
| **Test IC, day-weighted mean** | **0.145** |
| Test Rank IC, segments 1 / 2 / 3 | 0.183 / 0.104 / 0.182 |
| **Test Rank IC, day-weighted mean** | **0.151** |

Paper (KDD 2023), CSI300: test IC 0.0725, Rank IC 0.0806, averaged over 10 seeds.

## Caveats

- `best_ic_ret` is the pool's IC on the **training** segment. It is not comparable with the paper's test IC.
- Single seed. The paper averages 10, so the seed-to-seed spread is unknown here.
- The test window (Jan-Aug 2020) is shorter than the paper's (2020-2021).
- CSI300 membership comes from qlib's default data, not from baostock.
- Numbers are therefore not yet directly comparable with the paper.

## Files

- `251904_steps.zip`: final MaskablePPO checkpoint
- `251904_steps_pool.json`: final alpha pool (20 expressions and weights)
- `training_log.txt.gz`: full training log

## Reproducing

1. Build the data: `data_collection/fetch_baostock_data.py`, then `data_collection/finish_pipeline.py`.
2. Copy `csi300.txt`, `csi500.txt`, `csi100.txt` from the default qlib data into `instruments/`.
3. Run the command above.
