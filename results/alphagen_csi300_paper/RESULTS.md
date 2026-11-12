# AlphaGen reproduction: CSI300 and CSI500, paper setup, 10 seeds

Reproduction of AlphaGen (Yu et al., KDD 2023, arXiv 2306.12964) on the ICM Rysy cluster,
with the paper's data split, target, hyperparameters and step count.

Status on 2026-11-12 08:30: CSI300 complete at pool 20 and pool 100 (10 seeds each), CSI500
complete at pool 20 (10 seeds), reward-0 test complete (3 seeds), transformer runs done for
seeds 0–4, seed ensembles scored. Training: CSI500 at pool 100 (4 of 10 seeds done),
fixed-LSTM seeds 8–9, and the transformer at pool 100 on CSI300 (10 seeds, jobs 215491 on).

## Setup

| Item | Value |
|---|---|
| Command | `INDEX=csi300 sbatch --array=0-9 train.sbatch` → `python -m scripts.rl --instruments csi300 --pool_capacity 20 --random_seeds <seed> --steps 300000` |
| Policy | `LSTMSharedNet` (2 layers, d_model 128, dropout 0.1), MaskablePPO |
| PPO | gamma 1.0, ent_coef 0.01, batch_size 128 |
| Alpha pool | `KddAlphaPool`, capacity 20, l1_alpha 5e-3, Adam lr 5e-4, tolerance 500, mutual IC limit 0.99 |
| Expression length | max 20 tokens; reward for an unfinished expression **-1** (paper text) |
| Steps | 300,000 (paper's code, pool size 20); runs stop at the next 2,048-step boundary, 301,056 |
| Target | `Ref(close, -20) / close - 1` (20-day forward return) |
| Split | train 2009-01-01 – 2018-12-31, validation 2019, **test 2020-01-01 – 2021-12-31** |
| Universe | CSI300 / CSI500, daily membership built from baostock (`data_collection/build_index_membership.py`) |
| Prices | baostock daily bars, back-adjusted, re-referenced to 2023-01-15 (paper footnote 3) |
| Hardware | Tesla V100 32 GB, one GPU and 4 CPU cores per run; about 5–13 h per run |

### Data fixes before these runs

- baostock pages of 2,000 rows could end a download early. 28 files were cut short, so 0.30% of
  CSI300 member-days had no price. They were completed from baostock (4) and Sina (21), with Sina
  checked against baostock's own re-download (identical to float precision on 3,650+ rows).
- One file (sz.002477) held another stock's first page after a download timeout; rebuilt from
  Eastmoney. A scan of all 7,601 files found no other case.
- After the fixes, CSI300 and CSI500 members have no missing prices from 2009 to 2021.

## CSI300 results (test 2020–2021)

Pool IC and Rank IC of the final pool (step 301,056) on each segment, as logged by `scripts/rl.py`.

| Seed | Test IC | Test Rank IC | Valid IC (2019) | Valid Rank IC (2019) | Mean expression length (tokens) |
|---|---|---|---|---|---|
| 0 | 0.0574 | 0.0706 | 0.0411 | 0.0512 | 4.62 |
| 1 | 0.0456 | 0.0452 | -0.0161 | -0.0056 | 6.23 |
| 2 | 0.0437 | 0.0410 | 0.0193 | 0.0287 | 5.44 |
| 3 | 0.0687 | 0.0790 | 0.0097 | 0.0220 | 4.68 |
| 4 | 0.0554 | 0.0588 | 0.0524 | 0.0514 | 4.48 |
| 5 | 0.0509 | 0.0517 | -0.0213 | -0.0110 | 4.87 |
| 6 | 0.0570 | 0.0639 | 0.0083 | 0.0259 | 6.24 |
| 7 | 0.0307 | 0.0383 | -0.0136 | 0.0076 | 6.40 |
| 8 | 0.0725 | 0.0887 | 0.0355 | 0.0536 | 4.58 |
| 9 | 0.0395 | 0.0502 | 0.0397 | 0.0431 | 6.55 |
| **Mean ± std (10 seeds)** | **0.0521 ± 0.0128** | **0.0587 ± 0.0167** | 0.0155 ± 0.0265 | 0.0267 ± 0.0237 | 5.41 |

For reference, seeds 0–2 trained earlier on the uncorrected data (0.30% of CSI300 member-days
without prices) gave test IC 0.0415 / 0.0337 / 0.0525 and Rank IC 0.0494 / 0.0428 / 0.0521,
mean 0.0426 / 0.0481. They are not part of the main result.

## Comparison with the paper

Paper (KDD 2023, Table 2, row "Ours*"), mean (std) over 10 runs:

| | CSI300 IC | CSI300 Rank IC | CSI500 IC | CSI500 Rank IC |
|---|---|---|---|---|
| Paper | 0.0725 (0.0105) | 0.0806 (0.0106) | 0.0438 (0.0064) | 0.0727 (0.0112) |
| This reproduction | 0.0521 (0.0128) | 0.0587 (0.0167) | 0.0323 (0.0069) | 0.0430 (0.0062) |
| Ratio | 72% | 73% | 74% | 59% |

- On both universes the IC is about three quarters of the paper's. The best CSI300 seed (8) matches the
  paper's mean.
- Our seed spread (std 0.013) is close to the paper's (0.011), so the runs behave similarly; but
  with a standard error of about 0.004 the gap of 0.020 on CSI300 is not seed noise.
- **The paper's "Ours\*" is the best of pool sizes {10, 20, 50, 100}** for each universe. The table
  above is pool size 20 only. With pool size 100 (10 seeds, next section) CSI300 reaches
  0.0592 / 0.0636, **82% / 79%** of the paper, so pool size explains about a third of the gap.
- Otherwise the training code matches the authors' KDD version (commit e3b1526) except for the
  reward for unfinished expressions, which turns out not to matter (next section). Remaining
  candidates: data source and membership details, library versions.

## Reward for unfinished expressions: -1 vs 0

The paper's text gives -1; the authors' code at the KDD freeze (commit e3b1526) uses 0. Seeds
0–2 were run with 0 (`ALPHAGEN_INVALID_REWARD=0`, `RUN_TAG=r0`, job 214881) on the same fixed
data, to compare seed by seed with the -1 runs.

| Seed | Reward 0: test IC / Rank IC | Reward -1: test IC / Rank IC |
|---|---|---|
| 0 | 0.0518 / 0.0643 | 0.0574 / 0.0706 |
| 1 | 0.0669 / 0.0626 | 0.0456 / 0.0452 |
| 2 | 0.0340 / 0.0284 | 0.0437 / 0.0410 |
| **Mean** | **0.0509 / 0.0518** | **0.0489 / 0.0523** |

No difference beyond seed noise. With either value expressions shrink from 19 to about 5–7
tokens within the first 50k steps, so the -1 penalty is not what makes them short, and it does
not explain the gap to the paper.

## CSI500 results (test 2020–2021)

Same setup, `INDEX=csi500` (jobs 214954 and 214969).

| Seed | Test IC | Test Rank IC | Valid IC (2019) | Valid Rank IC (2019) | Mean expression length (tokens) |
|---|---|---|---|---|---|
| 0 | 0.0260 | 0.0391 | -0.0052 | -0.0111 | 6.78 |
| 1 | 0.0326 | 0.0440 | 0.0303 | 0.0230 | 5.52 |
| 2 | 0.0346 | 0.0482 | -0.0017 | -0.0001 | 5.08 |
| 3 | 0.0193 | 0.0354 | -0.0188 | -0.0201 | 6.42 |
| 4 | 0.0348 | 0.0384 | 0.0073 | -0.0043 | 4.12 |
| 5 | 0.0294 | 0.0338 | 0.0001 | -0.0054 | 9.76 |
| 6 | 0.0299 | 0.0471 | 0.0085 | 0.0025 | 6.10 |
| 7 | 0.0442 | 0.0451 | -0.0150 | -0.0206 | 5.11 |
| 8 | 0.0398 | 0.0534 | 0.0130 | 0.0115 | 4.42 |
| 9 | 0.0320 | 0.0457 | 0.0099 | 0.0031 | 5.46 |
| **Mean ± std (10 seeds)** | **0.0323 ± 0.0069** | **0.0430 ± 0.0062** | 0.0028 ± 0.0143 | -0.0021 ± 0.0135 | 5.88 |

The Rank IC gap is larger on CSI500 (59% of the paper) than the IC gap (74%).

## Pool size (CSI300)

Same setup with pool capacity 10, 50 and 100 and the step counts of the authors' code
(250k / 350k / 400k; jobs 215001–215003). Seed-matched means, test 2020–2021:

| Pool size | Seeds | Test IC | Test Rank IC | Same seeds at pool 20 (IC / Rank IC) |
|---|---|---|---|---|
| 10 | 0–1 | 0.0377 ± 0.0006 | 0.0546 ± 0.0001 | 0.0515 / 0.0579 |
| 20 | 0–4 | 0.0542 | 0.0589 | — |
| 50 | 0–4 | 0.0577 ± 0.0181 | 0.0639 ± 0.0171 | 0.0542 / 0.0589 |
| 100 | 0–4 | **0.0617 ± 0.0108** | **0.0677 ± 0.0074** | 0.0542 / 0.0589 |
| 100 | 0–9 | **0.0592 ± 0.0085** | **0.0636 ± 0.0086** | 0.0521 / 0.0587 |

Per seed (test IC / Rank IC):

| Seed | Pool 10 | Pool 50 | Pool 100 |
|---|---|---|---|
| 0 | 0.0372 / 0.0546 | 0.0610 / 0.0741 | 0.0749 / 0.0780 |
| 1 | 0.0381 / 0.0545 | 0.0393 / 0.0487 | 0.0548 / 0.0605 |
| 2 | | 0.0429 / 0.0494 | 0.0717 / 0.0725 |
| 3 | | 0.0607 / 0.0591 | 0.0513 / 0.0655 |
| 4 | | 0.0848 / 0.0883 | 0.0560 / 0.0620 |
| 5 | | | 0.0563 / 0.0627 |
| 6 | | | 0.0587 / 0.0555 |
| 7 | | | 0.0479 / 0.0484 |
| 8 | | | 0.0566 / 0.0601 |
| 9 | | | 0.0636 / 0.0709 |

- IC rises with pool size, as in the paper, and pool 100 is the best of the four, so it is the
  setting to compare with "Ours\*": 0.0592 ± 0.0085 / 0.0636 ± 0.0086 over 10 seeds (seeds 5–9:
  job 215199), 82% / 79% of the paper. Pool 100 also has the smallest seed spread.
- CSI500 at pool 100 is training (jobs 215208, 215411).
- Pool 100 runs take 18–24 h on a V100, against 7–12 h at pool 20.

## Seed ensemble

The final pools of all seeds combined into one predictor (`scripts/ensemble.py`, job 215453,
output in `ensemble_scores.txt`): each pool's prediction (weighted sum of its day-normalised
alphas) is normalised by day, then the seeds are averaged with equal weight. The single-pool
scores it prints match the training logs exactly. Test 2020–2021, IC / Rank IC:

| Group | Pools | Mean of seeds | Best seed (IC) | Ensemble |
|---|---|---|---|---|
| CSI300 pool 20 | 10 | 0.0521 / 0.0587 | 0.0725 | 0.0662 / 0.0712 |
| CSI300 pool 50 | 5 | 0.0577 / 0.0639 | 0.0848 | 0.0720 / 0.0743 |
| CSI300 pool 100 | 10 | 0.0592 / 0.0636 | 0.0749 | **0.0752 / 0.0791** |
| CSI300 pools 20 + 50 + 100 | 25 | 0.0561 / 0.0617 | 0.0848 | 0.0743 / 0.0763 |
| CSI300 transformer + fix, pool 20 | 5 | 0.0484 / 0.0500 | 0.0656 | 0.0637 / 0.0653 |
| CSI300 LSTM + fix, pool 20 | 5 | 0.0387 / 0.0456 | 0.0551 | 0.0476 / 0.0575 |
| CSI500 pool 20 | 10 | 0.0323 / 0.0430 | 0.0442 | 0.0368 / 0.0501 |

- Averaging seeds adds 0.009–0.016 IC on CSI300 in every group. The 10-seed ensemble at pool 100
  (0.0752 / 0.0791) is at the paper's "Ours\*" (0.0725 / 0.0806). That is a different quantity
  from the paper's mean over runs: it uses 10 training runs for one predictor.
- The gain is smaller on CSI500 (+0.0045 IC, +0.0071 Rank IC).
- Validation IC (2019) also rises with the ensemble but stays near 0.02, so picking checkpoints
  or seeds by validation IC is unlikely to be reliable.
- Adding pool 20 and 50 to the pool-100 seeds does not help (0.0743 against 0.0752).

## Transformer policy (first results)

`ALPHAGEN_POLICY=transformer` replaces the LSTM with `TransformerSharedNet` of similar size
(2 layers, d_model 128, 4 heads, feed-forward 512, dropout 0.1; 403k parameters against 271k).
Two fixes are switched on for it (`ALPHAGEN_OBS_OFFSET=1`):

- The transformer's output is averaged over real tokens only; the original averaged over all
  21 positions, so padding dominated short expressions.
- Observations stored action ids with 0 as padding, so action 0 (`Abs`) was invisible to the
  policy; for the LSTM, using `Abs` also hid the last token. Actions are now stored as id + 1.
  This bug is in the authors' code, so all LSTM results above have it.

Pool 20, CSI300, test IC / Rank IC. Transformer `RUN_TAG=tf` (jobs 215023, 215201); LSTM with
the observation fix `RUN_TAG=fix` (job 215200), as the control for the fix:

| Seed | LSTM (paper code) | LSTM + fix | Transformer + fix |
|---|---|---|---|
| 0 | 0.0574 / 0.0706 | 0.0341 / 0.0314 | 0.0467 / 0.0395 |
| 1 | 0.0456 / 0.0452 | 0.0331 / 0.0460 | 0.0486 / 0.0576 |
| 2 | 0.0437 / 0.0410 | 0.0551 / 0.0593 | 0.0320 / 0.0208 |
| 3 | 0.0687 / 0.0790 | 0.0447 / 0.0438 | 0.0490 / 0.0594 |
| 4 | 0.0554 / 0.0588 | 0.0266 / 0.0475 | 0.0656 / 0.0728 |
| **Mean** | **0.0542 / 0.0589** | **0.0387 / 0.0456** | **0.0484 / 0.0500** |

Paired over seeds 0–4 (test IC):

- **The observation fix lowers the LSTM's IC**, by 0.0154 on average, on 4 of 5 seeds (paired
  t-test p = 0.10). This is unexpected: making `Abs` visible should not hurt. Seeds 5–9 of the
  fixed LSTM are running (job 215412) to see whether it holds.
- **With the fix on both, the transformer beats the LSTM** by 0.0097, on 4 of 5 seeds (p = 0.39).
- Against the paper's LSTM without the fix, the transformer is 0.0058 lower (3 of 5 seeds, p = 0.35).
- None of these differences is significant with 5 seeds; the seed spread (std about 0.011) is
  as large as the effects.

## Caveats

- Validation IC on 2019 is close to zero for many seeds while test IC is positive, so the
  pools' quality varies a lot from year to year.
- Final pools contain many very short alphas (for example `Less(2.0,$close)`, `Div(-1.0,$close)`),
  reflecting the short expressions the policy converges to.
- Membership is built from baostock's index constituents. The authors' pipeline used Qlib's
  membership list; scoring finished pools on that list did not close the gap (test IC 0.035 vs
  0.042 on the old-data seeds).
- 28 truncated price files were completed from Sina, and one from Eastmoney (see Data fixes).

## Next steps

1. CSI500 at pool 100, 10 seeds (running), to finish the reproduction on both universes.
2. Fixed LSTM seeds 5–9 (running), to settle whether the observation fix matters.
3. Main comparison: LSTM vs transformer at pool 100, 10 seeds each, on CSI300 then CSI500.
   The transformer runs on CSI300 are queued (`RUN_TAG=tf POOL=100`); the LSTM side is the
   existing pool-100 runs.
4. Compare the policies as seed ensembles too, since that is the strongest predictor so far.

## Files

- `pools/seed<N>_301056_steps_pool.json`: final CSI300 alpha pool of each seed (20 expressions and weights)
- `pools/seed<N>_r0_301056_steps_pool.json`: final pools of the reward-0 runs, seeds 0–2
- `pools_csi500/seed<N>_301056_steps_pool.json`: final CSI500 pool of each seed
- `pools_poolsize/pool<P>_seed<N>_<steps>_steps_pool.json`: final CSI300 pools at pool sizes 10, 50, 100
- `pools_transformer/seed<N>_301056_steps_pool.json`: final pools of the transformer runs
- `pools_fix/seed<N>_301056_steps_pool.json`: final pools of the LSTM runs with the observation fix
- `pools_olddata/seed{0,1,2}.json`: final pools of the earlier runs on the uncorrected data
- `ensemble_scores.txt`: output of the seed-ensemble scoring (`scripts/ensemble.py`,
  `scripts/rysy/ensemble.sbatch`)
