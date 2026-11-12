#!/bin/bash
# Score final pools singly and as seed ensembles (ensemble.py), test 2020-2021 and valid 2019.
#SBATCH -J ag-ens
#SBATCH -A g107-2971
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.log
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alphagen
cd ~/alphagen
P=results/alphagen_csi300_paper
export PYTHONPATH=$PWD
python -u scripts/ensemble.py csi300 \
    "pool20=$P/pools/seed?_301056_steps_pool.json" \
    "pool50=$P/pools_poolsize/pool50_seed?_*_steps_pool.json" \
    "pool100=$P/pools_poolsize/pool100_seed?_*_steps_pool.json" \
    "pool20_50_100=$P/pools/seed?_301056_steps_pool.json,$P/pools_poolsize/pool50_*,$P/pools_poolsize/pool100_*" \
    "transformer=$P/pools_transformer/seed?_*_steps_pool.json" \
    "lstm_fix=$P/pools_fix/seed?_*_steps_pool.json"
python -u scripts/ensemble.py csi500 "csi500_pool20=$P/pools_csi500/seed?_*_steps_pool.json"
