#!/bin/bash
# One AlphaGen run per array task: CSI300, paper split, seed = array index.
#   sbatch --array=0-9 train_csi300.sbatch            (pool size 20)
#   POOL=50 sbatch --array=0-9 train_csi300.sbatch    (other pool sizes)
# The normal QOS caps jobs at 2 days, so each task resumes from its latest checkpoint
# and resubmits itself shortly before the limit until the paper's step count is reached.
#SBATCH -J ag-csi300
#SBATCH -A g107-2971
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/%x_%A_%a.log
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alphagen
cd ~/alphagen

POOL=${POOL:-20}
SEED=$SLURM_ARRAY_TASK_ID
# Total steps per pool size, from the paper's code (train_maskable_ppo.py, commit e3b1526)
declare -A TOTAL=([10]=250000 [20]=300000 [50]=350000 [100]=400000)
TARGET=${STEPS:-${TOTAL[$POOL]}}

latest=$(ls out/results/csi300_${POOL}_${SEED}_*_rl/*_steps_pool.json 2>/dev/null \
         | awk -F/ '{n=$NF; sub(/_steps_pool.json/, "", n); z=$0; sub(/_pool.json$/, ".zip", z); print n, z}' | sort -n | tail -1 || true)
DONE=0
if [ -n "$latest" ]; then
    DONE=${latest%% *}
    export RESUME_CHECKPOINT=${latest#* }
fi
REMAINING=$((TARGET - DONE))
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  seed $SEED  pool $POOL  done $DONE / $TARGET"
if [ "$REMAINING" -le 0 ]; then
    echo "already finished"
    exit 0
fi

resubmit() {
    echo "time limit near, resubmitting seed $SEED"
    POOL=$POOL STEPS=${STEPS:-} sbatch --export=ALL --array="$SEED" -J "$SLURM_JOB_NAME" train_csi300.sbatch
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" || true
    exit 0
}
trap resubmit USR1

python -u -m scripts.rl --instruments csi300 --pool_capacity "$POOL" --random_seeds "$SEED" --steps "$REMAINING" &
PID=$!
wait "$PID"
