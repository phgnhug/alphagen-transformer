#!/bin/bash
# One AlphaGen run per array task: paper split, seed = array index, any index universe.
#   INDEX=csi500 sbatch -J ag-csi500 --array=0-9 train.sbatch            (pool size 20)
#   INDEX=csi500 POOL=50 sbatch -J ag-csi500 --array=0-9 train.sbatch    (other pool sizes)
#   RUN_TAG=r0 ALPHAGEN_INVALID_REWARD=0 sbatch -J ag-r0 --array=0-2 train.sbatch   (variant, own results)
#   RUN_TAG=tf ALPHAGEN_POLICY=transformer ALPHAGEN_OBS_OFFSET=1 sbatch -J ag-tf --array=0-9 train.sbatch
# Same as train_csi300.sbatch (kept for the CSI300 jobs that resubmit it) with INDEX added.
# The normal QOS caps jobs at 2 days, so each task resumes from its latest checkpoint
# and resubmits itself shortly before the limit until the paper's step count is reached.
#SBATCH -J ag-train
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

INDEX=${INDEX:-csi300}
POOL=${POOL:-20}
export RUN_TAG=${RUN_TAG:-}
export ALPHAGEN_INVALID_REWARD=${ALPHAGEN_INVALID_REWARD:--1}
export ALPHAGEN_POLICY=${ALPHAGEN_POLICY:-lstm}
export ALPHAGEN_OBS_OFFSET=${ALPHAGEN_OBS_OFFSET:-0}
SEED=$SLURM_ARRAY_TASK_ID
# Total steps per pool size, from the paper's code (train_maskable_ppo.py, commit e3b1526)
declare -A TOTAL=([10]=250000 [20]=300000 [50]=350000 [100]=400000)
TARGET=${STEPS:-${TOTAL[$POOL]}}

latest=$(ls out/results/${INDEX}_${POOL}_${SEED}_*_rl${RUN_TAG:+_$RUN_TAG}/*_steps_pool.json 2>/dev/null \
         | awk -F/ '{n=$NF; sub(/_steps_pool.json/, "", n); z=$0; sub(/_pool.json$/, ".zip", z); print n, z}' | sort -n | tail -1 || true)
DONE=0
if [ -n "$latest" ]; then
    DONE=${latest%% *}
    export RESUME_CHECKPOINT=${latest#* }
fi
REMAINING=$((TARGET - DONE))
echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  index $INDEX  seed $SEED  pool $POOL  tag ${RUN_TAG:-none}  invalid reward $ALPHAGEN_INVALID_REWARD  policy $ALPHAGEN_POLICY  obs offset $ALPHAGEN_OBS_OFFSET  done $DONE / $TARGET"
if [ "$REMAINING" -le 0 ]; then
    echo "already finished"
    exit 0
fi

resubmit() {
    echo "time limit near, resubmitting seed $SEED"
    INDEX=$INDEX POOL=$POOL STEPS=${STEPS:-} RUN_TAG=$RUN_TAG ALPHAGEN_INVALID_REWARD=$ALPHAGEN_INVALID_REWARD ALPHAGEN_POLICY=$ALPHAGEN_POLICY ALPHAGEN_OBS_OFFSET=$ALPHAGEN_OBS_OFFSET sbatch --export=ALL --array="$SEED" -J "$SLURM_JOB_NAME" train.sbatch
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" || true
    exit 0
}
trap resubmit USR1

python -u -m scripts.rl --instruments "$INDEX" --pool_capacity "$POOL" --random_seeds "$SEED" --steps "$REMAINING" &
PID=$!
wait "$PID"
