#!/bin/bash
#SBATCH -J ag-install
#SBATCH -A g107-2971
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o install_%j.log
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alphagen
echo "== conda (compiled packages, CentOS 7 compatible builds)"
echo skip conda
echo "== pip torch"
echo skip torch
echo "== pip rest"
pip install -q "setuptools<66" wheel && pip install -q --no-build-isolation gym==0.26.2
pip install -q --prefer-binary --only-binary=numpy,pandas,scipy,matplotlib,pyarrow,torch torch==2.5.1 stable-baselines3==2.8.0 sb3-contrib==2.8.0 "gymnasium<1.3" pyqlib==0.9.7 tensorboard fire openai
echo "== check"
python - <<'PY'
import torch, numpy, pandas, stable_baselines3, sb3_contrib, gymnasium, qlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("numpy", numpy.__version__, "pandas", pandas.__version__, "sb3", stable_baselines3.__version__,
      "sb3c", sb3_contrib.__version__, "gymnasium", gymnasium.__version__, "qlib", qlib.__version__)
x = torch.randn(1000, 1000, device="cuda"); print("matmul ok", (x @ x).sum().item() != 0)
PY
echo "== done"
