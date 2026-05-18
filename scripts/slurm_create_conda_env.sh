#!/bin/bash -l
#SBATCH --job-name=slurm-create-conda-env
#SBATCH --output=logs/slurm-create-conda-env-%j.log
# Request the number of gpus usint "--gres=gpu:<number>". E.g.:
#SBATCH --gres=gpu:1
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=00:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
# Add host, time, and directory name for later troubleshooting
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
ENV_PATH="$HOME/anaconda3/envs/py311"

echo "Checking for environment at $ENV_PATH"
if ! conda env list | awk '{print $1}' | grep -qx "py311"; then
  echo "Create environment at $ENV_PATH"
  conda create -y -p "$ENV_PATH" python=3.11
else
  echo "Environment already exists at $ENV_PATH"
fi

echo "Activate environment"
conda activate "$ENV_PATH"

echo "Install requirements"
pip install -r /home/balalru/open-unlearning/requirements.txt
pip uninstall -y torch torchvision torchaudio

# PyTorch CUDA 12.6 wheels
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

pip install ".[lm-eval]"
# pip install --no-build-isolation flash-attn==2.6.3

date