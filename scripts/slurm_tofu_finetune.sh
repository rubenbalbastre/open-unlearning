#!/bin/bash
#SBATCH --job-name=tofu-qwen-ft
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Request more time using "--time=<hours:mins:secs>". E.g.:
#SBATCH --time=01:30:00
# Request time partition "--partition=<Partition>". E.g.:
#SBATCH --partition=sc-gpu
hostname; pwd; date

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$HOME/anaconda3/envs/py311"

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
echo "Running on node(s): ${SLURM_JOB_NODELIST:-local}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"


models=(
    "Qwen2.5-3B-Instruct"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)



########################################################################################################################
########################################### RETAIN Finetuned TOFU ######################################################
########################################################################################################################

# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)
    
#     for model in "${models[@]}"; do
#         accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#         src/train.py experiment=finetune/tofu/default.yaml \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         data/datasets@data.train=TOFU_QA_retain \
#         data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
#         trainer.args.ddp_find_unused_parameters=true \
#         trainer.args.gradient_checkpointing=true

    
#         python src/eval.py experiment=eval/tofu/default.yaml \
#         forget_split=${forget_split} \
#         holdout_split=${holdout_split} \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
#     done
# done


# ########################################################################################################################
# ########################################### FULL Finetuned TOFU models #################################################
# ########################################################################################################################


for model in "${models[@]}"; do
    accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
    src/train.py experiment=finetune/tofu/default.yaml \
    task_name=tofu_${model}_full \
    model=${model} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true

    # Evaluate the full models on each forget split
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)

        python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_full_${forget_split} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}
    done
done


date