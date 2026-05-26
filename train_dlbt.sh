#!/bin/bash

#### SLURM JOB OPTIONS ##############################################
#SBATCH --mail-user=lukas.s.huber@unibe.ch
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=task_gen
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt

#SBATCH --partition=gpu
#SBATCH --account=paygo
#SBATCH --wckey=psy_perception_models
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

#### SETUP ##########################################################
# ensure log dir exists
mkdir -p logs examples/plots

# load conda and activate environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate dlbt

# make PyTorch/OpenMP use all CPUs allocated
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "===== Job $SLURM_JOB_ID starting on $(hostname) ====="
echo "Running on GPU:" $(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "CPUs: $SLURM_CPUS_PER_TASK, Mem/CPU: $SLURM_MEM_PER_CPU"
echo "===== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) starting on $(hostname) ====="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sanity-check: confirm PyTorch can see the GPU before committing to a long run
python -c "
import torch
avail = torch.cuda.is_available()
name  = torch.cuda.get_device_name(0) if avail else 'N/A'
print(f'PyTorch {torch.__version__}  |  CUDA available: {avail}  |  GPU: {name}')
if not avail:
    raise SystemExit('ERROR: CUDA not available — aborting job to avoid silent CPU fallback.')
"

# Run
# python experiments/behavior/run1/021_efficiency_main/run.py
# python experiments/behavior/run1/021_efficiency_main/analysis.py

# python experiments/behavior/run1/05_ablations/run.py
# python experiments/behavior/run1/05_ablations/analysis.py

# python experiments/behavior/run1/06_slda_finetuning/run.py
# python experiments/behavior/run1/06_slda_finetuning/analysis.py

# python experiments/behavior/run1/07_task_generalization/run.py
# python experiments/behavior/run1/07_task_generalization/analysis.py