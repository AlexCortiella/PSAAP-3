#!/bin/bash

#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:$1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=CNN_Autoencoder
#SBATCH --output=sample-%j.out

echo "== Loading modules! =="
module purge

module load cuda/11.3
module load python
module load anaconda

conda activate deep_learning

conda list

#Confirm pytorch detects GPUs
python -c "import torch; print(f'GPUs available: {torch.cuda.is_available()}\nNumber of CUDA devices: {torch.cuda.device_count()}\n')"

echo "== Submitting job! =="
sleep 10
python main_CNNAE.py
echo "== End of Job =="
