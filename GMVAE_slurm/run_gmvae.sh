#!/bin/bash

#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --job-name=CNN_Autoencoder
#SBATCH --output=sample-%j.out

echo "== Loading modules! =="
module purge

module load cuda/11.3
#module load python
module load anaconda

echo "Activating deep_learning environment..."
conda activate deep_learning

#Confirm pytorch detects GPUs
python -c "import torch; print(f'GPUs available: {torch.cuda.is_available()}\nNumber of CUDA devices: {torch.cuda.device_count()}\n')"

#more /proc/cpuinfo

echo "== Submitting $SLURM_JOB_ID job! =="
sleep 10

echo "###################################################"
echo "Starting training..."
echo "###################################################"

python /projects/alco4204/CUResearch/psaap_codes/GMVAE/GMVAE_multigpu/train.py --config config_file_gmvae_ddp.yaml --jobid $SLURM_JOB_ID

echo "== End of $SLURM_JOB_ID job =="
