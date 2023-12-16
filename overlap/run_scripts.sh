#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=30GB
#SBATCH -c 4
#SBATCH --gres=gpu:v100:4
#SBATCH --output=job_output_4gpu_2_%j.txt

module purge
source /scratch/jx2076/miniconda3/bin/activate pytorch

# Print the configuration of the nodes
nvidia-smi topo -m

# Now run your Python scripts
python mlp_script.py
python comm_script.py
