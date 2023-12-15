#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=3000
#SBATCH -c 4
#SBATCH --gres=gpu:v100:4
#SBATCH --output=job_output_%j.txt

# Load Conda and activate the 'pytorch' environment
conda activate pytorch

# Print the configuration of the nodes
nvidia-smi topo -m

# Now run your Python scripts
python mlp_script.py
python comm_script.py
