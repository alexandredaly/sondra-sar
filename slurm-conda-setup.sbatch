#!/bin/bash

#SBATCH --job-name=super-SAR-env
#SBATCH --nodes=1
#SBATCH --partition=cpu_prod
#SBATCH --time=1:00:00
#SBATCH --output=logslurms/slurm-conda-%j.out
#SBATCH --error=logslurms/slurm-conda-%j.err

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Create conda environment 
conda create --name sondraSAR python=3.9 --force
source activate sondraSAR
pip install -r requirements.txt
