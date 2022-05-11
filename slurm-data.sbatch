#!/bin/bash

# The config file  config.yaml must be place at the same level than this
# script 

#SBATCH --job-name=super-SAR-data
#SBATCH --nodes=1
#SBATCH --partition=cpu_prod
#SBATCH --time=12:00:00
#SBATCH --output=logslurms/slurm-data-%j.out
#SBATCH --error=logslurms/slurm-data-%j.err

# Load the conda module
export PATH=/opt/conda/bin:$PATH
source activate sondraSAR

cd data
rm -rf data_files/train
python3 dataset_builder.py --path_to_config ../config.yaml
