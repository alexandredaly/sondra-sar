#!/bin/bash

# The config file  config.yaml must be place at the same level than this
# script 

#SBATCH --job-name=super-SAR
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-data-%j.out
#SBATCH --error=logslurms/slurm-data-%j.err

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

current_dir=`pwd`

date
echo "Setting up the virtual environment"
python3 -m pip install virtualenv --user

virtualenv -p python3 $TMPDIR/venv
source $TMPDIR/venv/bin/activate
python -m pip install -r requirements.txt

cd data
python3 dataset_builder.py --path_to_config ../config.yaml
