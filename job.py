#!/usr/bin/python

import os
import sys
import subprocess
import pathlib


SAVE_MODEL_DIR = pathlib.Path("./trained_models").resolve()


def makejob(commit_id, config_path):
    return f"""#!/bin/bash 

#SBATCH --job-name=super-SAR
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=12:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err
#SBATCH --exclude=sh00,sh[10-16]

# Load the conda module
export PATH=/opt/conda/bin:$PATH

current_dir=`pwd`

echo "Session with job_id ${{SLURM_JOBID}} and hostname $(hostname)"

date
echo "Copying the source directory and data"
mkdir $TMPDIR/SAR
rsync -r . $TMPDIR/SAR/ --exclude ./logslurms/ --exclude trained_models --exclude bibiography  --exclude venv

date
echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/SAR/
git checkout {commit_id}

date
echo "Setting up the virtual environment"
source activate sondraSAR


python -m pip list

date
echo "Training"
cd src/
python train.py --path_to_config {config_path} --runid $SLURM_JOBID

if [[ $? != 0 ]]; then
    exit -1
fi

date
echo "Clean up"
rm -f {config_path}


"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} path_to_config.yaml")
    sys.exit(-1)

# Ensure the log directory exists
os.system("mkdir -p logslurms")
os.system("mkdir -p tmpconfig")


def generate_job(loss, model):
    if model == "PixelShuffle":
        batch_size = 128
    elif model in ["SRCNN", "SRCNN2"]:
        batch_size = 32
    elif model == "SwinTransformer":
        batch_size = 2

    config_idx = 0
    while True:
        config_path = pathlib.Path("./tmpconfig") / f"config-{config_idx}.yaml"
        config_path = config_path.resolve()
        if not config_path.exists():
            break
        config_idx += 1

    with open(f"{sys.argv[1]}") as f:
        content = f.read()
        content = content.replace("@SAVE_MODEL_DIR@", str(SAVE_MODEL_DIR))
        content = content.replace("@BATCH_SIZE@", str(batch_size))
        content = content.replace("@LOSS@", str(loss))
        content = content.replace("@MODEL@", str(model))
    print(f"Writting into {config_path} for {loss} and {model}")
    with open(config_path, "w") as f:
        f.write(content)
    print(f"Config file saved as {config_path}")

    # Launch the batch jobs
    submit_job(makejob(commit_id, config_path))


# models = ["SRCNN", "SRCNN2", "PixelShuffle", "SwinTransformer"]
# losses = ["l1", "l2", "SSIM"]
models = ["SRCNN", "SRCNN2", "PixelShuffle", "SwinTransformer"]
losses = ["SSIM"]

for model in models:
    for loss in losses:
        generate_job(loss, model)
