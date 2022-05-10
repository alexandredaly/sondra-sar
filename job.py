#!/usr/bin/python

import os
import sys
import subprocess


def makejob(commit_id):
    return f"""#!/bin/bash 

#SBATCH --job-name=super-SAR
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

current_dir=`pwd`

echo "Session with job_id ${{SLURM_JOBID}}"

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
python3 -m pip install virtualenv --user

virtualenv -p python3 venv
source venv/bin/activate
python -m pip install -r requirements.txt

date
echo "Training"
cd src/
python train.py --path_to_config ../tmpconfig/config-{commit_id}.yaml --runid $SLURM_JOBID

if [[ $? != 0 ]]; then
    exit -1
fi

date
echo "Clean up"
rm -f $current_dir/tmpconfig/config-{commit_id}.yaml


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

os.system(f"cp {sys.argv[1]} tmpconfig/config-{commit_id}.yaml")

# Launch the batch jobs
submit_job(makejob(commit_id))
