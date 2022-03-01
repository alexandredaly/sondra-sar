#!/usr/bin/python

import os
import subprocess


def makejob(commit_id,config_name):
    return f"""#!/bin/bash

#SBATCH --job-name=Sondra-{config_name}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=01:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err

current_dir=`pwd`

echo "Session " {config_name}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Copying the source directory and data"
date
mkdir $TMPDIR/sondra-sar
rsync -r . $TMPDIR/sondra-sar/

cd $TMPDIR/sondra-sar/

echo "Installing Virtual Env with the requirements"
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
python -m pip install -r requirements.txt
echo "Installation finished"


echo "Training"
python3 train.py --path_to_config {config_name}

# Once the job is finished, you can copy back back
# files from $TMPDIR/sondra-sar to $current_dir

if [[ $? != 0 ]]; then
    exit -1
fi

"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")

if __name__ == "__main__":
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


    # Ensure the log directory exists
    os.system("mkdir -p logslurms")

    submit_job(makejob(commit_id,"./config.yaml"))
