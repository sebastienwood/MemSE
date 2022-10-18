#!/bin/bash
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=12000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-0:05     # DD-HH:MM:SS
#SBATCH --mail-user=sebastien.henwood@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/%x-%j.out
#SBATCH --account=def-franlp

module load StdEnv/2020 python/3.9 cuda cudnn

u=${u:-sebwood}
echo "User $u on shell $0"
SOURCEDIR=~/projects/def-franlp/$u/MemSE
cd ../..
echo "$PWD"

#aSBATCH --gres=gpu:1       # Request GPU "generic resources"
#aSBATCH --cpus-per-task=12  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#aSBATCH --mem=127000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.

###
# ENV PREPARATION
###
python3 -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip pytest pytest-cov torch torchvision
pip install --no-index -r $SOURCEDIR/requirements.txt --find-links "$SOURCEDIR"/experiments/paper/.installs/
datapath=$SLURM_TMPDIR/data
mkdir $datapath
TODAY=$(TZ=":America/Montreal" date)
COMMIT_ID=$(git rev-parse --verify HEAD)
echo "Experiment $SLURM_JOB_ID ($PWD) start $TODAY on node $SLURMD_NODENAME (git commit id $COMMIT_ID)"
nvidia-smi

# Prepare data
datapath=$SLURM_TMPDIR/data
mkdir $datapath
dset=${dataset:-CIFAR10}
if [ "$dset" == "CIFAR10" ]; then
    echo "CIFAR10 selected"
    cp ~/projects/def-franlp/$u/data/cifar-10-python.tar.gz $datapath
    tar xzf $datapath/cifar-10-python.tar.gz -C $datapath
fi

line_to_read=$(($SLURM_ARRAY_TASK_ID+1))
echo "Line to read = $line_to_read"
SED_RES=$(sed -n "$line_to_read"p "/experiments/paper/experiments.dat")
echo "$SED_RES"
eval "$SED_RES"
