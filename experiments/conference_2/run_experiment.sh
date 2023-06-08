#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=12  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=120G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-1:00     # DD-HH:MM:SS
#SBATCH --mail-user=sebastien.henwood@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/%x-%A-%a.out
#SBATCH --account=rrg-franlp

module load StdEnv/2020 python/3.9 cuda cudnn

u=${u:-sebwood}
echo "User $u on shell $0"
SOURCEDIR=~/projects/def-franlp/$u/MemSE
cd $SOURCEDIR
nvidia-smi

###
# ENV PREPARATION
# First method is to create the venv on the spot, ensuring fresh installation: it fails if the node is offline and some wheels are not cached
# Second is to push a cached venv
###
# python3 -m venv $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# python -m pip install --no-index --upgrade pip setuptools
# python -m pip install --no-index -r $SOURCEDIR/requirements.txt # --find-links "$SOURCEDIR"/experiments/paper/.installs/

cp ~/projects/def-franlp/$u/venv.tar.gz $SLURM_TMPDIR
tar xzf $SLURM_TMPDIR/venv.tar.gz -C $SLURM_TMPDIR
source $SLURM_TMPDIR/.venv/bin/activate

TODAY=$(TZ=":America/Montreal" date)
COMMIT_ID=$(git rev-parse --verify HEAD)
echo "Experiment $SLURM_JOB_ID ($PWD) start $TODAY on node $SLURMD_NODENAME (git commit id $COMMIT_ID)"
python -m torch.utils.collect_env

# Prepare data
datapath=$SLURM_TMPDIR/data
mkdir $datapath
dset=${dataset:-CIFAR10}
if [ "$dset" == "CIFAR10" ]; then
    echo "CIFAR10 selected"
    cp ~/projects/def-franlp/$u/data/cifar-10-python.tar.gz $datapath
    tar xzf $datapath/cifar-10-python.tar.gz -C $datapath
fi

line_to_read=$(($SLURM_ARRAY_TASK_ID))
echo "Line to read = $line_to_read"
SED_RES=$(sed -n "$line_to_read"p "$SOURCEDIR/experiments/conference_2/experiments.dat")
echo "${SED_RES} --datapath ${datapath}"
eval "${SED_RES} --datapath ${datapath}"
echo "Done"
