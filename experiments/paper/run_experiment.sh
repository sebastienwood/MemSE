#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=12  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=127000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-24:00     # DD-HH:MM:SS
#SBATCH --mail-user=sebastien.henwood@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/%x-%j.out
#SBATCH --account=rrg-franlp

module load StdEnv/2020 python/3.9 cuda cudnn

echo "User $u"
u=${u:-sebwood}
SOURCEDIR=~/projects/def-franlp/sebwood/MemSE

###
# ENV PREPARATION
###
python3 -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r $SOURCEDIR/requirements.txt
datapath=$SLURM_TMPDIR/data
mkdir $datapath
TODAY=$(TZ=":America/Montreal" date)
COMMIT_ID=$(git rev-parse --verify HEAD)
echo "Experiment $SLURM_JOB_ID ($PWD) start $TODAY on node $SLURMD_NODENAME ($COMMIT_ID)"
nvidia-smi

# Prepare data
datapath=$SLURM_TMPDIR/data
mkdir $datapath
dset=${dataset:-CIFAR10}
if ["$dset" == "CIFAR10"]; then
    echo "CIFAR10 selected"
    cp ~/projects/def-franlp/$u/data/cifar-10-python.tar.gz $datapath
    tar xzf $datapath/cifar-10-python.tar.gz -C $datapath
fi

line_to_read=$(($SLURM_ARRAY_TASK_ID+1))
echo "$line_to_read"
SED_RES=$(sed -n "$line_to_read"p "$SOURCE_DIR/paper/experiments.dat")
echo "$SED_RES"
eval "$SED_RES"
CMD_EXIT_CODE=$?
