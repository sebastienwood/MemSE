#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=12  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=120G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=3-1:00     # DD-HH:MM:SS
#SBATCH --mail-user=sebastien.henwood@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/%x-%A-%a.out
#SBATCH --account=def-franlp

module load StdEnv/2020 python/3.9 cuda cudnn

u=${u:-sebwood}
echo "User $u on shell $0"
SOURCEDIR=~/projects/def-franlp/$u/MemSE
cd ../..
echo "$SOURCEDIR"
echo "$PWD"
nvidia-smi

#aSBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#aSBATCH --mem=12000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
# python experiments/paper/opt_profile.py --network make_JohNet
# python experiments/paper/opt.py --memscale --power-budget 550000
# python experiments/paper/opt.py --memscale --power-budget 600000
# python experiments/paper/opt.py --memscale --power-budget 750000
# python experiments/paper/opt.py --memscale --power-budget 1000000
# python experiments/paper/opt.py --memscale --power-budget 10000000

###
# ENV PREPARATION
###
python3 -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip setuptools
python -m pip install --no-index -r $SOURCEDIR/requirements.txt # --find-links "$SOURCEDIR"/experiments/paper/.installs/
datapath=$SLURM_TMPDIR/data
mkdir $datapath
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

line_to_read=$(($SLURM_ARRAY_TASK_ID+1))
echo "Line to read = $line_to_read"
SED_RES=$(sed -n "$line_to_read"p "$SOURCEDIR/experiments/paper/experiments.dat")
echo "${SED_RES} --datapath ${datapath}"
eval "${SED_RES} --datapath ${datapath}"
echo "Done"
