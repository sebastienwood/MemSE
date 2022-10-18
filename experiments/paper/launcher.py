from subprocess import run
from pathlib import Path
import requests
import os


# ENSURE DIRS ARE CREATED
launcher_path = Path(__file__).parent.resolve()
os.chdir(launcher_path)
output_path = launcher_path / 'outputs'
output_path.mkdir(exist_ok=True)

# ENSURE DATA IS READY
datapath = launcher_path.parent.parent.parent / 'data' 
datapath.mkdir(exist_ok=True)
fname = 'cifar-10-python.tar.gz'
url = 'https://www.cs.toronto.edu/~kriz/' + fname
r = requests.get(url)
open(datapath / fname, 'wb').write(r.content)

# LAUNCH
experiments_path = launcher_path / 'experiments.dat'
with open(experiments_path, 'r') as fp:
    counted = len(fp.readlines())
cmd = ['sbatch']
if counted > 1:
    cmd.extend([f'--array=0-{counted - 1}'])
else:
    cmd.extend([f'--array=0']) # always an array 
cmd.extend([str(launcher_path / 'run_experiment.sh')])
out = run(cmd, capture_output=True)
print(out)