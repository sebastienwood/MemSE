from subprocess import run
from pathlib import Path
import requests


# ENSURE DIRS ARE CREATED
Path('./outputs').mkdir(exist_ok=True)

# ENSURE DATA IS READY
Path('../data/').mkdir(exist_ok=True)
fname = 'cifar-10-python.tar.gz'
url = 'https://www.cs.toronto.edu/~kriz/' + fname
r = requests.get(url)
open(Path(f'../data/{fname}'), 'wb').write(r.content)


# LAUNCH
with open(r"experiments.dat", 'r') as fp:
    counted = len(fp.readlines())
cmd = ['sbatch']
if counted > 1:
    cmd.extend([f'--array=0-{counted - 1}'])
else:
    cmd.extend([f'--array=0']) # always an array 
cmd.extend(['run_experiment.sh'])
out = run(cmd, capture_output=True)
print(out)