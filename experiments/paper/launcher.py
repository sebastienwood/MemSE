from subprocess import run
from pathlib import Path
import requests
import os
import subprocess
import sys

###
# HOWTO
# - provide a experiments.dat file in the same directory as launcher.py
# -- each line is a python cmd with a new experiment
##

# ENSURE DIRS ARE CREATED
launcher_path = Path(__file__).parent.resolve()
os.chdir(launcher_path)
output_path = launcher_path / 'outputs'
output_path.mkdir(exist_ok=True)
install_path = launcher_path / '.installs'
install_path.mkdir(exist_ok=True)

# PREPARE LIBRARIES THAT ARE NOT AVAILABLE IN WHEELHOUSE
to_install = []
noncc = open('noncc_requirements.txt', 'r')
for line in noncc:
    if not list(install_path.glob(f'{line.strip()}-*.tar.gz')):
        print(f'{line.strip()} not found on system')
        to_install.append(line)
if len(to_install) > 0:
    subprocess.check_call([sys.executable, '-m', 'pip', 'download', '--no-binary', ':all:', '--no-deps', '-d', str(install_path)] + to_install)

# ENSURE DATA IS READY
datapath = launcher_path.parent.parent.parent / 'data' 
datapath.mkdir(exist_ok=True)
fname = 'cifar-10-python.tar.gz'
url = 'https://www.cs.toronto.edu/~kriz/' + fname
r = requests.get(url)
open(datapath / fname, 'wb').write(r.content)

# LAUNCH
experiments_path = launcher_path / 'experiments.dat'
lines = []
with open(experiments_path, 'r') as fp:
    idx = 0
    while (line := fp.readline().rstrip()):
        if line[0] == '#':
            continue
        lines.append(str(idx))
        idx += 1
cmd = ['sbatch']
idxs = ','.join(lines)
if len(lines) > 1:
    cmd.extend([f'--array={idxs}']) # always an array 
    cmd.extend([str(launcher_path / 'run_experiment.sh')])
    out = run(cmd, capture_output=True)
    print(out)