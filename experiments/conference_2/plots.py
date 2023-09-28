import matplotlib.pyplot as plt
import json
from pathlib import Path

this_file = Path(__file__).parent

results = json.load(open(this_file / 'results/src/128.dict'))

def parse_key(key):
    return json.loads(key)


identique = {}
x, y = [], []

for k, v in results.items():
    k = parse_key(k)
    k.pop('gmax')
    k_n = json.dumps(k)
    if k_n not in identique:
        identique[k_n] = []
    identique[k_n].append(v)
    if v['top1'] > 50:
        x.append(v['power'])
        y.append(v['top1'])

fig = plt.subplot()
fig.scatter(x, y)

k = list(identique.values())
for i in range(5):
    fig.scatter([v['power'] for v in k[i]], [v['top1'] for v in k[i]])
fig.set_xscale('log')
fig.figure.savefig(this_file / '128.png')
