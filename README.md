# MemSE

This repository contains a Pytorch implementation of MemSE as discussed in the paper:
- "MemSE: Fast MSE Prediction for Noisy Memristor-Based DNN Accelerators" published in [AICAS2022](https://aicas2022.org)

If this project is useful for you, please cite our work:
```bibtex
@inproceedings{kern2022memse,
 title={MemSE: Fast MSE Prediction for Noisy Memristor-Based DNN Accelerators},
 author={Kern, Jonathan and Henwood, Sebastien and Gonçalo, Mordido and al.},
 booktitle={Proceedings of the IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)},
 pages={TBA},
 year={2022}
}
```

## Usage

### 1. Environment set-up and dependencies

* Python 3.8
* Libraries (Pytorch, Numpy, SciPy, Tensorly and opt_einsum) 

To install from source, run the following commands:

```bash
git clone https://github.com/sebastienwood/MemSE.git
cd MemSE
python setup.py install
```

### 2. Paper experiments
All the papers experiments can be found under `experiments/aicas`.

##  License
  
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
  
The software is for educational and academic research purpose only.