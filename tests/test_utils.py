from MemSE.utils import n_vars_computation
from MemSE.test_utils import MODELS

def test_nvar():
    n_vars_computation(model=MODELS['fc'])