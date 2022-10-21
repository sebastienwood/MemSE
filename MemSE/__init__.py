from .MemSE import MemSE
from .MemristorQuant import MemristorQuant
from .definitions import *
from .fx import *

METHODS = {
    'unfolded': conv_to_unfolded,
    'fc': conv_to_fc
}