from ofa.model_zoo import ofa_net
from MemSE.nn import OFAxMemSE

ofa = ofa_net('ofa_resnet50', False)
# print(ofa)
ofaxmemse = OFAxMemSE(ofa)
ofaxmemse.sample_active_subnet()