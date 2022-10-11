from pathlib import Path
import enum
import torch
import torch.nn as nn
import MemSE.nn as nnM

ROOT = Path(__file__).parent.parent

DEFAULT_DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class WMAX_MODE(enum.Enum):
	ALL = enum.auto()
	LAYERWISE = enum.auto()
	COLUMNWISE = enum.auto()

UNSUPPORTED_OPS = [
	nn.BatchNorm2d,
	nn.GELU,
	nn.Conv2d,
]

SUPPORTED_OPS = {
	nn.Linear: nnM.linear,
	nn.Softplus: nnM.Softplus,
	nn.AvgPool2d: nnM.avgPool2d,
	nn.AdaptiveAvgPool2d: nnM.avgPool2d,
	#nnM.Conv2DUF: nnM.Conv2DUF.memse,
	nn.ReLU: nnM.ReLU,
	#nnM.Padder: nnM.Padder.memse,
	#nnM.Flattener: nnM.Flattener.memse,
	#nnM.Reshaper: nnM.Reshaper.memse,
}
