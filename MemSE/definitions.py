from pathlib import Path
import enum
import torch

__all__ = ['ROOT', 'WMAX_MODE']

ROOT = Path(__file__).parent.parent

DEFAULT_DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class WMAX_MODE(enum.Enum):
	ALL = enum.auto()
	LAYERWISE = enum.auto()
	COLUMNWISE = enum.auto()
