from pathlib import Path
import enum

ROOT = Path(__file__).parent.parent

class WMAX_MODE(enum.Enum):
	ALL = enum.auto()
	LAYERWISE = enum.auto()
	COLUMNWISE = enum.auto()
