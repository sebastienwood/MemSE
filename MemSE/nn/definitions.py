from .Conv2DUF import Conv2DUF
import torch.nn as nn

TYPES_HANDLED = {
	Conv2DUF: ['weight', 'bias'],
	nn.Linear: ['weight'],
}
