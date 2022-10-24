import torch
import torch.nn as nn
import numpy as np
from MemSE.train_test_loop import test
from MemSE.utils import count_parameters
from MemSE.model_loader import load_model
from MemSE.dataset import get_dataloader
from MemSE.fx.network_manipulations import conv_to_fc, conv_decomposition

device = torch.device('cpu')

train_loader, valid_loader, test_loader, nclasses, input_shape = get_dataloader('CIFAR10')
criterion = nn.CrossEntropyLoss()
model = load_model('smallest_vgg', nclasses).to(device)
print(count_parameters(model))
_, original_acc = test(test_loader, model, criterion, device)
print(f'Original acc is {original_acc}')
#model = conv_decomposition(model).to(device)
#print(count_parameters(model))
#_, approx_acc = test(test_loader, model, criterion, device)
#print(f'Approx acc is {approx_acc}')
model = conv_to_fc(model, input_shape).to(device)
print(count_parameters(model))
