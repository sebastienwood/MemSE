from typing import Callable
import torch.nn as nn
import math
from MemSE.nn import Flattener

__all__ = ['make_JohNet', 'make_small_JohNet']

class JohNet(nn.Module):
    def __init__(self, features, num_classes=1000, classifier_size: int = 16):
        super(JohNet, self).__init__()
        self.features = features
        self.flattener = Flattener()
        self.classifier = nn.Linear(classifier_size, classifier_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.classifier2 = nn.Linear(classifier_size, num_classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.flattener(x)
        x = self.classifier(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier2(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]

        elif isinstance(v,str) and v[0] == 'D':
            layers += [nn.Dropout(float(v[1:]))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d,  nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d,nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)

cfg =[16, 16, 'A', 'D0.1', 32, 32, 'A', 'D0.1', 64, 64, 64, 'A','D0.1', 128,128, 128, 'A','D0.2',  256,256,256,'A','D0.2']
cfg_small =[16, 16, 'A', 'D0.1', 32, 32, 'A', 'D0.1', 32, 32, 32, 'A','D0.1', 32,32, 32, 'A','D0.2',  32,32,32,'A','D0.2']



def make_JohNet(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    cfg_ = kwargs.pop('cfg', cfg)
    model = JohNet(make_layers(cfg_, batch_norm=True), classifier_size=cfg_[-3], **kwargs)
    return model

def make_small_JohNet(**kwargs):
    return make_JohNet(cfg=cfg_small, **kwargs)


if __name__ == '__main__':
    net = make_small_JohNet()
    print(net)