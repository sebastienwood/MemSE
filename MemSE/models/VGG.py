from typing import Callable
import torch.nn as nn
import math
from MemSE.nn import Flattener

__all__ = ['small_vgg', 'really_small_vgg', 'smallest_vgg','small_vgg_ReLU', 'really_small_vgg_ReLU', 'smallest_vgg_ReLU']

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, classifier_size: int = 16):
        super(VGG, self).__init__()
        self.features = features
        self.flattener = Flattener()
        self.classifier = nn.Linear(classifier_size, num_classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.flattener(x)
        x = self.classifier(x)
        return x

    #forward pass but with only fully connected layers
    def forward_fc(self, x):
        #channels = 64 #TODO: this is architecture specific
        #channels = 16 #TODO: this is architecture specific
        channels = 2 #TODO: this is architecture specific
        width = 32 #TODO: this is architecture specific
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                x = nn.functional.pad(x, (1, 1, 1, 1)) #padding
                x = x.view(x.size(0), -1)
                x = layer(x)
                #x = x.reshape(x.shape[0], min(512, int(channels)), int(width), int(width))
                #x = x.reshape(x.shape[0], min(128, int(channels)), int(width), int(width))
                x = x.reshape(x.shape[0], min(16, int(channels)), int(width), int(width))
                channels*=2 #TODO: this is architecture specific
                width/=2 #TODO: this is architecture specific
            else:
                x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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



def make_layers(cfg, batch_norm=False, activation: Callable = nn.Softplus):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation()]
            else:
                layers += [conv2d, activation()]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_ReLU(cfg, batch_norm=False):
    return make_layers(cfg, batch_norm, nn.ReLU)


cfg = {
    'small_vgg': [64, 'A', 128, 'A', 256, 'A', 512, 'A', 512, 'A'],
    'really_small_vgg': [16, 'A', 32, 'A', 64, 'A', 128, 'A', 128, 'A'],
    'smallest_vgg': [2, 'A', 4, 'A', 8, 'A', 16, 'A', 16, 'A'],
}


def small_vgg(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    method = kwargs.pop('activation', nn.Softplus)
    model = VGG(make_layers(cfg['small_vgg'], batch_norm=False, activation=method), classifier_size=512, **kwargs)
    return model

def really_small_vgg(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    method = kwargs.pop('activation', nn.Softplus)
    model = VGG(make_layers(cfg['really_small_vgg'], batch_norm=False, activation=method), classifier_size=128, **kwargs)
    return model

def smallest_vgg(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    method = kwargs.pop('activation', nn.Softplus)
    model = VGG(make_layers(cfg['smallest_vgg'], batch_norm=False, activation=method), **kwargs)
    return model


def small_vgg_ReLU(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return small_vgg(activation=nn.ReLU, **kwargs)

def really_small_vgg_ReLU(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return really_small_vgg(activation=nn.ReLU, **kwargs)

def smallest_vgg_ReLU(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    features_only = kwargs.pop('features_only', False)
    if features_only:
        model = make_layers_ReLU(cfg['smallest_vgg'], batch_norm=False)
    else:
        model = smallest_vgg(activation=nn.ReLU, **kwargs)
    return model
