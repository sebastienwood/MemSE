import torch.nn as nn
import math

__all__ = ['small_vgg', 'really_small_vgg', 'smallest_vgg']

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        #self.classifier = nn.Linear(512, num_classes, bias=False)
        #self.classifier = nn.Linear(128, num_classes, bias=False)
        self.classifier = nn.Linear(16, num_classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Softplus()]
            else:
                layers += [conv2d, nn.Softplus()]
            in_channels = v
    return nn.Sequential(*layers)


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
    model = VGG(make_layers(cfg['small_vgg'], batch_norm=False), **kwargs)
    return model

def really_small_vgg(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['really_small_vgg'], batch_norm=False), **kwargs)
    return model

def smallest_vgg(**kwargs):
    """Small VGG model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['smallest_vgg'], batch_norm=False), **kwargs)
    return model