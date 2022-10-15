import torch

__all__ = ['fuse_conv_bn']

def resnet_layer_fusion_generator(layer_idx, downsample: bool):
    lys = []
    b = f'layer{layer_idx}.'
    for i in range(2):
        for j in range(1, 3):
            lys.append([f'{b}{i}.conv{j}', f'{b}{i}.bn{j}'])
        if downsample and i == 0:
            lys.append([f'{b}{i}.downsample.0', f'{b}{i}.downsample.1'])
    return lys


MODELS_FUSION = {
    'resnet18': [
                  ['conv1', 'bn1'],
                  *resnet_layer_fusion_generator(1, False), # TODO c'est pas joli joli mais c'est plutôt flexible
                  *resnet_layer_fusion_generator(2, True),
                  *resnet_layer_fusion_generator(3, True),
                  *resnet_layer_fusion_generator(4, True)
                ],
    'smallest_vgg': None,
    'make_johnet' : [['features.0', 'features.1'],['features.3', 'features.4'],['features.8', 'features.9'],
                ['features.11', 'features.12'],['features.16', 'features.17'],['features.19', 'features.20'],
                ['features.22', 'features.23'],['features.27', 'features.28'],['features.30', 'features.31'],
                ['features.33', 'features.34'],['features.38', 'features.39'],['features.41', 'features.42'],['features.44', 'features.45']]
}

def fuse_conv_bn(model, model_name: str, model_fusion=MODELS_FUSION):
    #TODO https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50
    modules_to_fuse = MODELS_FUSION.get(model_name.lower(), None)
    if modules_to_fuse is not None:
        return torch.quantization.fuse_modules(model, modules_to_fuse)
    return model