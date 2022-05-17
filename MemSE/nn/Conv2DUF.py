import torch
import torch.nn as nn

class Conv2DUF(nn.Module):
    def __init__(self, conv, input_shape, output_shape):
        super().__init__()
        self.c = conv
        self.output_shape = output_shape
        #exemplar = self.unfold_input(torch.rand(input_shape))
        self.weight = conv.weight.detach().clone().view(conv.weight.size(0), -1).t()#.unsqueeze_(0)
        #self.weight = torch.repeat_interleave(self.weight, exemplar.shape[-1], dim=0)
        self.bias = conv.bias.detach().clone()

    def forward(self, x):
        inp_unf = self.unfold_input(x)
        #out_unf = torch.einsum('bfp,pfc->bcp', inp_unf, self.weight)
        out_unf = inp_unf.transpose(1, 2).matmul(self.weight).transpose(1, 2)
        out = out_unf.view(*self.output_shape)
        if self.bias is not None:
            out += self.bias[:, None, None]
        return out

    def unfold_input(self, x):
        return nn.functional.unfold(x, self.c.kernel_size, self.c.dilation, self.c.padding, self.c.stride)

    @staticmethod
    def memse(conv2duf, memse_dict):
        mu = conv2duf(memse_dict['mu']) * memse_dict['r']

        c = conv2duf.weight.learnt_Gmax / conv2duf.weight.Wmax
        #gamma = 

    @staticmethod
    def mse_var(conv2duf, memse_dict, c):
        #gamma ... is the diagonal elements of memse_dict gamma in shape of mu
        #first_comp = (gamma + memse_dict['mu'] ** 2 + ) * memse_dict['sigma'] ** 2 / c ** 2
        #first_comp = nn.functional.conv2d(first_comp, torch.ones_like(conv2duf.weight), stride=conv2duf.stride, padding=conv2duf.padding, dilation=conv2duf.dilation, groups=conv2duf.groups)
        #conv_sq = conv2duf.weight ** 2
        #first_comp += nn.functional.conv2d(gamma, conv_sq, stride=conv2duf.stride, padding=conv2duf.padding, dilation=conv2duf.dilation, groups=conv2duf.groups)
        pass
        #second_comp = 
        # TODO find a way to convolve w with itself with gamma