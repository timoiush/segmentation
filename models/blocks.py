import torch
import torch.nn as nn

def conv_layer(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, dim=2):
    assert dim == 2 or dim == 3, "dim should be 2 or 3"
    if dim == 3:
        return nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
    else:
        return nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)


def up_conv_layer(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, dim=2):
    assert dim == 2 or dim == 3, "dim should be 2 or 3"
    if dim == 3:
        return nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
    else:
        return nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)


def activation_layer(activation='leaky'):
    assert activation in ['relu', 'elu', 'leaky']
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    else:
        return nn.LeakyReLU(0.2, inplace=True)


def normalization_layer(normalization, n_ch, dim):
    assert normalization in ['batch', 'instance']
    if normalization == 'batch':
        if dim == 2:
            return nn.BatchNorm2d(n_ch)
        elif dim == 3:
            return nn.BatchNorm3d(n_ch)
    elif normalization == 'instance':
        if dim == 2:
            return nn.InstanceNorm2d(n_ch)
        elif dim == 3:
            return nn.InstanceNorm3d(n_ch)

        
def pooling_layer(pooling, dim):
    assert pooling in ['max', 'avg']
    if dim == 2:
        return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    elif dim == 3:
        return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation='leaky', normalization=None, dim=2):
        super().__init__()
        self.normalization = normalization
        conv1 = conv_layer(in_ch, out_ch, dim=dim)
        conv2= conv_layer(out_ch, out_ch, dim=dim)
        act = activation_layer(activation)
        layers = []
        if self.normalization:
            norm = normalization_layer(normalization, out_ch, dim)
            layers = [conv1, norm, act, conv2, norm, act]
        else:
            layers = [conv1, act, conv2, act]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation='leaky', normalization=None, dim=2):
        super().__init__()
        self.normalization = normalization
        self.in_ch = in_ch
        self.out_ch = out_ch 
        self.up_conv = up_conv_layer(in_ch, out_ch, kernel_size=4, stride=2, dim=dim)
        conv1 = conv_layer(in_ch, out_ch, dim=dim)
        conv2 = conv_layer(out_ch, out_ch, dim=dim)
        act = activation_layer(activation)
        layers = []
        if self.normalization:
            norm = normalization_layer(normalization, out_ch, dim)
            layers = [conv1, norm, act, conv2, norm, act]
        else:
            layers = [conv1, act, conv2, act]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print(x.shape, self.in_ch, self.out_ch)
        x = self.up_conv(x)
        x = torch.cat((x, skip_input), dim=1)
        x = self.layers(x)
        return x
