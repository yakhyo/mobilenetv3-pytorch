import torch
import torch.nn as nn


class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1,
                 activation_layer=torch.nn.ReLU, dilation=1, inplace=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            self.act = activation_layer(**params)

        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SqueezeExcitation(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c1, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.hard = nn.Hardsigmoid()

    def _scale(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hard(x)
        return x

    def forward(self, x):
        return x * self._scale(x)


def _make_divisible(width):
    divisor = 8
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return new_width
