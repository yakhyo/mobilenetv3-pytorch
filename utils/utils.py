import torch
import torch.nn as nn


class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1,
                 norm_layer=torch.nn.BatchNorm2d, activation_layer=torch.nn.ReLU, dilation=1, inplace=True, bias=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=bias)
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        self.layers = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.layers(x)


class SqueezeExcitation(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c1, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.hard = nn.Hardsigmoid()

    def _scale(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hard(x)
        return x

    def forward(self, x):
        scale = self._scale(x)
        return scale * x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
