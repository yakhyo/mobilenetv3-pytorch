import torch
from torch import nn

from utils.utils import Conv2dNormActivation, SqueezeExcitation, _make_divisible


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, in_channels, kernel, exp_channels, out_channels, use_se, activation, stride, dilation,
                 width_mult):
        super().__init__()
        self.in_channels = _make_divisible(in_channels * width_mult)
        self.exp_channels = _make_divisible(exp_channels * width_mult)
        self.out_channels = _make_divisible(out_channels * width_mult)

        self.use_res_connect = stride == 1 and self.in_channels == self.out_channels

        layers = []
        activation_layer = nn.Hardswish if activation == 'HS' else nn.ReLU

        # expand
        if self.exp_channels != self.in_channels:
            layers.append(Conv2dNormActivation(self.in_channels,
                                               self.exp_channels,
                                               kernel_size=1,
                                               activation_layer=activation_layer)
                          )

        # depthwise
        layers.append(Conv2dNormActivation(self.exp_channels,
                                           self.exp_channels,
                                           kernel_size=kernel,
                                           stride=stride,
                                           dilation=dilation,
                                           groups=self.exp_channels,
                                           activation_layer=activation_layer)
                      )
        if use_se:
            squeeze_channels = _make_divisible(self.exp_channels // 4)
            layers.append(SqueezeExcitation(self.exp_channels, squeeze_channels))

        # project
        layers.append(Conv2dNormActivation(self.exp_channels,
                                           self.out_channels,
                                           kernel_size=1,
                                           activation_layer=None)
                      )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting, last_channel, num_classes=1000, dropout=0.2):
        super().__init__()

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].in_channels
        layers.append(Conv2dNormActivation(3,
                                           firstconv_output_channels,
                                           kernel_size=3,
                                           stride=2,
                                           activation_layer=nn.Hardswish)
                      )

        # building inverted residual blocks
        layers.extend(inverted_residual_setting)


        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels

        layers.append(Conv2dNormActivation(lastconv_input_channels,
                                           lastconv_output_channels,
                                           kernel_size=1,
                                           activation_layer=nn.Hardswish)
                      )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


def _mobilenet_v3_conf(arch: str, width_mult: float = 1.0):
    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            InvertedResidual(16, 3, 16, 16, False, "RE", 1, 1, width_mult),
            InvertedResidual(16, 3, 64, 24, False, "RE", 2, 1, width_mult),  # C1
            InvertedResidual(24, 3, 72, 24, False, "RE", 1, 1, width_mult),
            InvertedResidual(24, 5, 72, 40, True, "RE", 2, 1, width_mult),  # C2
            InvertedResidual(40, 5, 120, 40, True, "RE", 1, 1, width_mult),
            InvertedResidual(40, 5, 120, 40, True, "RE", 1, 1, width_mult),
            InvertedResidual(40, 3, 240, 80, False, "HS", 2, 1, width_mult),  # C3
            InvertedResidual(80, 3, 200, 80, False, "HS", 1, 1, width_mult),
            InvertedResidual(80, 3, 184, 80, False, "HS", 1, 1, width_mult),
            InvertedResidual(80, 3, 184, 80, False, "HS", 1, 1, width_mult),
            InvertedResidual(80, 3, 480, 112, True, "HS", 1, 1, width_mult),
            InvertedResidual(112, 3, 672, 112, True, "HS", 1, 1, width_mult),
            InvertedResidual(112, 5, 672, 160, True, "HS", 2, 1, width_mult),  # C4
            InvertedResidual(160, 5, 960, 160, True, "HS", 1, 1, width_mult),
            InvertedResidual(160, 5, 960, 160, True, "HS", 1, 1, width_mult),
        ]
        last_channel = _make_divisible(1280 * width_mult)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            InvertedResidual(16, 3, 16, 16, True, "RE", 2, 1, width_mult),  # C1
            InvertedResidual(16, 3, 72, 24, False, "RE", 2, 1, width_mult),  # C2
            InvertedResidual(24, 3, 88, 24, False, "RE", 1, 1, width_mult),
            InvertedResidual(24, 5, 96, 40, True, "HS", 2, 1, width_mult),  # C3
            InvertedResidual(40, 5, 240, 40, True, "HS", 1, 1, width_mult),
            InvertedResidual(40, 5, 240, 40, True, "HS", 1, 1, width_mult),
            InvertedResidual(40, 5, 120, 48, True, "HS", 1, 1, width_mult),
            InvertedResidual(48, 5, 144, 48, True, "HS", 1, 1, width_mult),
            InvertedResidual(48, 5, 288, 96, True, "HS", 2, 1, width_mult),  # C4
            InvertedResidual(96, 5, 576, 96, True, "HS", 1, 1, width_mult),
            InvertedResidual(96, 5, 576, 96, True, "HS", 1, 1, width_mult),
        ]
        last_channel = _make_divisible(1024 * width_mult)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(inverted_residual_setting, last_channel):
    return MobileNetV3(inverted_residual_setting, last_channel)


def mobilenet_v3_large() -> MobileNetV3:
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)
    return _mobilenet_v3(inverted_residual_setting, last_channel)


def mobilenet_v3_small() -> MobileNetV3:
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)
    return _mobilenet_v3(inverted_residual_setting, last_channel)


if __name__ == '__main__':
    model = mobilenet_v3_small()
    print("Num params. {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
