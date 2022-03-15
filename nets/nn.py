from functools import partial

import torch
from torch import nn

from utils.utils import Conv2dNormActivation, SqueezeExcitation, _make_divisible


# __all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(self, in_channels, kernel, exp_channels, out_channels, use_se, activation, stride, dilation,
                 width_mult):
        self.input_channels = self.adjust_channels(in_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(exp_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels, width_mult):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, config, norm_layer):
        super().__init__()

        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels

        layers = []
        activation_layer = nn.Hardswish if config.use_hs else nn.ReLU

        # expand
        if config.expanded_channels != config.input_channels:
            layers.append(Conv2dNormActivation(config.input_channels,
                                               config.expanded_channels,
                                               kernel_size=1,
                                               norm_layer=norm_layer,
                                               activation_layer=activation_layer)
                          )

        # depthwise
        layers.append(Conv2dNormActivation(config.expanded_channels,
                                           config.expanded_channels,
                                           kernel_size=config.kernel,
                                           stride=config.stride,
                                           dilation=config.dilation,
                                           groups=config.expanded_channels,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer)
                      )
        if config.use_se:
            squeeze_channels = _make_divisible(config.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(config.expanded_channels, squeeze_channels))

        # project
        layers.append(Conv2dNormActivation(config.expanded_channels,
                                           config.out_channels,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=None)
                      )

        self.block = nn.Sequential(*layers)
        self.out_channels = config.out_channels
        self._is_cn = config.stride > 1

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting, last_channel, num_classes=1000, norm_layer=None, dropout=0.2):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3,
                                           firstconv_output_channels,
                                           kernel_size=3,
                                           stride=2,
                                           norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish)
                      )

        # building inverted residual blocks
        for config in inverted_residual_setting:
            layers.append(InvertedResidual(config, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels

        layers.append(Conv2dNormActivation(lastconv_input_channels,
                                           lastconv_output_channels,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
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
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160, True, "HS", 2, 1),  # C4
            bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
            bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
        ]
        last_channel = adjust_channels(1280)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96, True, "HS", 2, 1),  # C4
            bneck_conf(96, 5, 576, 96, True, "HS", 1, 1),
            bneck_conf(96, 5, 576, 96, True, "HS", 1, 1),
        ]
        last_channel = adjust_channels(1024)  # C5
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
