from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor

__all__ = [
    "MobileNetV3",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]


class SqueezeExcitation(torch.nn.Module):
    """This Squeeze-and-Excitation block
    Args:
        in_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
    """

    def __init__(
            self,
            in_channels: int,
            squeeze_channels: int,
    ) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, in_channels, 1)
        self.relu = nn.ReLU()  # `delta` activation
        self.hard = nn.Hardsigmoid()  # `sigma` (aka scale) activation

    def forward(self, x: Tensor) -> Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hard(scale)
        return scale * x


def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(torch.nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional = None,
            groups: int = 1,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: bool = False,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.01)
        ]

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """Inverted Residual block"""

    def __init__(
            self,
            in_channels: int,
            kernel: int,
            exp_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
    ) -> None:
        super().__init__()
        self._shortcut = stride == 1 and in_channels == out_channels

        in_channels = _make_divisible(in_channels)
        exp_channels = _make_divisible(exp_channels)
        out_channels = _make_divisible(out_channels)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if activation == "HS" else nn.ReLU

        # expand
        if exp_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels=in_channels,
                    out_channels=exp_channels,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )

        # depth-wise convolution
        layers.append(
            Conv2dNormActivation(
                in_channels=exp_channels,
                out_channels=exp_channels,
                kernel_size=kernel,
                stride=1 if dilation > 1 else stride,
                dilation=dilation,
                groups=exp_channels,
                activation_layer=activation_layer,
            )
        )
        if use_se:
            squeeze_channels = _make_divisible(exp_channels // 4, 8)
            layers.append(
                SqueezeExcitation(
                    in_channels=exp_channels,
                    squeeze_channels=squeeze_channels
                )
            )

        # project layer
        layers.append(
            Conv2dNormActivation(
                in_channels=exp_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self._shortcut:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[List[int | str | bool]],
            last_channel: int,
            num_classes: int = 1000,
            dropout: float = 0.2,
    ) -> None:
        """MobileNet V3 main class
        Args:
            inverted_residual_setting: network structure
            last_channel: number of channels on the penultimate layer
            num_classes: number of classes
            dropout: dropout probability
        """
        super().__init__()

        # building first layer
        first_conv_out_channels = inverted_residual_setting[0][0]
        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels=3,
                out_channels=first_conv_out_channels,
                kernel_size=3,
                stride=2,
                activation_layer=nn.Hardswish,
            )
        ]

        # building inverted residual blocks
        for params in inverted_residual_setting:
            layers.append(InvertedResidual(*params))

        # building last several layers
        last_conv_in_channels = inverted_residual_setting[-1][3]
        last_conv_out_channels = 6 * last_conv_in_channels
        layers.append(
            Conv2dNormActivation(
                in_channels=last_conv_in_channels,
                out_channels=last_conv_out_channels,
                kernel_size=1,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

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

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _mobilenet_v3(arch: str, **kwargs: Any, ) -> MobileNetV3:
    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            [16, 3, 16, 16, False, "RE", 1, 1],
            [16, 3, 64, 24, False, "RE", 2, 1],  # C1
            [24, 3, 72, 24, False, "RE", 1, 1],
            [24, 5, 72, 40, True, "RE", 2, 1],  # C2
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 3, 240, 80, False, "HS", 2, 1],  # C3
            [80, 3, 200, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 480, 112, True, "HS", 1, 1],
            [112, 3, 672, 112, True, "HS", 1, 1],
            [112, 5, 672, 160, True, "HS", 2, 1],  # C4
            [160, 5, 960, 160, True, "HS", 1, 1],
            [160, 5, 960, 160, True, "HS", 1, 1],
        ]
        last_channel = 1280  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            [16, 3, 16, 16, True, "RE", 2, 1],  # C1
            [16, 3, 72, 24, False, "RE", 2, 1],  # C2
            [24, 3, 88, 24, False, "RE", 1, 1],
            [24, 5, 96, 40, True, "HS", 2, 1],  # C3
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 120, 48, True, "HS", 1, 1],
            [48, 5, 144, 48, True, "HS", 1, 1],
            [48, 5, 288, 96, True, "HS", 2, 1],  # C4
            [96, 5, 576, 96, True, "HS", 1, 1],
            [96, 5, 576, 96, True, "HS", 1, 1],
        ]
        last_channel = 1024  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    return model


def mobilenet_v3_large(**kwargs: Any) -> MobileNetV3:
    return _mobilenet_v3(arch="mobilenet_v3_large", **kwargs)


def mobilenet_v3_small(**kwargs: Any) -> MobileNetV3:
    return _mobilenet_v3(arch="mobilenet_v3_small", **kwargs)
