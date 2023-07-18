from typing import Callable, List, Optional

import torch
from torch import nn, Tensor

__all__ = [
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]


def _make_divisible(v: float, divisor: int) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            activation_layer: Optional[
                Callable[..., torch.nn.Module]] = torch.nn.ReLU
    ) -> None:
        super().__init__()
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
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.01)
        ]

        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        self.out_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SqueezeExcitation(torch.nn.Module):
    """Squeeze-and-Excitation block
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
    """

    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
    ) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.relu = nn.ReLU()
        self.hard = nn.Hardsigmoid()

    def _scale(self, x: Tensor) -> Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return self.hard(scale)

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return scale * x


class InvertedResidual(nn.Module):
    """Inverted Residual block"""

    def __init__(
            self,
            input_channels: int,
            kernel_size: int,
            expanded_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int
    ):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self._shortcut = stride == 1 and input_channels == out_channels
        activation_layer = nn.Hardswish if activation == "HS" else nn.ReLU
        layers: List[nn.Module] = []
        # expand
        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )

        # depth-wise
        layers.append(
            Conv2dNormActivation(
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=expanded_channels,
                activation_layer=activation_layer,
            )
        )
        # squeeze excitation
        if use_se:
            squeeze_channels = _make_divisible(expanded_channels // 4, 8)
            layers.append(
                SqueezeExcitation(expanded_channels, squeeze_channels))

        layers.append(
            Conv2dNormActivation(
                in_channels=expanded_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self._shortcut:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3(nn.Module):
    """MobileNet V3 main class"""

    def __init__(
            self,
            cnf: List[List[int | bool | str]],
            last_channel: int,
            num_classes: int = 1000,
            dropout: float = 0.2,
    ) -> None:

        super().__init__()
        # building first layer
        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                activation_layer=nn.Hardswish,
            )
        ]

        # building inverted residual blocks
        for args in cnf:
            layers.append(InvertedResidual(*args))

        last_in_channels = cnf[-1][0]
        last_out_channels = cnf[-1][2]
        # building last several layers
        layers.append(
            Conv2dNormActivation(
                in_channels=last_in_channels,
                out_channels=last_out_channels,
                kernel_size=1,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=last_out_channels, out_features=last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=last_channel, out_features=num_classes),
        )

        # initialize weights
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


def _mobilenet_v3_conf(arch: str):
    if arch == "mobilenet_v3_large":
        inverted_residual_settings = [
            [16, 3, 16, 16, False, "RE", 1, 1],
            [16, 3, 64, 24, False, "RE", 2, 1],
            # C1

            [24, 3, 72, 24, False, "RE", 1, 1],
            [24, 5, 72, 40, True, "RE", 2, 1],
            # C2

            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 3, 240, 80, False, "HS", 2, 1],
            # C3

            [80, 3, 200, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 480, 112, True, "HS", 1, 1],
            [112, 3, 672, 112, True, "HS", 1, 1],
            [112, 5, 672, 160, True, "HS", 2, 1],
            # C4

            [160, 5, 960, 160, True, "HS", 1, 1],
            [160, 5, 960, 160, True, "HS", 1, 1]
        ]
        last_channel = 1280
    elif arch == "mobilenet_v3_small":
        inverted_residual_settings = [
            [16, 3, 16, 16, True, "RE", 2, 1],
            # C1
            [16, 3, 72, 24, False, "RE", 2, 1],
            # C2
            [24, 3, 88, 24, False, "RE", 1, 1],
            [24, 5, 96, 40, True, "HS", 2, 1],
            # C3
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 120, 48, True, "HS", 1, 1],
            [48, 5, 144, 48, True, "HS", 1, 1],
            [48, 5, 288, 96, True, "HS", 2, 1],
            # C4
            [96, 5, 576, 96, True, "HS", 1, 1],
            [96, 5, 576, 96, True, "HS", 1, 1],
        ]
        last_channel = 1024
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_settings, last_channel


def mobilenet_v3_large(**kwargs) -> MobileNetV3:
    inverted_residual_settings, lc = _mobilenet_v3_conf("mobilenet_v3_large")
    return MobileNetV3(inverted_residual_settings, lc, **kwargs)


def mobilenet_v3_small(**kwargs) -> MobileNetV3:
    inverted_residual_settings, lc = _mobilenet_v3_conf("mobilenet_v3_small")
    return MobileNetV3(inverted_residual_settings, lc, **kwargs)


if __name__ == '__main__':
    v3_large = mobilenet_v3_large()
    v3_small = mobilenet_v3_small()

    print("Number of parameters of MobileNet V3 Large: {}".format(
        sum(p.numel() for p in v3_large.parameters() if p.requires_grad)))
    print("Number of parameters of MobileNet V3 Small: {}".format(
        sum(p.numel() for p in v3_small.parameters() if p.requires_grad)))
