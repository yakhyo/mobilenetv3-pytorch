import torch
import torch.nn as nn
from utils.utils import Conv2dAct, InvertedResidual, _make_divisible, _init_weight


class MobileNetV3L(nn.Module):
    _width = 1.0

    def __init__(self, num_classes=1000, dropout=0.2, init_weight=True):
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160]
        _mid = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
        _out = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960, 1280]
        _last_io = _make_divisible(_out[-1] * self._width)

        self._layers = [Conv2dAct(3, _inp[0], 3, 2, act=nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 1, False, nn.ReLU),
            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False, nn.ReLU),  # C1 1/2
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False, nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True, nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True, nn.ReLU),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True, nn.ReLU),

            InvertedResidual(_inp[6], _mid[6], _out[6], 3, 2, False, nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[7], _mid[7], _out[7], 3, 1, False, nn.Hardswish),
            InvertedResidual(_inp[8], _mid[8], _out[8], 3, 1, False, nn.Hardswish),
            InvertedResidual(_inp[9], _mid[9], _out[9], 3, 1, False, nn.Hardswish),

            InvertedResidual(_inp[10], _mid[10], _out[10], 3, 1, True, nn.Hardswish),
            InvertedResidual(_inp[11], _mid[11], _out[11], 3, 1, True, nn.Hardswish),

            InvertedResidual(_inp[12], _mid[12], _out[12], 5, 2, True, nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[13], _mid[13], _out[13], 5, 1, True, nn.Hardswish),
            InvertedResidual(_inp[14], _mid[14], _out[14], 5, 1, True, nn.Hardswish),
        ])

        self._layers.append(Conv2dAct(_inp[15], _out[15], act=nn.Hardswish))
        self._features = nn.Sequential(*self._layers)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._classifier = nn.Sequential(
            nn.Linear(in_features=_out[15], out_features=_last_io),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=_last_io, out_features=num_classes),
        )

    def features(self, x):
        return self._features(x)

    def forward(self, x):
        x = self._features(x)

        x = self._pool(x)
        x = torch.flatten(x, 1)

        x = self._classifier(x)

        return x


class MobileNetV3S(nn.Module):
    _width = 1.0

    def __init__(self, num_classes=1000, dropout=0.2, init_weight=True):
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = []
        _mid = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
        _out = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960, 1280]
        _last_io = _make_divisible(_out[-1] * self._width)

        self._layers = [Conv2dAct(3, _inp[0], 3, 2, act=nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 1, False, nn.ReLU),
            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False, nn.ReLU),  # C1 1/2
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False, nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True, nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True, nn.ReLU),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True, nn.ReLU),

            InvertedResidual(_inp[6], _mid[6], _out[6], 3, 2, False, nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[7], _mid[7], _out[7], 3, 1, False, nn.Hardswish),
            InvertedResidual(_inp[8], _mid[8], _out[8], 3, 1, False, nn.Hardswish),
            InvertedResidual(_inp[9], _mid[9], _out[9], 3, 1, False, nn.Hardswish),

            InvertedResidual(_inp[10], _mid[10], _out[10], 3, 1, True, nn.Hardswish),
            InvertedResidual(_inp[11], _mid[11], _out[11], 3, 1, True, nn.Hardswish),

            InvertedResidual(_inp[12], _mid[12], _out[12], 5, 2, True, nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[13], _mid[13], _out[13], 5, 1, True, nn.Hardswish),
            InvertedResidual(_inp[14], _mid[14], _out[14], 5, 1, True, nn.Hardswish),
        ])

        self._layers.append(Conv2dAct(_inp[15], _out[15], act=nn.Hardswish))
        self._features = nn.Sequential(*self._layers)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._classifier = nn.Sequential(
            nn.Linear(in_features=_out[15], out_features=_last_io),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=_last_io, out_features=num_classes),
        )

    def features(self, x):
        return self._features(x)

    def forward(self, x):
        x = self._features(x)

        x = self._pool(x)
        x = torch.flatten(x, 1)

        x = self._classifier(x)

        return x


#     elif arch == "mobilenet_v3_small":
#         inverted_residual_setting = [
#             InvertedResidual(16, 16, 16, 3, True, "RE", 2),  # C1
#
#             InvertedResidual(16, 72, 24, 3, False, "RE", 2),  # C2
#             InvertedResidual(24, 88, 24, 3, False, "RE", 1),
#
#             InvertedResidual(24, 96, 40, 5, True, "HS", 2),  # C3
#             InvertedResidual(40, 240, 40, 5, True, "HS", 1),
#             InvertedResidual(40, 240, 40, 5, True, "HS", 1),
#             InvertedResidual(40, 120, 48, 5, True, "HS", 1),
#             InvertedResidual(48, 144, 48, 5, True, "HS", 1),
#
#             InvertedResidual(48, 288, 96, 5, True, "HS", 2),  # C4
#             InvertedResidual(96, 576, 96, 5, True, "HS", 1),
#             InvertedResidual(96, 576, 96, 5, True, "HS", 1),
#         ]
#         last_channel = _make_divisible(1024 * width_mult)  # C5


def mobilenet_v3_large(**kwargs) -> MobileNetV3L:
    arch = "mobilenet_v3_large"
    return MobileNetV3L(**kwargs)


def mobilenet_v3_small(**kwargs) -> MobileNetV3L:
    arch = "mobilenet_v3_small"
    return MobileNetV3L(**kwargs)


if __name__ == '__main__':
    model = mobilenet_v3_large()
    img = torch.randn(1, 3, 224, 224)
    print(model(img).shape)
    print("Num params. {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
