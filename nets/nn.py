import torch
import torch.nn as nn
from utils.misc import Conv2dAct, InvertedResidual, _make_divisible, _init_weight


class MobileNetV3L(nn.Module):
    """ [https://arxiv.org/abs/1905.02244] """
    def __init__(self, width_mult, num_classes=1000, dropout=0.2, init_weight=True):
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = [16, 16, 24, 24, 40,  40,  40,  80,  80,  80,  80,  112, 112, 160, 160, 160]
        _mid = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
        _out = [16, 24, 24, 40, 40,  40,  80,  80,  80,  80,  112, 112, 160, 160, 160, 960, 1280]

        _mult = lambda x: x * width_mult
        _inp[:] = list(map(_make_divisible, map(_mult, _inp)))
        _mid[:] = list(map(_make_divisible, map(_mult, _mid)))
        _out[:] = list(map(_make_divisible, map(_mult, _out)))

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
        self.features = nn.Sequential(*self._layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=_out[15], out_features=_out[16]),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=_out[16], out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV3S(nn.Module):
    """ [https://arxiv.org/abs/1905.02244] """
    def __init__(self, width_mult, num_classes=1000, dropout=0.2, init_weight=True):
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = [16, 16, 24, 24, 40,  40,  40,  48,  48,  96,  96, 96]
        _mid = [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576]
        _out = [16, 24, 24, 40, 40,  40,  48,  48,  96,  96,  96, 576, 1024]

        _mult = lambda x: x * width_mult
        _inp[:] = list(map(_make_divisible, map(_mult, _inp)))
        _mid[:] = list(map(_make_divisible, map(_mult, _mid)))
        _out[:] = list(map(_make_divisible, map(_mult, _out)))

        self._layers = [Conv2dAct(3, _inp[0], 3, 2, act=nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 2, True, nn.ReLU),  # C1 1/2

            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False, nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False, nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True, nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True, nn.Hardswish),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True, nn.Hardswish),
            InvertedResidual(_inp[6], _mid[6], _out[6], 5, 1, True, nn.Hardswish),
            InvertedResidual(_inp[7], _mid[7], _out[7], 5, 1, True, nn.Hardswish),

            InvertedResidual(_inp[8], _mid[8], _out[8], 5, 2, True, nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[9], _mid[9], _out[9], 5, 1, True, nn.Hardswish),
            InvertedResidual(_inp[10], _mid[10], _out[10], 5, 1, True, nn.Hardswish),

        ])

        self._layers.append(Conv2dAct(_inp[11], _out[11], 1, 2, act=nn.Hardswish))  # C5 1/32
        self.features = nn.Sequential(*self._layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=_out[11], out_features=_out[12]),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=_out[12], out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v3_large(width_mult: float = 1.0, **kwargs) -> MobileNetV3L:
    """ MobileNet V3 Large """
    return MobileNetV3L(width_mult=width_mult, **kwargs)


def mobilenet_v3_small(width_mult: float = 1.0, **kwargs) -> MobileNetV3S:
    """ MobileNet V3 Small """
    return MobileNetV3S(width_mult=width_mult, **kwargs)


if __name__ == '__main__':
    mobilenetv3_large = mobilenet_v3_large()
    mobilenetv3_small = mobilenet_v3_small()

    img = torch.randn(1, 3, 224, 224)
    print(mobilenetv3_large(img).shape)
    print(mobilenetv3_small(img).shape)

    print("Num params. V3 Large: {}".format(sum(p.numel() for p in mobilenetv3_large.parameters() if p.requires_grad)))
    print("Num params. V3 Small: {}".format(sum(p.numel() for p in mobilenetv3_small.parameters() if p.requires_grad)))
