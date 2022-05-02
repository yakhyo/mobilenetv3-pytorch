import torch
import torch.nn.functional as F
from copy import deepcopy


def _pad(kernel_size, dilation=1):
    return kernel_size // (2 * dilation)


def _make_divisible(width):
    divisor = 8
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return new_width


def _init_weight(self):
    for m in self.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.zeros_(m.bias)


class Conv2dAct(torch.nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=c1,
                                    out_channels=c2,
                                    kernel_size=k,
                                    stride=s,
                                    padding=_pad(k, d) if p is None else p,
                                    dilation=d,
                                    groups=g,
                                    bias=False)
        self.norm = torch.nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.01)
        self.act = act(inplace=True) if act is not None else torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SqueezeExcitation(torch.nn.Module):
    """ [https://arxiv.org/abs/1709.01507] """

    def __init__(self, c1):
        super().__init__()
        c2 = _make_divisible(c1 // 4)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=c2, out_channels=c1, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.hard = torch.nn.Hardsigmoid()

    def _scale(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hard(x)
        return x

    def forward(self, x):
        return x * self._scale(x)


class InvertedResidual(torch.nn.Module):
    """ [https://arxiv.org/abs/1801.04381] """

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, act):
        super().__init__()
        self._inp = in_channels
        self._mid = mid_channels
        self._out = out_channels
        self._shortcut = stride == 1 and self._inp == self._out

        self._block = torch.nn.Sequential(
            Conv2dAct(self._inp, self._mid, 1, act=act) if self._mid != self._inp else torch.nn.Identity(),
            Conv2dAct(self._mid, self._mid, kernel_size, stride, g=self._mid, act=act),
            SqueezeExcitation(self._mid) if use_se else torch.nn.Identity(),
            Conv2dAct(self._mid, self._out, k=1, act=None),
        )

    def forward(self, x):
        return x + self._block(x) if self._shortcut else self._block(x)


class MobileNetV3L(torch.nn.Module):
    """ [https://arxiv.org/abs/1905.02244] """

    def __init__(self, width_mult, num_classes=1000, dropout=0.2, init_weight=True):
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160]
        _mid = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
        _out = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960, 1280]

        _mult = lambda x: x * width_mult
        _inp[:] = list(map(_make_divisible, map(_mult, _inp)))
        _mid[:] = list(map(_make_divisible, map(_mult, _mid)))
        _out[:] = list(map(_make_divisible, map(_mult, _out)))

        self._layers = [Conv2dAct(3, _inp[0], 3, 2, act=torch.nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 1, False, torch.nn.ReLU),
            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False, torch.nn.ReLU),  # C1 1/2
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False, torch.nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True, torch.nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True, torch.nn.ReLU),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True, torch.nn.ReLU),

            InvertedResidual(_inp[6], _mid[6], _out[6], 3, 2, False, torch.nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[7], _mid[7], _out[7], 3, 1, False, torch.nn.Hardswish),
            InvertedResidual(_inp[8], _mid[8], _out[8], 3, 1, False, torch.nn.Hardswish),
            InvertedResidual(_inp[9], _mid[9], _out[9], 3, 1, False, torch.nn.Hardswish),

            InvertedResidual(_inp[10], _mid[10], _out[10], 3, 1, True, torch.nn.Hardswish),
            InvertedResidual(_inp[11], _mid[11], _out[11], 3, 1, True, torch.nn.Hardswish),

            InvertedResidual(_inp[12], _mid[12], _out[12], 5, 2, True, torch.nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[13], _mid[13], _out[13], 5, 1, True, torch.nn.Hardswish),
            InvertedResidual(_inp[14], _mid[14], _out[14], 5, 1, True, torch.nn.Hardswish),
        ])

        self._layers.append(Conv2dAct(_inp[15], _out[15], act=torch.nn.Hardswish))
        self.features = torch.nn.Sequential(*self._layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=_out[15], out_features=_out[16]),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Dropout(p=dropout, inplace=True),
            torch.nn.Linear(in_features=_out[16], out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV3S(torch.nn.Module):
    """ [https://arxiv.org/abs/1905.02244] """

    def __init__(self, width_mult, num_classes=1000, dropout=0.2, init_weight=True):
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96]
        _mid = [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576]
        _out = [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576, 1024]

        _mult = lambda x: x * width_mult
        _inp[:] = list(map(_make_divisible, map(_mult, _inp)))
        _mid[:] = list(map(_make_divisible, map(_mult, _mid)))
        _out[:] = list(map(_make_divisible, map(_mult, _out)))

        self._layers = [Conv2dAct(3, _inp[0], 3, 2, act=torch.nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 2, True, torch.nn.ReLU),  # C1 1/2

            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False, torch.nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False, torch.nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True, torch.nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True, torch.nn.Hardswish),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True, torch.nn.Hardswish),
            InvertedResidual(_inp[6], _mid[6], _out[6], 5, 1, True, torch.nn.Hardswish),
            InvertedResidual(_inp[7], _mid[7], _out[7], 5, 1, True, torch.nn.Hardswish),

            InvertedResidual(_inp[8], _mid[8], _out[8], 5, 2, True, torch.nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[9], _mid[9], _out[9], 5, 1, True, torch.nn.Hardswish),
            InvertedResidual(_inp[10], _mid[10], _out[10], 5, 1, True, torch.nn.Hardswish),

        ])

        self._layers.append(Conv2dAct(_inp[11], _out[11], 1, 2, act=torch.nn.Hardswish))  # C5 1/32
        self.features = torch.nn.Sequential(*self._layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=_out[11], out_features=_out[12]),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Dropout(p=dropout, inplace=True),
            torch.nn.Linear(in_features=_out[12], out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EMA(torch.nn.Module):
    """ [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage] """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            ema_v = self.model.state_dict().values()
            model_v = model.state_dict().values()
            for e, m in zip(ema_v, model_v):
                e.copy_(update_fn(e, m))

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


class PolyLoss:
    """ [https://arxiv.org/abs/2204.12511?context=cs] """

    def __init__(self, reduction='none', label_smoothing=0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, prediction, target, epsilon=1.0):
        ce = F.cross_entropy(prediction, target, reduction=self.reduction, label_smoothing=self.label_smoothing)
        pt = torch.sum(F.one_hot(target, num_classes=1000) * self.softmax(prediction), dim=-1)
        pl = torch.mean(ce + epsilon * (1 - pt))
        return pl


class CrossEntropyLoss:
    """ [https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html] """

    def __init__(self, reduction='mean', label_smoothing=0.0) -> None:
        super().__init__()

        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(self, prediction, target):
        return F.cross_entropy(prediction, target, reduction=self.reduction, label_smoothing=self.label_smoothing)


def mobilenet_v3_large(width_mult: float = 1.0, **kwargs) -> MobileNetV3L:
    """ MobileNet V3 Large """
    return MobileNetV3L(width_mult=width_mult, **kwargs)


def mobilenet_v3_small(width_mult: float = 1.0, **kwargs) -> MobileNetV3S:
    """ MobileNet V3 Small """
    return MobileNetV3S(width_mult=width_mult, **kwargs)


if __name__ == '__main__':
    v3_large = mobilenet_v3_large()
    v3_small = mobilenet_v3_small()

    img = torch.randn(1, 3, 224, 224)
    print(v3_large(img).shape)
    print(v3_small(img).shape)

    print("Num params. V3 Large: {}".format(sum(p.numel() for p in v3_large.parameters() if p.requires_grad)))
    print("Num params. V3 Small: {}".format(sum(p.numel() for p in v3_small.parameters() if p.requires_grad)))
