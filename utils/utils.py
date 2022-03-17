import torch


class Conv2dAct(torch.nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=c1,
                                    out_channels=c2,
                                    kernel_size=k,
                                    stride=s,
                                    padding=_pad(k) if p is None else p,
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
    _width_mult = 1.0

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, act):
        super().__init__()
        self._inp = _make_divisible(in_channels * self._width_mult)
        self._mid = _make_divisible(mid_channels * self._width_mult)
        self._out = _make_divisible(out_channels * self._width_mult)
        self._shortcut = stride == 1 and self._inp == self._out

        self._block = torch.nn.Sequential(
            Conv2dAct(self._inp, self._mid, 1, act=act) if self._mid != self._inp else torch.nn.Identity(),
            Conv2dAct(self._mid, self._mid, kernel_size, stride, g=self._mid, act=act),
            SqueezeExcitation(self._mid) if use_se else torch.nn.Identity(),
            Conv2dAct(self._mid, self._out, k=1, act=None),
        )

    def forward(self, x):
        return x + self._block(x) if self._shortcut else self._block(x)


def _make_divisible(width):
    divisor = 8
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return new_width


def _pad(kernel_size, dilation=1):
    return kernel_size // (2 * dilation)


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
