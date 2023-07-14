from copy import deepcopy

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from utils.general import _make_divisible, round_filters
from typing import Optional, Callable, List


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


class Conv2dNormActivation(nn.Module):
    """Standard Convolutional Block
    Consists of Convolutional, Normalization, Activation Layers
    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size
        stride: stride size
        padding: padding size
        dilation: dilation rate
        groups: number of groups
        activation: activation function
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            dilation: int = 1,
            groups: int = 1,
            activation: Optional[Callable[..., torch.nn.Module]] = None
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // (2 * dilation)
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                eps=0.001,
                momentum=0.01
            )
        ]

        if activation is not None:
            layers.append(activation(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SqueezeExcitation(torch.nn.Module):
    """ [https://arxiv.org/abs/1709.01507] """

    def __init__(self, in_channels):
        super().__init__()
        squeeze_channels = _make_divisible(in_channels // 4)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        # activation
        self.relu = nn.ReLU()
        # scale activation
        self.hard = nn.Hardsigmoid()

    def _scale(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hard(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return scale * x


class InvertedResidual(torch.nn.Module):
    """Inverted Residual Block
    Args:
        in_channels:
        mid_channels:
        out_channels:
        kernel_size:
        stride:
        use_se:
        activation:
    """

    def __init__(
            self,
            in_channels: int,
            mid_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            use_se: bool,
            activation: Callable[..., nn.Module]
    ) -> None:
        super().__init__()
        self._shortcut = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if mid_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=1,
                    activation=activation
                )
            )

        layers.append(
            Conv2dNormActivation(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=mid_channels,
                activation=activation
            )
        )

        if use_se:

            layers.append(SqueezeExcitation(mid_channels))

        layers.append(
            Conv2dNormActivation(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self._shortcut:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3L(torch.nn.Module):
    """ [https://arxiv.org/abs/1905.02244] """

    def __init__(
            self,
            width_mult: float,
            num_classes: int = 1000,
            dropout: float = 0.2,
            init_weight: bool = True
    ) -> None:
        super().__init__()
        if init_weight:
            _init_weight(self)

        _inp = [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160,
                160,
                160]
        _mid = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672,
                960, 960]
        _out = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160,
                160,
                960, 1280]

        _inp = [round_filters(in_channels, width_mult) for in_channels in
                _inp]
        _mid = [round_filters(mid_channels, width_mult) for mid_channels in
                _mid]
        _out = [round_filters(out_channels, width_mult) for out_channels in
                _out]

        self._layers = [
            Conv2dNormActivation(3, _inp[0], 3, 2,
                                 activation=torch.nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 1, False,
                             torch.nn.ReLU),
            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False,
                             torch.nn.ReLU),  # C1 1/2
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False,
                             torch.nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True,
                             torch.nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True,
                             torch.nn.ReLU),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True,
                             torch.nn.ReLU),

            InvertedResidual(_inp[6], _mid[6], _out[6], 3, 2, False,
                             torch.nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[7], _mid[7], _out[7], 3, 1, False,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[8], _mid[8], _out[8], 3, 1, False,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[9], _mid[9], _out[9], 3, 1, False,
                             torch.nn.Hardswish),

            InvertedResidual(_inp[10], _mid[10], _out[10], 3, 1, True,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[11], _mid[11], _out[11], 3, 1, True,
                             torch.nn.Hardswish),

            InvertedResidual(_inp[12], _mid[12], _out[12], 5, 2, True,
                             torch.nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[13], _mid[13], _out[13], 5, 1, True,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[14], _mid[14], _out[14], 5, 1, True,
                             torch.nn.Hardswish),

        ])

        self._layers.append(
            Conv2dNormActivation(_inp[15], _out[15],
                                 activation=torch.nn.Hardswish))
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

    def __init__(self, width_mult, num_classes=1000, dropout=0.2,
                 init_weight=True):
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

        self._layers = [
            Conv2dNormActivation(3, _inp[0], 3, 2,
                                 activation=torch.nn.Hardswish)]
        self._layers.extend([
            InvertedResidual(_inp[0], _mid[0], _out[0], 3, 2, True,
                             torch.nn.ReLU),  # C1 1/2

            InvertedResidual(_inp[1], _mid[1], _out[1], 3, 2, False,
                             torch.nn.ReLU),  # C2 1/4
            InvertedResidual(_inp[2], _mid[2], _out[2], 3, 1, False,
                             torch.nn.ReLU),

            InvertedResidual(_inp[3], _mid[3], _out[3], 5, 2, True,
                             torch.nn.Hardswish),  # C3 1/8
            InvertedResidual(_inp[4], _mid[4], _out[4], 5, 1, True,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[5], _mid[5], _out[5], 5, 1, True,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[6], _mid[6], _out[6], 5, 1, True,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[7], _mid[7], _out[7], 5, 1, True,
                             torch.nn.Hardswish),

            InvertedResidual(_inp[8], _mid[8], _out[8], 5, 2, True,
                             torch.nn.Hardswish),  # C4 1/16
            InvertedResidual(_inp[9], _mid[9], _out[9], 5, 1, True,
                             torch.nn.Hardswish),
            InvertedResidual(_inp[10], _mid[10], _out[10], 5, 1, True,
                             torch.nn.Hardswish),

        ])

        self._layers.append(Conv2dNormActivation(_inp[11], _out[11], 1, 2,
                                                 activation=torch.nn.Hardswish))  # C5 1/32
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
    """Tensorflow Implementation of Exponential Moving Average"""

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
        self._update(model, update_fn=lambda e, m: self.decay * e + (
                1. - self.decay) * m)


class PolyLoss:
    """ [https://arxiv.org/abs/2204.12511?context=cs] """

    def __init__(self, reduction='none', label_smoothing=0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, prediction, target, epsilon=1.0):
        ce = F.cross_entropy(prediction, target, reduction=self.reduction,
                             label_smoothing=self.label_smoothing)
        pt = torch.sum(
            F.one_hot(target, num_classes=1000) * self.softmax(prediction),
            dim=-1)
        pl = torch.mean(ce + epsilon * (1 - pt))
        return pl


class CrossEntropyLoss:
    """Cross Entropy Loss"""

    def __init__(self, reduction='mean', label_smoothing=0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            prediction,
            target,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


class RMSprop(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-2,
            alpha=0.9,
            eps=1e-7,
            weight_decay=0,
            momentum=0.,
            centered=False,
            decoupled_decay=False,
            lr_in_momentum=True
    ) -> None:

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        decoupled_decay=decoupled_decay,
                        lr_in_momentum=lr_in_momentum)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(
                        p.data)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if 'decoupled_decay' in group and group[
                        'decoupled_decay']:
                        p.data.add_(p.data, alpha=-group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg,
                                alpha=one_minus_alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(
                        group['eps']).sqrt_()
                else:
                    avg = square_avg.add(group['eps']).sqrt_()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in group and group[
                        'lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg,
                                                             value=group[
                                                                 'lr'])
                        p.data.add_(-buf)
                    else:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss


class StepLR:

    def __init__(self, optimizer, step_size, gamma=1., warmup_epochs=0,
                 warmup_lr_init=0):

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lr_values = [group['initial_lr'] for group in
                               self.optimizer.param_groups]
        self.update_groups(self.base_lr_values)

        if self.warmup_epochs:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_epochs
                                 for v
                                 in self.base_lr_values]
            self.update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_lr_values]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if
                key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in
                      self.warmup_steps]
        else:
            values = [base_lr * (self.gamma ** (epoch // self.step_size))
                      for
                      base_lr in self.base_lr_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


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

    print("Num params. V3 Large: {}".format(
        sum(p.numel() for p in v3_large.parameters() if p.requires_grad)))
    print("Num params. V3 Small: {}".format(
        sum(p.numel() for p in v3_small.parameters() if p.requires_grad)))
