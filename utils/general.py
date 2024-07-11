import os
from copy import deepcopy

from torch import nn
from torch.nn import functional as F

import torch
import torch.distributed as distributed


def reduce_tensor(tensor, n):
    """Getting the average of tensors over multiple GPU devices
    Args:
        tensor: input tensor
        n: world size (number of gpus)
    Returns:
        reduced tensor
    """
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= n
    return rt


def add_weight_decay(model, weight_decay=1e-5):
    # Applying weight decay to only weights, not biases
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


""" 
The order of `def setup_for_distributed()` and 
`def init_distributed_mode()` must be kept
"""


def setup_for_distributed(is_master):
    # This function disables printing when not in master process
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # Initializing distributed mode
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.distributed:
        print(f"| distributed init (rank {args.local_rank}): env://", flush=True)
    else:
        print("Warning: DP is On. Please use DDP(Distributed Data Parallel")

    setup_for_distributed(args.local_rank == 0)


class EMA(torch.nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        super().__init__()
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay

    def _update(self, model: nn.Module, update_fn) -> None:
        with torch.no_grad():
            ema_v = self.model.state_dict().values()
            model_v = model.state_dict().values()
            for e, m in zip(ema_v, model_v):
                e.copy_(update_fn(e, m))

    def update_parameters(self, model: nn.Module) -> None:
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


class CrossEntropyLoss:
    def __init__(self, reduction='mean', label_smoothing=0.0) -> None:
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(self, prediction, target):
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

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            decoupled_decay=decoupled_decay,
            lr_in_momentum=lr_in_momentum
        )
        super().__init__(params, defaults)

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
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p.data)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    if 'decoupled_decay' in group and group['decoupled_decay']:
                        p.data.add_(p.data, alpha=-group['weight_decay'])
                    else:
                        grad = grad.add(p.data, alpha=group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).add(group['eps']).sqrt_()
                else:
                    avg = square_avg.add(group['eps']).sqrt_()
                    
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in group and group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg, value=group['lr'])
                        p.data.add_(-buf)
                    else:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss


class StepLR:

    def __init__(
        self,
        optimizer,
        step_size,
        gamma=1.,
        warmup_epochs=0,
        warmup_lr_init=0
    ) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init

        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lr_values = [group['initial_lr'] for group in self.optimizer.param_groups]
        self.update_groups(self.base_lr_values)

        if self.warmup_epochs:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_epochs for v in self.base_lr_values]
            self.update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_lr_values]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [base_lr * (self.gamma ** (epoch // self.step_size)) for base_lr in self.base_lr_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value
