import os

import torch
import torch.distributed as distributed


def reduce_tensor(tensor, n):
    """ Getting the average of tensors over multiple GPU devices """
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= n
    return rt


def add_weight_decay(net, weight_decay=1e-5):
    """ Applying weight decay to only weights, not biases """
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


""" The order of `def setup_for_distributed()` and `def init_distributed_mode()` must be kept """


def setup_for_distributed(is_master):
    """ This function disables printing when not in master process """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """ Initializing distributed mode """
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.distributed:
        print(f"| distributed init (rank {args.local_rank}): env://", flush=True)
    else:
        print('Warning: Data Parallel is ON. Please use Distributed Data Parallel.')

    setup_for_distributed(args.local_rank == 0)
