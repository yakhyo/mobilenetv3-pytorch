import os
import time
import datetime
from copy import deepcopy

import torch
import torch.utils.data

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import transforms, autoaugment

import utils
import nets as nn


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="MobileNetV3 Large/Small training code")
    parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")

    parser.add_argument("--batch-size", default=32, type=int, help="images per gpu, total = num_GPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, help="number of total epochs to run")
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers")

    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")

    parser.add_argument("--warmup-epochs", default=0, type=int, help="number of warmup epochs")
    parser.add_argument("--warmup-lr-init", default=0, type=float, help="warmup learning rate init")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")

    parser.add_argument("--interval", default=10, type=int, help="print frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")

    parser.add_argument("--sync-bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability")

    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local-rank", default=0, type=int, help="number of distributed processes")

    parser.add_argument("--test", action='store_true', help='model testing')
    parser = parser.parse_args()

    return parser


def load_data(args):
    """ Preparing the dataset and Data sampler """
    print('Loading Data')
    print('Loading Training Data')
    st = time.time()
    train_dataset = utils.dataset.ImageFolder(
        os.path.join(args.data_path, "train"),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(size=224, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(0.5),
            autoaugment.AutoAugment(interpolation=InterpolationMode.BILINEAR),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(args.random_erase),
        ])
    )
    print(f'Done! Took {time.time() - st}')

    print('Loading Validation Data')
    st = time.time()
    test_dataset = utils.dataset.ImageFolder(
        os.path.join(args.data_path, "val"),
        transform=transforms.Compose([
            transforms.Resize(size=256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    )
    print(f'Done! Took {time.time() - st}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return train_dataset, test_dataset, train_sampler, test_sampler


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None):
    model.train()
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    lr_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    for batch_idx, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_ema.update_parameters(model)

        acc1, acc5 = utils.accuracy(output, target, top_k=(1, 5))
        batch_size = image.shape[0]

        if args.distributed:
            reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), batch_size)
        top1_m.update(acc1.item(), batch_size)
        top5_m.update(acc5.item(), batch_size)
        batch_time_m.update(batch_size / (time.time() - start_time))
        lr_m.update(optimizer.param_groups[0]['lr'])

        if args.local_rank == 0 and batch_idx % args.interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print(f'Train: [{epoch:>3d}][{batch_idx:>4d}/{len(data_loader)}] '
                  f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                  f'Time: {batch_time_m.val:.3f}s, {batch_size * args.world_size / batch_time_m.val:>4.2f}/s '
                  f'LR: {lr:.7f} '
                  f'Acc@1: {top1_m.val:.4f} ({top1_m.avg:.4f}) '
                  f'Acc@5: {top5_m.val:.4f} ({top5_m.avg:.4f})')


def validate(model, criterion, train_loader, device, args, log_suffix=""):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    end = time.time()
    last_idx = len(train_loader) - 1
    with torch.inference_mode():
        for batch_idx, (image, target) in enumerate(train_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, top_k=(1, 5))
            batch_size = image.shape[0]

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            batch_time_m.update(time.time() - end)
            losses_m.update(reduced_loss.item(), batch_size)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

            end = time.time()
            if args.local_rank == 0 and batch_idx % args.interval == 0:
                print(f'Test_{log_suffix}: [{batch_idx:>4d}/{last_idx}]  '
                      f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                      f'Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f})  '
                      f'Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f})  '
                      f'Acc@5: {top5_m.val:>7.4f} ({top5_m.avg:>7.4f})')

    print(f'Acc@1: {top1_m.avg:>7.4f} Acc@5: {top5_m.avg:>7.4f}')

    return losses_m.avg, top1_m.avg, top5_m.avg


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    os.makedirs('./weights', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    train_dataset, test_dataset, train_sampler, test_sampler = load_data(args)

    print('Creating Data Loaders')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print('Creating Model')
    model = nn.MobileNetV3L(width_mult=1.0).to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    parameters = utils.add_weight_decay(model, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.RMSprop(parameters, lr=args.lr, alpha=0.9, eps=1e-3, weight_decay=0, momentum=args.momentum)
    scheduler = nn.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma, warmup_epochs=args.warmup_epochs,
                          warmup_lr_init=args.warmup_lr_init)
    model_ema = nn.EMA(model, decay=0.9999)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        model_ema.model.load_state_dict(checkpoint["model_ema"])

    print("Start Training")
    start_time = time.time()
    best = 0

    if args.test:
        model_ema = torch.load('weights/last.pth', 'cuda')['model'].float()
        _, acc1, acc5 = validate(model_ema, criterion, test_loader, device=device, args=args, log_suffix='EMA')
    else:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args, model_ema)
            scheduler.step(epoch + 1)
            _, acc1, acc5 = validate(model_ema.model, criterion, test_loader, device=device, args=args, log_suffix='EMA')
            checkpoint = {
                'model': model.module.state_dict(),
                'model_ema': model_ema.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            state_ema = {
                'model': deepcopy(model_ema.model).half()
            }

            torch.save(checkpoint, 'weights/last.ckpt')
            torch.save(state_ema, 'weights/last.pth')
            if acc1 > best:
                torch.save(checkpoint, 'weights/best.ckpt')
                torch.save(state_ema, 'weights/last.pth')
            best = max(acc1, best)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training Time {total_time_str}")


if __name__ == "__main__":
    params = get_args_parser()
    main(args=params)
