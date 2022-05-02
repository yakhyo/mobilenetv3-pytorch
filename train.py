import datetime
import os
import time

import torch
import torch.utils.data

from torch.optim.lr_scheduler import StepLR
from torch import optim
from torchvision.transforms.functional import InterpolationMode

from nets import nn
from torchvision.transforms import transforms, autoaugment

from tools import utils, dataset


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_ema.update_parameters(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, train_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    with torch.inference_mode():
        for image, target in metric_logger.log_every(train_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")

    return metric_logger.acc1.global_avg, acc1, acc5


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")

    print("Loading training data")
    st = time.time()
    train_dataset = dataset.ImageFolder(
        traindir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.train_crop_size, interpolation=InterpolationMode(args.interpolation)),
            transforms.RandomHorizontalFlip(0.5),
            autoaugment.AutoAugment(interpolation=InterpolationMode(args.interpolation)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(args.random_erase),
        ])
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    test_dataset = dataset.ImageFolder(
        valdir,
        transform=transforms.Compose([
            transforms.Resize(args.val_resize_size, interpolation=InterpolationMode(args.interpolation)),
            transforms.CenterCrop(args.val_crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    )

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    return train_dataset, test_dataset, train_loader, test_loader, train_sampler, test_sampler


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("COULD NOT FOUND GPU MACHINE")

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    train_dataset, test_dataset, train_loader, test_loader, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                                                    args)

    num_classes = len(train_dataset.classes)

    print("Creating model")
    model = nn.MobileNetV3L(num_classes=num_classes, width_mult=1.0)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    parameters = utils.add_weight_decay(model)
    criterion = nn.PolyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.RMSprop(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-5, eps=0.0316, alpha=0.9)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = nn.EMA(model_without_ddp, decay=0.9999)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        model_ema.model.load_state_dict(checkpoint["model_ema"])

    print("Start training")
    start_time = time.time()
    best = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args, model_ema)
        scheduler.step()
        evaluate(model, criterion, test_loader, device=device)
        _, acc1, acc5 = evaluate(model_ema.model, criterion, test_loader, device=device, log_suffix='EMA')
        checkpoint = {"model": model_without_ddp.state_dict(),
                      "model_ema": model_ema.model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "epoch": epoch,
                      "args": args,
                      }

        torch.save(checkpoint, "weights/last.pth")
        if acc1 > best:
            torch.save(checkpoint, "weights/best.pth")
        best = max(acc1, best)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="MobileNetV3 Large/Small training code")

    parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")

    parser.add_argument("--batch-size", default=32, type=int, help="images per gpu, total = $NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, help="number of total epochs to run")
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers")

    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")

    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")

    parser.add_argument("--sync-bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # dataset resize
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
