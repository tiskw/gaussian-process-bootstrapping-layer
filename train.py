#!/usr/bin/env python3

"""
Training script of the Gaussian process bootstrapping layer (GPB layer).
"""

import argparse
import time

import rich.progress
import torch
import torchmetrics
import torchvision

from model import LeNet5, ResNet18, update_gpbl_scale


def parse_args():
    """
    Parse command line arguments.
    """
    # Create parser instance.
    fmtcls = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=32)
    parser = argparse.ArgumentParser(description=__doc__.strip(), formatter_class=fmtcls, add_help=False)

    # Define group 1: model options.
    group1 = parser.add_argument_group("Model options")
    group1.add_argument("--model", metavar="NAME", type=str, default="resnet18",
                        help="name of neural network model")
    group1.add_argument("--use_pretrain", action="store_true",
                        help="use pretrained weights")

    # Define group 2: GPB layer options.
    group2 = parser.add_argument_group("GPB layer options")
    group2.add_argument("--gpb_dec_epochs", metavar="INT", default=None, type=int,
                        help="decrease GPBLayer effect last few epochs")
    group2.add_argument("--gpb_layer_pos", metavar="STR", default="", type=str,
                        help="position of GPB layers in a neural network")
    group2.add_argument("--gpb_max_scale", metavar="FLT", type=float, default=1.0,
                        help="scale of noise applied in GPB layers")
    group2.add_argument("--gpb_std_error", metavar="FLT", type=float, default=0.2,
                        help="standard deviation of observation")

    # Define group 3: dataset options.
    group3 = parser.add_argument_group("Dataset options")
    group3.add_argument("--dataset", metavar="NAME", type=str, default="cifar10",
                        help="dataset name")

    # Define group 4: training options.
    group4 = parser.add_argument_group("Training options")
    group4.add_argument("--batch_size", metavar="INT", type=int, default=250,
                        help="size of batch")
    group4.add_argument("--data_aug", action="store_true",
                        help="enable data augmentation")
    group4.add_argument("--device", metavar="STR", type=str, default="cuda",
                        help="device type for NN model")
    group4.add_argument("--epochs", metavar="INT", type=int, default=200,
                        help="number of training epochs")
    group4.add_argument("--n_cpus", metavar="INT", type=int, default=4,
                        help="number of available CPUs")

    # Define group 5: output options.
    group5 = parser.add_argument_group("Output options")
    group5.add_argument("--log", metavar="PATH", type=str, default=None,
                        help="path to log file")
    group5.add_argument("--save", metavar="PATH", type=str, default=None,
                        help="path to trained weights")

    # Define group 5: other options.
    group6 = parser.add_argument_group("Other options")
    group6.add_argument("-h", "--help", action="help",
                        help="show this message and exit")
    group6.add_argument("-v", "--version", action="version", version="",
                        help="show version info and exit")

    return parser.parse_args()


def train(model, dataloader, loss_func, n_cls, optimizer, scheduler=None):
    """
    Train NN model.

    Args:
        model      (torch.nn.Module)            : Target model.
        dataloader (torch.utils.data.DataLoader): Data loader.
        loss_func  (torch.nn.Module)            : Loss function.
        optimizer  (torch.optim.Optimizer)      : Optimizer.
        scheduler  (torch.optim.LRScheduler)    : Learning rate scheduler.
    """
    # Get target device name from the model.
    device = next(model.parameters()).device

    # Define and instanciate metrics.
    metrics = {"train/loss": torchmetrics.MeanMetric(),
               "train/acc" : torchmetrics.Accuracy(task="multiclass", num_classes=n_cls)}

    # Transition to traning mode.
    model.train()

    # Start traninig loop.
    for images, labels in rich.progress.track(dataloader, total=len(dataloader), transient=True):

        # Move images/labels to the target device.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Initialize optimizer.
        optimizer.zero_grad()

        # Compute model output and loss value.
        output = model(images)
        loss   = loss_func(output, labels)

        # Run back propergation.
        loss.backward()
        optimizer.step()

        # Step learning rate scheduler.
        if scheduler is not None:
            scheduler.step()

        # Update metrics.
        metrics["train/loss"].update(loss.to("cpu"))
        metrics["train/acc"].update(output.to("cpu"), labels.to("cpu"))

    # Returns metrics.
    return {key:float(metric.compute()) for key, metric in metrics.items()}


def test(model, dataloader, loss_func, n_cls):
    """
    Test NN model.

    Args:
        model      (torch.nn.Module)            : Target model.
        dataloader (torch.utils.data.DataLoader): Data loader.
        loss_func  (torch.nn.Module)            : Loss function.
    """
    # Get target device name from the model.
    device = next(model.parameters()).device

    # Define and instanciate metrics.
    metrics = {"test/loss": torchmetrics.MeanMetric(),
               "test/acc" : torchmetrics.Accuracy(task="multiclass", num_classes=n_cls)}

    # Transition to evaluation mode.
    model.eval()

    # Start test loop.
    for images, labels in rich.progress.track(dataloader, total=len(dataloader), transient=True):

        # Move images/labels to the target device.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Compute model output and loss value.
        output = model(images)
        loss   = loss_func(output, labels)

        # Update metrics.
        metrics["test/loss"].update(loss.to("cpu"))
        metrics["test/acc"].update(output.to("cpu"), labels.to("cpu"))

    # Returns metrics.
    return {key:float(metric.compute()) for key, metric in metrics.items()}


def main(args):
    """
    Entry point of this script.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # Dump arguments.
    print("args =", args)

    # Enables cuDNN's benchmark multiple convolution algorithms.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Setup transforms.
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Setup dataset.
    if args.dataset == "cifar10":
        ds_common_args = {"root": "dataset", "download": True}
        dataset_train  = torchvision.datasets.CIFAR10(train=True,  transform=transform_train, **ds_common_args)
        dataset_test   = torchvision.datasets.CIFAR10(train=False, transform=transform_test,  **ds_common_args)
        num_classes    = 10
    elif args.dataset == "cifar100":
        ds_common_args = {"root": "dataset", "download": True}
        dataset_train  = torchvision.datasets.CIFAR100(train=True,  transform=transform_train, **ds_common_args)
        dataset_test   = torchvision.datasets.CIFAR100(train=False, transform=transform_test,  **ds_common_args)
        num_classes    = 100

    # Setup data loader.
    dl_common_args   = {"batch_size": args.batch_size, "num_workers": args.n_cpus, "pin_memory": True}
    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True,  **dl_common_args)
    dataloader_test  = torch.utils.data.DataLoader(dataset_test , shuffle=False, **dl_common_args)

    # Create NN model instance.
    model_common_args = {"gpb_layer_pos": args.gpb_layer_pos, "std_error": args.gpb_std_error, "pretrain": args.use_pretrain}
    if   args.model == "lenet5"  : model = LeNet5(num_classes, **model_common_args)
    elif args.model == "resnet18": model = ResNet18(num_classes, **model_common_args)
    model.to(args.device)
    print(model)

    # Define loss function.
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.1)

    # Setup optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0E-4, momentum=0.9, weight_decay=5.0E-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader_train), epochs=args.epochs)

    for epoch in range(1, args.epochs + 1):

        # Change scale of GPB layers if decreasing epochs specified.
        if args.gpb_dec_epochs is not None and (epoch > (args.epochs - args.gpb_dec_epochs)):
            scale = (args.epochs - epoch) / args.gpb_dec_epochs
            model = update_gpbl_scale(model, scale)

        # Train the model for one epoch and get metrics.
        result_train = train(model, dataloader_train, loss_func, num_classes, optimizer, scheduler)

        # Test the model and get metrics.
        result_test = test(model, dataloader_test, loss_func, num_classes)

        # Print metrics.
        result = dict(**result_train, **result_test)
        print("\t".join(["[%03d]" % epoch] + [key + "=" + str(val)[:10].rjust(10) for key, val in result.items()]))

    # Save model.
    if args.save is not None:
        torch.save(model.to("cpu").state_dict(), args.save)


if __name__ == "__main__":
    main(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
