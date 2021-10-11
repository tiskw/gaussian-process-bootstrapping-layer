#!/usr/bin/env python3

"""
Training script of the Gaussian process bootstrapping layer (GPB layer).
"""

import argparse
import time

import sklearn.metrics
import torch
import torchvision

from model import LeNet5


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--batch_size", default=500, type=int, help="batch size")
    parser.add_argument("--data_aug", action="store_true", help="enable data augmentation")
    parser.add_argument("--device", default="cuda", type=str, help="device type for NN model")
    parser.add_argument("--epoch", default=300, type=int, help="number of training epochs")
    parser.add_argument("--gpb_layer_pos", default="", type=str, help="number of GPB layers")
    parser.add_argument("--log_interval", default=10, type=int, help="logging interval")
    parser.add_argument("--n_cpus", default=4, type=int, help="number of available CPUs")
    parser.add_argument("--save", default="", type=str, help="save trained model")
    parser.add_argument("--std_error", default=0.2, type=float, help="standard deviation of observation")
    return parser.parse_args()


def train(model, args, quiet=False, alpha_avg_loss=0.9):
    """
    Train NN model.

    Args:
        model (torch.nn.Module): NN model to be trained.
        args (argpaese.Namespace): Command line arguments.
        quiet (bool): If true, no training log is dumped.
        alpha_avg_loss (float): Coefficient of moving average for loss values.
    """
    model.to(args.device)

    # Setup training dataset.
    transforms = [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    transform = torchvision.transforms.Compose(transforms if args.data_aug else transforms[2:])
    dataset = torchvision.datasets.CIFAR10("dataset", train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus)

    # Setup optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0E-3)

    # Initialize average loss.
    avg_loss = 0.0

    time_start = time.time()

    for epoch in range(args.epoch):

        model.train()

        for step, (images, labels) in enumerate(dataloader):

            images = images.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()

            # Compute model output and loss value.
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, labels, reduction="mean")

            # Update average loss.
            avg_loss = alpha_avg_loss * avg_loss + (1.0 - alpha_avg_loss) * float(loss)

            # Run back propergation.
            loss.backward()
            optimizer.step()

            time_elapsed = time.time() - time_start

            # Dump log.
            if (not quiet) and (step % args.log_interval == 0):
                messages = [
                    f"epoch = {epoch:d}",
                    f"step = {step:d}",
                    f"loss = {loss:.4f}",
                    f"avg_loss = {avg_loss:.4f}",
                    f"time_elapsed = {time_elapsed:.1f} [sec]",
                ]
                print("Train: " + "\t".join(messages))

        # Run test procedure.
        test(model, args)


def test(model, args):
    """
    Test NN model.

    Args:
        model (torch.nn.Module): NN model to be tested.
        args (argpaese.Namespace): Command line arguments.
    """
    model.to(args.device)
    model.eval()

    # Setup test dataset.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_cpus)

    # Initialize values that are necessary for computing test scores.
    num_dat = 0
    lossval = 0.0
    ys_pred = list()
    ys_true = list()

    for images, labels in dataloader:

        images = images.to(args.device)
        labels = labels.to(args.device)

        # Compute model output.
        output = model(images)

        # Update values.
        num_dat += images.shape[0]
        lossval += torch.nn.functional.cross_entropy(output, labels, reduction="sum")
        ys_pred += torch.argmax(output, dim=1).tolist()
        ys_true += labels.tolist()

    # Compute test scores.
    scores = {
        "accuracy" : sklearn.metrics.accuracy_score(ys_true, ys_pred),
        "precision": sklearn.metrics.precision_score(ys_true, ys_pred, average="macro"),
        "recall"   : sklearn.metrics.recall_score(ys_true, ys_pred, average="macro"),
        "f1"       : sklearn.metrics.f1_score(ys_true, ys_pred, average="macro"),
        "loss"     : lossval / num_dat,
    }

    # Dump log.
    messages = [
        f"accuracy = {scores['accuracy']:.4f}",
        f"precision = {scores['precision']:.4f}",
        f"recall = {scores['recall']:.4f}",
        f"f1 = {scores['f1']:.4f}",
        f"loss = {scores['loss']:.4f}",
    ]
    print("Test: " + "\t".join(messages))


def main(args):
    """
    Main function.
    """
    # Dump arguments.
    print("args =", args)

    # Create NN model instance.
    model = LeNet5(gpb_layer_pos=args.gpb_layer_pos, std_error=args.std_error)
    print(model)

    # Start training procedure.
    train(model, args)

    # Save model.
    if args.save:
        torch.save(model.to("cpu").state_dict(), args.save)


if __name__ == "__main__":
    main(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
