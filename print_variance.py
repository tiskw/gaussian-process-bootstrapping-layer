#!/usr/bin/env python3

"""
Training script for CIFAR10 classifier.
"""

import argparse
import os
import time

import numpy
import sklearn.metrics
import torch
import torchvision

from model import LeNet5, GaussianProcessBootstrapping


class GaussianProcessBootstrappingPrintVar(GaussianProcessBootstrapping):
    """
    PyTorch implementation of Gaussian process bootstrapping layer.
    """
    def forward(self, X):
        """
        Forward function for 2D tensor of shape (n_samples, n_channels).

        Args:
            inputs (torch.tensor): Input tensor of shape (n_samples, n_channels).
        """
        if self.P is None:
            self.P = torch.zeros((X.shape[1], X.shape[1]), requires_grad=False, device=X.device)

        with torch.no_grad():
            X_copy = X.clone().detach()
            self.P = self.a * self.P + (1.0 - self.a) * torch.matmul(torch.transpose(X_copy, 0, 1), X_copy)

        with torch.no_grad():
            e = self.e
            P = self.P.clone().detach().double()
            I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
            S = I - torch.linalg.lstsq(P + e * I, P)[0]
            M = (I - torch.matmul(P, S) / e).float()

        V = torch.sum(torch.matmul(X, M) * X, dim=1, keepdim=True)

        if not self.training:
            for v in V.clone().detach().cpu().flatten().tolist():
                print(v)

        return X


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--batch_size", default=500, type=int, help="batch size")
    parser.add_argument("--device", default="cuda", type=str, help="device type for NN model")
    parser.add_argument("--epoch", default=10, type=int, help="number of training epochs")
    parser.add_argument("--n_cpus", default=max(1, os.cpu_count()//2), type=int, help="number of available CPUs")
    parser.add_argument("--std_error", default=0.2, type=float, help="standard deviation of the gp resampling layer")
    parser.add_argument("--model", default=None, type=str, help="path to model")
    return parser.parse_args()


def main(args):
    """
    Main function.
    """
    # Dump arguments.
    print("args =", args)

    # Create NN model instance.
    model = LeNet5(gpb_layer_pos="bottom", std_error=args.std_error)
    model[9] = GaussianProcessBootstrappingPrintVar(std_error=args.std_error)
    model.load_state_dict(torch.load(args.model))
    model.to(args.device)
    print(model)

    # Setup training dataset.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpus)

    model.train()
    for _ in range(args.epoch):
        for images, labels in dataloader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            output = model(images)

    model.eval()
    for images, labels in dataloader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        output = model(images)


if __name__ == "__main__":
    main(parse_args())


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
