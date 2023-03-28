#!/usr/bin/env python3

"""
PyTorch implementation of Gaussian process bootstrapping (GPB) layer and LeNet5 with GPB layer.
"""

import collections

import torch
import torchvision


class GaussianProcessBootstrapping(torch.nn.Module):
    """
    PyTorch implementation of Gaussian process bootstrapping layer.
    """
    def __init__(self, std_error=1.0, alpha=0.9, scale=1.0, skip=200):
        """
        Constructor.

        Args:
            std_error (float): Standerd deviation of measurement error.
            alpha     (float): Coefficient of moving average for the matrix P.
            skip      (int)  : Skip first `skip` steps because of the unstability of P.
        """
        torch.nn.Module.__init__(self)
        self.P = None
        self.a = alpha
        self.e = std_error**2
        self.k = scale
        self.s = skip
        self.n = 0

    def forward(self, X):
        """
        Run forward inference. Do nothing in evaluation mode.

        Args:
            inputs (torch.tensor): Input tensor of shape (n_samples, n_channels).
                                   Tensor of rank 3 and 4 are not tested yet.
        """
        if   not self.training: return X
        elif len(X.shape) == 1: raise RuntimeError("1D tensor not supported")
        elif len(X.shape) == 2: return self.forward_rank2(X)
        else                  : return self.forward_rankX(X)

    def forward_rank2(self, X):
        """
        Forward function for 2D tensor of shape (n_samples, n_channels).

        Args:
            inputs (torch.tensor): Input tensor of shape (n_samples, n_channels).
        """
        if self.P is None:
            self.P = torch.zeros((X.shape[1], X.shape[1]), requires_grad=False, device=X.device)

        # with torch.no_grad():
        #     X_copy = X.clone().detach()
        #     self.P = self.a * self.P + (1.0 - self.a) * torch.matmul(torch.transpose(X_copy, 0, 1), X_copy)
        #     self.s = max(-1, self.s - 1)

        X_copy = X.clone().detach()
        self.P = self.a * self.P + (1.0 - self.a) * torch.matmul(torch.transpose(X_copy, 0, 1), X_copy)
        self.s = max(-1, self.s - 1)

        if self.s >= 0:
            return X

        # with torch.no_grad():
        #     e = self.e
        #     P = self.P.clone().detach().double()
        #     I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
        #     S = I - torch.linalg.solve(P + e * I, P)
        #     M = (I - torch.matmul(P, S) / e).float()

        e = self.e
        P = self.P.clone().detach().double()
        I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
        S = I - torch.linalg.solve(P + e * I, P)
        M = (I - torch.matmul(P, S) / e).float()

        # NOTE: In the above code, (P + e * I)^-1 @ P was solved by `torch.linalg.solve` function,
        #       but we also be able to use `torch.linalg.lstsq` function instead. The author don't
        #       know which function is better for this case.
        #         >> S = I - torch.linalg.solve(P + e * I, P)    # Current code
        #         >> S = I - torch.linalg.lstsq(P + e * I, P)[0] # This is also acceptable

        # with torch.no_grad():
        #     V = torch.sum(torch.matmul(X, M) * X, dim=1, keepdim=True)
        #     N = torch.sqrt(torch.clip(V, min=1.0E-10, max=None)) * torch.randn(X.shape, device=X.device)

        V = torch.sum(torch.matmul(X, M) * X, dim=1, keepdim=True)
        N = torch.sqrt(torch.clip(V, min=1.0E-10, max=None)) * torch.randn(X.shape, device=X.device)

        return X + self.k * N

    def forward_rankX(self, X):
        """
        Forward function for 3D or more higher tensor of shape (n_samples, n_channels1, n_channels2, ...).

        Args:
            inputs (torch.tensor): Input tensor of shape (n_samples, n_channels1, n_channels2, ...).
        """
        # Convert tensor shape from (N, C, H, W) to (N, W, H, C).
        X_3dim = torch.transpose(X, 1, 3)

        # Convert tensor shape from (N, W, H, C) to (N*W*H, C).
        X_2dim = X_3dim.reshape((-1, X.shape[1]))

        # Apply GPB layer and restore the shape.
        return torch.transpose(self.forward_rank2(torch.flatten(X_2dim, start_dim=1)).reshape(X_3dim.shape), 1, 3)


def LeNet5(num_classes, gpb_layer_pos="", std_error=1.0, skip=100, pretrain=False):
    """
    Returns LeNet model instance.

    Args:
        gpb_layer_pos (str): Position of Gaussian process bootstrapping layer.
                             This string should be a comma separated value of
                             "top", "middle", or "bottom" (e.g. "top,bottom").

    Reference:
        [1] Y. Lecun, L. Bottou, Y. Bengio and P. Haffner,
            "Gradient-based learning applied to document recognition," in Proceedings of the IEEE,
            vol. 86, no. 11, pp. 2278-2324, Nov. 1998, doi: 10.1109/5.726791.
    """
    # Check the contents of `gpb_layer_pos`.
    for element in gpb_layer_pos.split(","):
        if element and (element not in ["top", "middle", "bottom"]):
            raise ValueError(f"unkown gpb layer position: {element}")

    # LeNet5: 1st block.
    layers = [("block1_conv", torch.nn.Conv2d(3, 20, kernel_size=5)),
              ("block1_relu", torch.nn.ReLU()),
              ("block1_pool", torch.nn.MaxPool2d(kernel_size=2))]

    # Gaussian process bootstrapping layer: top position.
    if "top" in gpb_layer_pos:
        layers += [("block1_gpbl", GaussianProcessBootstrapping(std_error=std_error, skip=skip, scale=0.2))]

    # LeNet5: 2nd block.
    layers += [("block2_conv", torch.nn.Conv2d(20, 50, kernel_size=5)),
               ("block2_relu", torch.nn.ReLU()),
               ("block2_pool", torch.nn.MaxPool2d(kernel_size=2)),
               ("block2_flat", torch.nn.Flatten())]

    # Gaussian process bootstrapping layer: middle position.
    if "middle" in gpb_layer_pos:
        layers += [("block2_gpbl", GaussianProcessBootstrapping(std_error=std_error, skip=skip))]

    # LeNet5: 3rd block.
    layers += [("block3_full", torch.nn.Linear(1250, 500)),
               ("block3_relu", torch.nn.ReLU())]

    # Gaussian process bootstrapping layer: bottom position.
    if "bottom" in gpb_layer_pos:
        layers += [("block3_gpbl", GaussianProcessBootstrapping(std_error=std_error, skip=skip))]

    # LeNet5: 4th block.
    layers += [("block4_full", torch.nn.Linear(500, num_classes))]

    return torch.nn.Sequential(collections.OrderedDict(layers))


def ResNet18(num_classes, gpb_layer_pos="", std_error=0.1, scale=1.0, skip=100, pretrain=False):
    """
    """
    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrain else None)

    layers = [
        ("conv1", torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)),
        ("bn1",   base_model.bn1),
        ("relu",  base_model.relu),
        # ("maxpool", base_model.maxpool),
    ]

    if "0" in gpb_layer_pos:
        layers += [("gpbl0", GaussianProcessBootstrapping(std_error=std_error, scale=scale, skip=skip))]

    layers += [("layer1", base_model.layer1)]

    if "1" in gpb_layer_pos:
        layers += [("gpbl1", GaussianProcessBootstrapping(std_error=std_error, scale=scale, skip=skip))]

    layers += [("layer2", base_model.layer2)]

    if "2" in gpb_layer_pos:
        layers += [("gpbl2", GaussianProcessBootstrapping(std_error=std_error, scale=scale, skip=skip))]

    layers += [("layer3", base_model.layer3)]

    if "3" in gpb_layer_pos:
        layers += [("gpbl3", GaussianProcessBootstrapping(std_error=std_error, scale=scale, skip=skip))]

    layers += [("layer4", base_model.layer4)]

    if "4" in gpb_layer_pos:
        layers += [("gpbl4", GaussianProcessBootstrapping(std_error=std_error, scale=scale, skip=skip))]

    layers += [
        ("avgpool", base_model.avgpool),
        ("flatten", torch.nn.Flatten()),
    ]

    if "5" in gpb_layer_pos:
        layers += [("gpbl5", GaussianProcessBootstrapping(std_error=std_error, scale=scale, skip=skip))]

    layers += [
        ("fc", torch.nn.Linear(512, num_classes)),
    ]

    return torch.nn.Sequential(collections.OrderedDict(layers))


def update_gpbl_scale(model, scale):

    for layer in model.modules():
        if layer is isinstance(layer, GaussianProcessBootstrapping):
            layer.k = scale

    return model


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
