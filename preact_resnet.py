"""
PyTorch implementatino of Pre-activation ResNet.

Reference:
    [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks", arXiv, 2016.
"""

import torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    """
    3x3 convolution with padding
    """
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """
    1x1 convolution
    """
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PreactBasicBlock(torch.nn.Module):
    """
    BasicBlock of ResNet with pre-activation.
    """
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        """
        super().__init__()

        self.norm1 = torch.nn.BatchNorm2d(in_planes)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.norm2 = torch.nn.BatchNorm2d(planes)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        if (stride != 1) or (in_planes != planes):
            self.shortcut = conv1x1(in_planes, planes, stride)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        out = self.norm1(x)
        res = self.relu1(out)
        out = self.conv1(res)
        out = self.conv2(self.relu2(self.norm2(out)))
        res = self.shortcut(res) if self.shortcut is not None else x
        return out + res


class PreactBottleneck(torch.nn.Module):
    """
    Pre-activation version of the original Bottleneck module.
    """
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        """
        super().__init__()

        self.norm1 = torch.nn.BatchNorm2d(in_planes)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_planes, planes)
        self.norm2 = torch.nn.BatchNorm2d(planes)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.norm3 = torch.nn.BatchNorm2d(planes)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv3 = conv1x1(planes, self.expansion * planes)

        if (stride != 1) or (in_planes != (self.expansion * planes)):
            self.shortcut = conv1x1(in_planes, self.expansion * planes, stride)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        out = self.norm1(x)
        res = self.relu1(out)
        out = self.conv1(res)
        out = self.conv2(self.relu2(self.norm2(out)))
        out = self.conv3(self.relu3(self.norm3(out)))
        res = self.shortcut(res) if self.shortcut is not None else x
        return out + res


class PreactResNet(torch.nn.Module):
    """
    """
    def __init__(self, block: torch.nn.Module, num_blocks: list, num_classes: int = 10):
        """
        """
        super().__init__()

        self.in_planes = 64

        self.conv1  = conv3x3(3, 64, stride=1)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: torch.nn.Module, planes: int, num_blocks: list, stride: int) -> torch.nn.Sequential:
        """
        """
        layers = []

        for idx in range(num_blocks):
            layers.append(block(self.in_planes, planes, stride if idx == 0 else 1))
            self.in_planes = block.expansion * planes

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreactResNet18():
    """
    """
    return PreactResNet(PreactBasicBlock, [2, 2, 2, 2])


def PreactResNet34():
    """
    """
    return PreactResNet(PreactBasicBlock, [3, 4, 6, 3])


def PreactResNet50():
    """
    """
    return PreactResNet(PreactBottleneck, [3, 4, 6, 3])


def PreactResNet101():
    """
    """
    return PreactResNet(PreactBottleneck, [3, 4, 23, 3])


def PreactResNet152():
    """
    """
    return PreactResNet(PreactBottleneck, [3, 8, 36, 3])


if __name__ == "__main__":

    def test_model(model_class):
        """
        """
        model  = model_class()
        tensor = torch.randn((1, 3, 32, 32))
        output = model(tensor)
        print(model)
        print(tensor.shape, "->", output.shape)
        print()

    test_model(PreactResNet18)
    test_model(PreactResNet34)
    test_model(PreactResNet50)
    test_model(PreactResNet101)
    test_model(PreactResNet152)

