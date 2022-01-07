from functools import partial

from torch import nn


def activation_func(activation: str):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def norm_module(norm: str):
    return {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
    }[norm]


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = 'relu'):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(
            self, in_channels: int, out_channels: int,
            expansion: int = 1, downsampling: int = 1,
            conv=conv3x3, norm: str = 'batch', *args, **kwargs
    ):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling = expansion, downsampling
        self.conv, self.norm = conv, norm_module(norm)
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            self.norm(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_norm(in_channels: int, out_channels: int, conv, norm, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), norm(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(
            self, in_channels: int, out_channels: int, bias: bool = False, *args, **kwargs
    ):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_norm(
                self.in_channels, self.out_channels, conv=self.conv, norm=self.norm,
                bias=bias, stride=self.downsampling
            ),
            self.activate,
            conv_norm(self.out_channels, self.expanded_channels, conv=self.conv, norm=self.norm, bias=bias),
        )

