from argparse import Namespace, ArgumentParser
from functools import partial

from torch import nn

from .resnet import ResNetBasicBlock, activation_func, norm_module, Conv2dAuto


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--latent_size", type=int, default=512, help="latent size")
    return parser


def create_model(args) -> nn.Module:
    in_channels = 3 if "rgb" in args and args.rgb else 1
    return Encoder(in_channels, args.encoder_size, latent_size=args.latent_size)


class Flatten(nn.Module):
    def forward(self, input_):
        return input_.view(input_.size(0), -1)


class Encoder(nn.Module):
    def __init__(
            self, in_channels: int, size: int, latent_size: int = 512,
            activation: str = 'leaky_relu', norm: str = "instance"
    ):
        super().__init__()

        out_channels0 = 64
        norm_m = norm_module(norm)
        self.conv0 = nn.Sequential(
            Conv2dAuto(in_channels, out_channels0, kernel_size=5),
            norm_m(out_channels0),
            activation_func(activation),
        )

        pool_kernel = 2
        self.pool = nn.AvgPool2d(pool_kernel)

        num_channels = [128, 256, 512, 512]
        # FIXME: this is a hack
        if size >= 256:
            num_channels.append(512)

        residual = partial(ResNetBasicBlock, activation=activation, norm=norm, bias=True)
        residual_blocks = nn.ModuleList()
        for in_channel, out_channel in zip([out_channels0] + num_channels[:-1], num_channels):
            residual_blocks.append(residual(in_channel, out_channel))
            residual_blocks.append(nn.AvgPool2d(pool_kernel))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.last = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(4),    # TODO: not sure whehter this would cause problem
            Flatten(),
            nn.Linear(num_channels[-1], latent_size, bias=True)
        )

    def forward(self, input_):
        out = self.conv0(input_)
        out = self.pool(out)
        out = self.residual_blocks(out)
        out = self.last(out)
        return out
