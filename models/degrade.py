from argparse import (
    ArgumentParser,
    Namespace,
)

import torch
from torch import nn
from torch.nn import functional as F

from utils.misc import optional_string

from .gaussian_smoothing import GaussianSmoothing


class DegradeArguments:
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument('--spectral_sensitivity', choices=["g", "b", "gb"], default="g",
            help="Type of spectral sensitivity. g: grayscale (panchromatic), b: blue-sensitive, gb: green+blue (orthochromatic)")
        parser.add_argument('--gaussian', type=float, default=0,
            help="estimated blur radius in pixels of the input photo if it is scaled to 1024x1024")

    @staticmethod
    def to_string(args: Namespace) -> str:
        return (
            f"{args.spectral_sensitivity}"
            + optional_string(args.gaussian > 0, f"-G{args.gaussian}")
        )


class CameraResponse(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_parameter("gamma", nn.Parameter(torch.ones(1)))
        self.register_parameter("offset", nn.Parameter(torch.zeros(1)))
        self.register_parameter("gain", nn.Parameter(torch.ones(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, max=1, min=-1+1e-2)
        x = (1 + x) * 0.5
        x = self.offset + self.gain * torch.pow(x, self.gamma)
        x = (x - 0.5) * 2
        # b = torch.clamp(b, max=1, min=-1)
        return x


class SpectralResponse(nn.Module):
    # TODO: use enum instead for color mode
    def __init__(self, spectral_sensitivity: str = 'b'):
        assert spectral_sensitivity in ("g", "b", "gb"), f"spectral_sensitivity {spectral_sensitivity} is not implemented."

        super().__init__()

        self.spectral_sensitivity = spectral_sensitivity

        if self.spectral_sensitivity == "g":
            self.register_buffer("to_gray", torch.tensor([0.299, 0.587, 0.114]).reshape(1, -1, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.spectral_sensitivity == "b":
            x = rgb[:, -1:]
        elif self.spectral_sensitivity == "gb":
            x = (rgb[:, 1:2] + rgb[:, -1:]) * 0.5
        else:
            assert self.spectral_sensitivity == "g"
            x = (rgb * self.to_gray).sum(dim=1, keepdim=True)
        return x


class Downsample(nn.Module):
    """Antialiasing downsampling"""
    def __init__(self, input_size: int, output_size: int, channels: int):
        super().__init__()
        if input_size % output_size == 0:
            self.stride = input_size // output_size
            self.grid = None
        else:
            self.stride = 1
            step = input_size / output_size
            x = torch.arange(output_size) * step
            Y, X = torch.meshgrid(x, x)
            grid = torch.stack((X, Y), dim=-1)
            grid /= torch.Tensor((input_size - 1, input_size - 1)).view(1, 1, -1)
            grid = grid * 2 - 1
            self.register_buffer("grid", grid)
        sigma = 0.5 * input_size / output_size
        #print(f"{input_size} -> {output_size}: sigma={sigma}")
        self.blur = GaussianSmoothing(channels, int(2 * (sigma * 2) + 1 + 0.5), sigma)

    def forward(self, im: torch.Tensor):
        out = self.blur(im, stride=self.stride)
        if self.grid is not None:
            out = F.grid_sample(out, self.grid[None].expand(im.shape[0], -1, -1, -1))
        return out



class Degrade(nn.Module):
    """
    Simulate the degradation of antique film
    """
    def __init__(self, args:Namespace):
        super().__init__()
        self.srf = SpectralResponse(args.spectral_sensitivity)
        self.crf = CameraResponse()
        self.gaussian = None
        if args.gaussian is not None and args.gaussian > 0:
            self.gaussian = GaussianSmoothing(3, 2 * int(args.gaussian * 2 + 0.5) + 1, args.gaussian)

    def forward(self, img: torch.Tensor, downsample: nn.Module = None):
        if self.gaussian is not None:
            img = self.gaussian(img)
        if downsample is not None:
            img = downsample(img)
        img = self.srf(img)
        img = self.crf(img)
        # Note that I changed it back to 3 channels
        return img.repeat((1, 3, 1, 1)) if img.shape[1] == 1 else img



