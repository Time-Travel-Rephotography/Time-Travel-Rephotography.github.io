from argparse import (
    ArgumentParser,
    Namespace,
)
from typing import Optional

import numpy as np
import torch
from torch import nn

from losses.perceptual_loss import PerceptualLoss
from models.degrade import Downsample
from utils.misc import optional_string


class ReconstructionArguments:
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument("--vggface", type=float, default=0.3, help="vggface")
        parser.add_argument("--vgg", type=float, default=1, help="vgg")
        parser.add_argument('--recon_size', type=int, default=256, help="size for face reconstruction loss")

    @staticmethod
    def to_string(args: Namespace) -> str:
        return (
            f"s{args.recon_size}"
            + optional_string(args.vgg > 0, f"-vgg{args.vgg}")
            + optional_string(args.vggface > 0, f"-vggface{args.vggface}")
        )


def create_perceptual_loss(args: Namespace):
    return PerceptualLoss(lambda_vgg=args.vgg, lambda_vggface=args.vggface, cos_dist=False)


class EyeLoss(nn.Module):
    def __init__(
            self,
            target: torch.Tensor,
            input_size: int = 1024,
            input_channels: int = 3,
            percept: Optional[nn.Module] = None,
            args: Optional[Namespace] = None
    ):
        """
        target: target image
        """
        assert not (percept is None and args is None)

        super().__init__()

        self.target = target

        target_size = target.shape[-1]
        self.downsample = Downsample(input_size, target_size, input_channels) \
                if target_size != input_size else (lambda x: x)

        self.percept = percept if percept is not None else create_perceptual_loss(args)

        eye_size = np.array((224, 224))
        btlrs = []
        for sgn in [1, -1]:
            center = np.array((480, 384 * sgn))   # (y, x)
            b, t = center[0] - eye_size[0] // 2, center[0] + eye_size[0] // 2
            l, r = center[1] - eye_size[1] // 2, center[1] + eye_size[1] // 2
            btlrs.append((np.array((b, t, l, r)) / 1024 * target_size).astype(int))
        self.btlrs = np.stack(btlrs, axis=0)

    def forward(self, img: torch.Tensor, degrade: nn.Module = None):
        """
        img: it should be the degraded version of the generated image
        """
        if degrade is not None:
            img = degrade(img, downsample=self.downsample)

        loss = 0
        for (b, t, l, r) in self.btlrs:
            loss = loss + self.percept(
                img[:, :, b:t, l:r], self.target[:, :, b:t, l:r],
                use_vggface=False, max_vgg_layer=4,
                # use_vgg=False,
            )
        return loss


class FaceLoss(nn.Module):
    def __init__(
            self,
            target: torch.Tensor,
            input_size: int = 1024,
            input_channels: int = 3,
            size: int = 256,
            percept: Optional[nn.Module] = None,
            args: Optional[Namespace] = None
    ):
        """
        target: target image
        """
        assert not (percept is None and args is None)

        super().__init__()

        target_size = target.shape[-1]
        self.target = target if target_size == size \
                else Downsample(target_size, size, target.shape[1]).to(target.device)(target)

        self.downsample = Downsample(input_size, size, input_channels) \
                if size != input_size else (lambda x: x)

        self.percept = percept if percept is not None else create_perceptual_loss(args)

    def forward(self, img: torch.Tensor, degrade: nn.Module = None):
        """
        img: it should be the degraded version of the generated image
        """
        if degrade is not None:
            img = degrade(img, downsample=self.downsample)
        loss = self.percept(img, self.target)
        return loss
