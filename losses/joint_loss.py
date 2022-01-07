from argparse import (
    ArgumentParser,
    Namespace,
)
from typing import (
    Dict,
    Iterable,
    Optional,
    Tuple,
)

import numpy as np
import torch
from torch import nn

from utils.misc import (
    optional_string,
    iterable_to_str,
)

from .contextual_loss import ContextualLoss
from .color_transfer_loss import ColorTransferLoss
from .regularize_noise import NoiseRegularizer
from .reconstruction import (
    EyeLoss,
    FaceLoss,
    create_perceptual_loss,
    ReconstructionArguments,
)

class LossArguments:
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        ReconstructionArguments.add_arguments(parser)

        parser.add_argument("--color_transfer", type=float, default=1e10, help="color transfer loss weight")
        parser.add_argument("--eye", type=float, default=0.1, help="eye loss weight")
        parser.add_argument('--noise_regularize', type=float, default=5e4)
        # contextual loss
        parser.add_argument("--contextual", type=float, default=0.1, help="contextual loss weight")
        parser.add_argument("--cx_layers", nargs='*', help="contextual loss layers",
                            choices=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'],
                            default=['relu3_4', 'relu2_2', 'relu1_2'])

    @staticmethod
    def to_string(args: Namespace) -> str:
        return (
            ReconstructionArguments.to_string(args)
            + optional_string(args.eye > 0, f"-eye{args.eye}")
            + optional_string(args.color_transfer, f"-color{args.color_transfer:.1e}")
            + optional_string(
                args.contextual,
                f"-cx{args.contextual}({iterable_to_str(args.cx_layers)})"
            )
            #+ optional_string(args.mse, f"-mse{args.mse}")
            + optional_string(args.noise_regularize, f"-NR{args.noise_regularize:.1e}")
        )


class BakedMultiContextualLoss(nn.Module):
    """Random sample different image patches for different vgg layers."""
    def __init__(self, sibling: torch.Tensor, args: Namespace, size: int = 256):
        super().__init__()

        self.cxs = nn.ModuleList([ContextualLoss(use_vgg=True, vgg_layers=[layer])
            for layer in args.cx_layers])
        self.size = size
        self.sibling = sibling.detach()

    def forward(self, img: torch.Tensor):
        cx_loss = 0
        for cx in self.cxs:
            h, w = np.random.randint(0, high=img.shape[-1] - self.size, size=2)
            cx_loss = cx(self.sibling[..., h:h+self.size, w:w+self.size], img[..., h:h+self.size, w:w+self.size]) + cx_loss
        return cx_loss


class BakedContextualLoss(ContextualLoss):
    def __init__(self, sibling: torch.Tensor, args: Namespace, size: int = 256):
        super().__init__(use_vgg=True, vgg_layers=args.cx_layers)
        self.size = size
        self.sibling = sibling.detach()

    def forward(self, img: torch.Tensor):
        h, w = np.random.randint(0, high=img.shape[-1] - self.size, size=2)
        return super().forward(self.sibling[..., h:h+self.size, w:w+self.size], img[..., h:h+self.size, w:w+self.size])


class JointLoss(nn.Module):
    def __init__(
            self,
            args: Namespace,
            target: torch.Tensor,
            sibling: Optional[torch.Tensor],
            sibling_rgbs: Optional[Iterable[torch.Tensor]] = None,
    ):
        super().__init__()

        self.weights = {
            "face": 1., "eye": args.eye,
            "contextual": args.contextual, "color_transfer": args.color_transfer,
            "noise": args.noise_regularize,
        }

        reconstruction = {}
        if args.vgg > 0 or args.vggface > 0:
            percept = create_perceptual_loss(args)
            reconstruction.update(
                {"face": FaceLoss(target, input_size=args.generator_size, size=args.recon_size, percept=percept)}
            )
            if args.eye > 0:
                reconstruction.update(
                    {"eye": EyeLoss(target, input_size=args.generator_size, percept=percept)}
                )
        self.reconstruction = nn.ModuleDict(reconstruction)

        exemplar = {}
        if args.contextual > 0 and len(args.cx_layers) > 0:
            assert sibling is not None
            exemplar.update(
                {"contextual": BakedContextualLoss(sibling, args)}
            )
        if args.color_transfer > 0:
            assert sibling_rgbs is not None
            self.sibling_rgbs = sibling_rgbs
            exemplar.update(
                {"color_transfer": ColorTransferLoss(init_rgbs=sibling_rgbs)}
            )
        self.exemplar = nn.ModuleDict(exemplar)

        if args.noise_regularize > 0:
            self.noise_criterion = NoiseRegularizer()

    def forward(
            self, img, degrade=None, noises=None, rgbs=None, rgb_level: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            rgbs: results from the ToRGB layers
        """
        # TODO: add current optimization resolution for noises

        losses = {}

        # reconstruction losses
        for name, criterion in self.reconstruction.items():
            losses[name] = criterion(img, degrade=degrade)

        # exemplar losses
        if 'contextual' in self.exemplar:
            losses["contextual"] = self.exemplar["contextual"](img)
        if "color_transfer" in self.exemplar:
            assert rgbs is not None
            losses["color_transfer"] = self.exemplar["color_transfer"](rgbs, level=rgb_level)

        # noise regularizer
        if self.weights["noise"] > 0:
            losses["noise"] = self.noise_criterion(noises)

        total_loss = 0
        for name, loss in losses.items():
            total_loss = total_loss + self.weights[name] * loss
        return total_loss, losses

    def update_sibling(self, sibling: torch.Tensor):
        assert "contextual" in self.exemplar
        self.exemplar["contextual"].sibling = sibling.detach()
