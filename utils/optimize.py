import math
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
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import Resize

#from optim import get_optimizer_class, OPTIMIZER_MAP
from losses.regularize_noise import NoiseRegularizer
from optim import RAdam
from utils.misc import (
    iterable_to_str,
    optional_string,
)


class OptimizerArguments:
    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument('--coarse_min', type=int, default=32)
        parser.add_argument('--wplus_step', type=int, nargs="+", default=[250, 750], help="#step for optimizing w_plus")
        #parser.add_argument('--lr_rampup', type=float, default=0.05)
        #parser.add_argument('--lr_rampdown', type=float, default=0.25)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--noise_strength', type=float, default=.0)
        parser.add_argument('--noise_ramp', type=float, default=0.75)
        #parser.add_argument('--optimize_noise', action="store_true")
        parser.add_argument('--camera_lr', type=float, default=0.01)

        parser.add_argument("--log_dir", default="log/projector", help="tensorboard log directory")
        parser.add_argument("--log_freq", type=int, default=10, help="log frequency")
        parser.add_argument("--log_visual_freq", type=int, default=50, help="log frequency")

    @staticmethod
    def to_string(args: Namespace) -> str:
        return (
            f"lr{args.lr}_{args.camera_lr}-c{args.coarse_min}"
            + f"-wp({iterable_to_str(args.wplus_step)})"
            + optional_string(args.noise_strength, f"-n{args.noise_strength}")
        )


class LatentNoiser(nn.Module):
    def __init__(
            self, generator: torch.nn,
            noise_ramp: float = 0.75, noise_strength: float = 0.05,
            n_mean_latent: int = 10000
    ):
        super().__init__()

        self.noise_ramp = noise_ramp
        self.noise_strength = noise_strength

        with torch.no_grad():
            # TODO: get 512 from generator
            noise_sample = torch.randn(n_mean_latent, 512, device=generator.device)
            latent_out = generator.style(noise_sample)

            latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    def forward(self, latent: torch.Tensor, t: float) -> torch.Tensor:
        strength = self.latent_std * self.noise_strength * max(0, 1 - t / self.noise_ramp) ** 2
        noise = torch.randn_like(latent) * strength
        return latent + noise


class Optimizer:
    @classmethod
    def optimize(
            cls,
            generator: torch.nn,
            criterion: torch.nn,
            degrade: torch.nn,
            target: torch.Tensor,  # only used in writer since it's mostly baked in criterion
            latent_init: torch.Tensor,
            noise_init: torch.Tensor,
            args: Namespace,
            writer: Optional[SummaryWriter] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # do not optimize generator
        generator = generator.eval()
        target = target.detach()
        # prepare parameters
        noises = []
        for n in noise_init:
            noise = n.detach().clone()
            noise.requires_grad = True
            noises.append(noise)


        def create_parameters(latent_coarse):
            parameters = [
                {'params': [latent_coarse], 'lr': args.lr},
                {'params': noises, 'lr': args.lr},
                {'params': degrade.parameters(), 'lr': args.camera_lr},
            ]
            return parameters


        device = target.device

        # start optimize
        total_steps = np.sum(args.wplus_step)
        max_coarse_size = (2 ** (len(args.wplus_step) - 1)) * args.coarse_min
        noiser = LatentNoiser(generator, noise_ramp=args.noise_ramp, noise_strength=args.noise_strength).to(device)
        latent = latent_init.detach().clone()
        for coarse_level, steps in enumerate(args.wplus_step):
            if criterion.weights["contextual"] > 0:
                with torch.no_grad():
                    # synthesize new sibling image using the current optimization results
                    # FIXME: update rgbs sibling
                    sibling, _, _ = generator([latent], input_is_latent=True, randomize_noise=True)
                    criterion.update_sibling(sibling)

            coarse_size = (2 ** coarse_level) * args.coarse_min
            latent_coarse, latent_fine = cls.split_latent(
                    latent, generator.get_latent_size(coarse_size))
            parameters = create_parameters(latent_coarse)
            optimizer = RAdam(parameters)

            print(f"Optimizing {coarse_size}x{coarse_size}")
            pbar = tqdm(range(steps))
            for si in pbar:
                latent = torch.cat((latent_coarse, latent_fine), dim=1)
                niters = si + np.sum(args.wplus_step[:coarse_level])
                latent_noisy = noiser(latent, niters / total_steps)
                img_gen, _, rgbs = generator([latent_noisy], input_is_latent=True, noise=noises)
                # TODO: use coarse_size instead of args.coarse_size for rgb_level
                loss, losses = criterion(img_gen, degrade=degrade, noises=noises, rgbs=rgbs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                NoiseRegularizer.normalize(noises)

                # log
                pbar.set_description("; ".join([f"{k}: {v.item(): .3e}" for k, v in losses.items()]))

                if writer is not None and niters % args.log_freq == 0:
                    cls.log_losses(writer, niters, loss, losses, criterion.weights)
                    cls.log_parameters(writer, niters, degrade.named_parameters())
                if writer is not None and niters % args.log_visual_freq == 0:
                    cls.log_visuals(writer, niters, img_gen, target, degraded=degrade(img_gen), rgbs=rgbs)

            latent = torch.cat((latent_coarse, latent_fine), dim=1).detach()

        return latent, noises

    @staticmethod
    def split_latent(latent: torch.Tensor, coarse_latent_size: int):
        latent_coarse = latent[:, :coarse_latent_size]
        latent_coarse.requires_grad = True
        latent_fine = latent[:, coarse_latent_size:]
        latent_fine.requires_grad = False
        return latent_coarse, latent_fine

    @staticmethod
    def log_losses(
            writer: SummaryWriter,
            niters: int,
            loss_total: torch.Tensor,
            losses: Dict[str, torch.Tensor],
            weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        writer.add_scalar("loss", loss_total.item(), niters)

        for name, loss in losses.items():
            writer.add_scalar(name, loss.item(), niters)
            if weights is not None:
                writer.add_scalar(f"weighted_{name}", weights[name] * loss.item(), niters)

    @staticmethod
    def log_parameters(
            writer: SummaryWriter,
            niters: int,
            named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    ):
        for name, para in named_parameters:
            writer.add_scalar(name, para.item(), niters)

    @classmethod
    def log_visuals(
            cls,
            writer: SummaryWriter,
            niters: int,
            img: torch.Tensor,
            target: torch.Tensor,
            degraded=None,
            rgbs=None,
    ):
        if target.shape[-1] != img.shape[-1]:
            visual = make_grid(img, nrow=1, normalize=True, range=(-1, 1))
            writer.add_image("pred", visual, niters)

        def resize(img):
            return F.interpolate(img, size=target.shape[2:], mode="area")

        vis = resize(img)
        if degraded is not None:
            vis = torch.cat((resize(degraded), vis), dim=-1)
        visual = make_grid(torch.cat((target.repeat(1, vis.shape[1] // target.shape[1], 1, 1), vis), dim=-1), nrow=1, normalize=True, range=(-1, 1))
        writer.add_image("gnd[-degraded]-pred", visual, niters)

        # log to rgbs
        if rgbs is not None:
            cls.log_torgbs(writer, niters, rgbs)

    @staticmethod
    def log_torgbs(writer: SummaryWriter, niters: int, rgbs: Iterable[torch.Tensor], prefix: str = ""):
        for ri, rgb in enumerate(rgbs):
            scale = 2 ** (-(len(rgbs) - ri))
            visual = make_grid(torch.cat((rgb, rgb / scale), dim=-1), nrow=1, normalize=True, range=(-1, 1))
            writer.add_image(f"{prefix}to_rbg_{2 ** (ri + 2)}", visual, niters)

