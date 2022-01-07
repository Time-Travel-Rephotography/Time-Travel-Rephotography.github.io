from argparse import ArgumentParser, Namespace
from typing import (
    List,
    Tuple,
)

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    Grayscale,
    Resize,
    ToTensor,
)

from models.encoder import Encoder
from models.encoder4editing import (
    get_latents as get_e4e_latents,
    setup_model as setup_e4e_model,
)
from utils.misc import (
    optional_string,
    iterable_to_str,
    stem,
)



class ColorEncoderArguments:
    def __init__(self):
        parser = ArgumentParser("Encode an image via a feed-forward encoder")

        self.add_arguments(parser)

        self.parser = parser

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument("--encoder_ckpt", default=None,
                            help="encoder checkpoint path. initialize w with encoder output if specified")
        parser.add_argument("--encoder_size", type=int, default=256,
                            help="Resize to this size to pass as input to the encoder")


class InitializerArguments:
    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        ColorEncoderArguments.add_arguments(parser)
        cls.add_e4e_arguments(parser)
        parser.add_argument("--mix_layer_range", default=[10, 18], type=int, nargs=2,
                help="replace layers <start> to <end> in the e4e code by the color code")

        parser.add_argument("--init_latent", default=None, help="path to init wp")

    @staticmethod
    def to_string(args: Namespace):
        return (f"init{stem(args.init_latent).lstrip('0')[:10]}" if args.init_latent
               else f"init({iterable_to_str(args.mix_layer_range)})")
            #+ optional_string(args.init_noise > 0, f"-initN{args.init_noise}")

    @staticmethod
    def add_e4e_arguments(parser: ArgumentParser):
        parser.add_argument("--e4e_ckpt", default='checkpoint/e4e_ffhq_encode.pt',
                            help="e4e checkpoint path.")
        parser.add_argument("--e4e_size", type=int, default=256,
                            help="Resize to this size to pass as input to the e4e")



def create_color_encoder(args: Namespace):
    encoder = Encoder(1, args.encoder_size, 512)
    ckpt = torch.load(args.encoder_ckpt)
    encoder.load_state_dict(ckpt["model"])
    return encoder


def transform_input(img: Image):
    tsfm = Compose([
        Grayscale(),
        Resize(args.encoder_size),
        ToTensor(),
    ])
    return tsfm(img)


def encode_color(imgs: torch.Tensor, args: Namespace) -> torch.Tensor:
    assert args.encoder_size is not None

    imgs = Resize(args.encoder_size)(imgs)

    color_encoder = create_color_encoder(args).to(imgs.device)
    color_encoder.eval()
    with torch.no_grad():
        latent = color_encoder(imgs)
    return latent.detach()


def resize(imgs: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(imgs, size=size, mode='bilinear')


class Initializer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.path = None
        if args.init_latent is not None:
            self.path = args.init_latent
            return


        assert args.encoder_size is not None
        self.color_encoder = create_color_encoder(args)
        self.color_encoder.eval()
        self.color_encoder_size = args.encoder_size

        self.e4e, e4e_opts = setup_e4e_model(args.e4e_ckpt)
        assert 'cars_' not in e4e_opts.dataset_type
        self.e4e.decoder.eval()
        self.e4e.eval()
        self.e4e_size = args.e4e_size

        self.mix_layer_range = args.mix_layer_range

    def encode_color(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Get the color W code
        """
        imgs = resize(imgs, self.color_encoder_size)

        latent = self.color_encoder(imgs)

        return latent

    def encode_shape(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = resize(imgs, self.e4e_size)
        imgs = (imgs - 0.5) / 0.5
        if imgs.shape[1] == 1: # 1 channel
            imgs = imgs.repeat(1, 3, 1, 1)
        return get_e4e_latents(self.e4e, imgs)

    def load(self, device: torch.device):
        latent_np = np.load(self.path)
        return torch.tensor(latent_np, device=device)[None, ...]

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.path is not None:
            return self.load(imgs.device)

        shape_code = self.encode_shape(imgs)
        color_code = self.encode_color(imgs)

        # style mix
        latent = shape_code
        start, end = self.mix_layer_range
        latent[:, start:end] = color_code
        return latent
