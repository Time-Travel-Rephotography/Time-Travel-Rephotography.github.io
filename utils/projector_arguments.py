import os
from argparse import (
    ArgumentParser,
    Namespace,
)

from models.degrade import DegradeArguments
from tools.initialize import InitializerArguments
from losses.joint_loss import LossArguments
from utils.optimize import OptimizerArguments
from .misc import (
    optional_string,
    iterable_to_str,
)


class ProjectorArguments:
    def __init__(self):
        parser = ArgumentParser("Project image into stylegan2")
        self.add_arguments(parser)
        self.parser = parser

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        parser.add_argument('--rand_seed', type=int, default=None,
                            help="random seed")
        cls.add_io_args(parser)
        cls.add_preprocess_args(parser)
        cls.add_stylegan_args(parser)

        InitializerArguments.add_arguments(parser)
        LossArguments.add_arguments(parser)
        OptimizerArguments.add_arguments(parser)
        DegradeArguments.add_arguments(parser)

    @staticmethod
    def add_stylegan_args(parser: ArgumentParser):
        parser.add_argument('--ckpt', type=str, default="checkpoint/stylegan2-ffhq-config-f.pt",
                            help="stylegan2 checkpoint")
        parser.add_argument('--generator_size', type=int, default=1024,
                            help="output size of the generator")

    @staticmethod
    def add_io_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('input', type=str, help="input image path")
        parser.add_argument('--results_dir', default="results/projector", help="directory to save results.")

    @staticmethod
    def add_preprocess_args(parser: ArgumentParser):
       # parser.add_argument("--match_histogram", action='store_true', help="match the histogram of the input image to the sibling")
       pass

    def parse(self, args=None, namespace=None) -> Namespace:
        args = self.parser.parse_args(args, namespace=namespace)
        self.print(args)
        return args

    @staticmethod
    def print(args: Namespace):
        print("------------ Parameters -------------")
        args = vars(args)
        for k, v in sorted(args.items()):
            print(f"{k}: {v}")
        print("-------------------------------------")

    @staticmethod
    def to_string(args: Namespace) -> str:
        return "-".join([
            #+ optional_string(args.no_camera_response, "-noCR")
            #+ optional_string(args.match_histogram, "-MH")
            DegradeArguments.to_string(args),
            InitializerArguments.to_string(args),
            LossArguments.to_string(args),
            OptimizerArguments.to_string(args),
        ]) + optional_string(args.rand_seed is not None, f"-S{args.rand_seed}")

