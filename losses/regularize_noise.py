from typing import Iterable

import torch
from torch import nn


class NoiseRegularizer(nn.Module):
    def forward(self, noises: Iterable[torch.Tensor]):
        loss = 0

        for noise in noises:
            size = noise.shape[2]

            while True:
                loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                )

                if size <= 8:
                    break

                noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2

        return loss

    @staticmethod
    def normalize(noises: Iterable[torch.Tensor]):
        for noise in noises:
            mean = noise.mean()
            std = noise.std()

            noise.data.add_(-mean).div_(std)

