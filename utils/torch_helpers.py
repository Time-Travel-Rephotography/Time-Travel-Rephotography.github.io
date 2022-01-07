import torch
from torch import nn


def device(gpu_id=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def load_matching_state_dict(model: nn.Module, state_dict):
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model.load_state_dict(filtered_dict)


def resize(t: torch.Tensor, size: int) -> torch.Tensor:
    B, C, H, W = t.shape
    t = t.reshape(B, C, size, H // size, size, W // size)
    return t.mean([3, 5])


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to('cpu')
            .numpy()
    )


