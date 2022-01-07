"""
Code borrowed from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#file-vgg_perceptual_loss-py-L5
"""
import torch
import torchvision
from models.vggface import VGGFaceFeats


def cos_loss(fi, ft):
    return 1 - torch.nn.functional.cosine_similarity(fi, ft).mean()


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, max_layer=4, cos_dist: bool = False):
        target = (target + 1) * 0.5
        input = (input + 1) * 0.5

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        x = input
        y = target
        loss = 0.0
        loss_func = cos_loss if cos_dist else torch.nn.functional.l1_loss
        for bi, block in enumerate(self.blocks[:max_layer]):
            x = block(x)
            y = block(y)
            loss += loss_func(x, y.detach())
        return loss


class VGGFacePerceptualLoss(torch.nn.Module):
    def __init__(self, weight_path: str = "checkpoint/vgg_face_dag.pt", resize: bool = False):
        super().__init__()
        self.vgg = VGGFaceFeats()
        self.vgg.load_state_dict(torch.load(weight_path))

        mean = torch.tensor(self.vgg.meta["mean"]).view(1, 3, 1, 1) / 255.0
        self.register_buffer("mean", mean)

        self.transform = torch.nn.functional.interpolate
        self.resize = resize

    def forward(self, input, target, max_layer: int = 4, cos_dist: bool = False):
        target = (target + 1) * 0.5
        input = (input + 1) * 0.5

        # preprocessing
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = input - self.mean
        target = target - self.mean
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        input_feats = self.vgg(input)
        target_feats = self.vgg(target)

        loss_func = cos_loss if cos_dist else torch.nn.functional.l1_loss
        # calc perceptual loss
        loss = 0.0
        for fi, ft in zip(input_feats[:max_layer], target_feats[:max_layer]):
            loss = loss + loss_func(fi, ft.detach())
        return loss


class PerceptualLoss(torch.nn.Module):
    def __init__(
            self, lambda_vggface: float = 0.025 / 0.15, lambda_vgg: float = 1,  eps: float = 1e-8, cos_dist: bool = False
    ):
        super().__init__()
        self.register_buffer("lambda_vggface", torch.tensor(lambda_vggface))
        self.register_buffer("lambda_vgg", torch.tensor(lambda_vgg))
        self.cos_dist = cos_dist

        if lambda_vgg > eps:
            self.vgg = VGGPerceptualLoss()
        if lambda_vggface > eps:
            self.vggface = VGGFacePerceptualLoss()

    def forward(self, input, target, eps=1e-8, use_vggface: bool = True, use_vgg=True, max_vgg_layer=4):
        loss = 0.0
        if self.lambda_vgg > eps and use_vgg:
            loss = loss + self.lambda_vgg * self.vgg(input, target, max_layer=max_vgg_layer)
        if self.lambda_vggface > eps and use_vggface:
            loss = loss + self.lambda_vggface * self.vggface(input, target, cos_dist=self.cos_dist)
        return loss

