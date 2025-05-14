import torch.nn as nn
import timm
import torch.nn.functional as F
from timm.models.registry import register_model
from .resnetm import resnet18m
from timm.models.vision_transformer import Block
import torch
from functools import partial

__all__ = ['HliverFormer']  # model_registry will add each entrypoint fn to this



#   vf output 8,50,384     phase_seqence_block input  embedding 50,8*384
class HliverFormer(nn.Module):
    def __init__(self, model="vit_tiny_patch16_224", pretrained=True):
        super(HliverFormer, self).__init__()
        self.model_name = model
        self.phase_sequence_block_depth = 2
        norm_layer= partial(nn.LayerNorm, eps=1e-6)

        if self.model_name == 'vit_tiny_patch16_224':
            self.vf = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, img_size=112)
            self.fc = nn.Linear(1536, 7)
        if self.model_name == 'vit_small_patch16_224':
            self.vf = timm.create_model("vit_small_patch16_224", pretrained=pretrained, img_size=112)
            self.phase_fc = nn.Linear(3072, 7)
            self.phase_sequence_fc = nn.Linear(3072, 7)
            self.embed_dim = 384
            self.phase_sequence_pos_embed = nn.Parameter(torch.randn(1, 8, self.embed_dim) * .02)
            self.phase_sequence_norm = norm_layer(self.embed_dim*8)
        if self.model_name == 'vit_base_patch16_224':
            self.vf = timm.create_model("vit_base_patch16_224", pretrained=pretrained, img_size=112)
            self.fc = nn.Linear(6144, 7)
        elif self.model_name == 'resnet18':
            self.vf = timm.create_model("resnet18", pretrained=pretrained)
            self.fc = nn.Linear(4096, 7)

        self.phase_sequence_block = nn.Sequential(*[
            Block(
                dim=self.embed_dim*8,
                num_heads=6,
            )
            for _ in range(self.phase_sequence_block_depth)])

    def forward(self, x):
        x = x.reshape(-1, 3, x.shape[3], x.shape[4])
        x = self.vf.forward_features(x)
        if self.model_name == 'vit_tiny_patch16_224':
            x = x.permute(0, 2, 1)  # for vit
            x = x.reshape(-1, x.shape[1] * 8, x.shape[2])  # for vit
            x = F.adaptive_avg_pool1d(x, 1)[:, :, 0]  # for vit
        if self.model_name == 'vit_small_patch16_224':
            x = x.permute(0, 2, 1)  # for vit
            fc_f = x.reshape(-1, x.shape[1] * 8, x.shape[2])  # for vit
            x = fc_f.permute(0,2,1)
            fc_i = F.adaptive_avg_pool1d(fc_f, 1)[:, :, 0]  # for vit

        if self.model_name == 'vit_base_patch16_224':
            x = x.permute(0, 2, 1)  # for vit
            x = x.reshape(-1, x.shape[1] * 8, x.shape[2])  # for vit
            x = F.adaptive_avg_pool1d(x, 1)[:, :, 0]  # for vit
        elif self.model_name == 'resnet18':
            x = x.reshape(-1, x.shape[1] * 8, x.shape[2] * x.shape[3])  # for resnet
            x = F.adaptive_avg_pool1d(x, 1)[:, :, 0]
        fc_out = self.phase_fc(fc_i)
        x = self.phase_sequence_block(x)
        x = self.phase_sequence_norm(x)
        x = F.adaptive_avg_pool1d(x.permute(0,2,1), 1)[:, :, 0]
        x = self.phase_sequence_fc(x)
        return fc_out,x


@register_model
def hilverformer_resnet18(pretrained=True, **kwargs):
    return HliverFormer(model="resnet18", pretrained=pretrained)


@register_model
def hilverformer_vit_tiny(pretrained=True, **kwargs):
    return HliverFormer(model="vit_tiny_patch16_224", pretrained=pretrained)


@register_model
def hilverformer_vit_small(pretrained=True, **kwargs):
    return HliverFormer(model="vit_small_patch16_224", pretrained=pretrained)


@register_model
def hilverformer_vit_base(pretrained=True, **kwargs):
    return HliverFormer(model="vit_base_patch16_224", pretrained=pretrained)
