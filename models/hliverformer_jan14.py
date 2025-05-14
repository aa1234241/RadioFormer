import torch.nn as nn
import timm
import torch.nn.functional as F
from timm.models.registry import register_model
from .resnetm import resnet18m
from timm.models.vision_transformer import Block
import torch
from functools import partial

__all__ = ['HliverFormer']  # model_registry will add each entrypoint fn to this


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

            self.phase_fc = nn.Linear(self.vf.embed_dim*8, 7)
            self.phase_norm = norm_layer(self.vf.embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vf.embed_dim))

            self.phase_sequence_map = nn.Linear(8*self.vf.embed_dim, self.vf.embed_dim,bias=False)
            self.phase_sequence_fc = nn.Linear(self.vf.embed_dim, 7)

            self.phase_sequence_pos_embed = nn.Parameter(torch.randn(1, 49, self.vf.embed_dim) * .02)
            self.phase_sequence_norm = norm_layer(self.vf.embed_dim)
        if self.model_name == 'vit_base_patch16_224':
            self.vf = timm.create_model("vit_base_patch16_224", pretrained=pretrained, img_size=112)
            self.fc = nn.Linear(6144, 7)
        elif self.model_name == 'resnet18':
            self.vf = timm.create_model("resnet18", pretrained=pretrained)
            self.fc = nn.Linear(4096, 7)

        self.phase_sequence_block = nn.Sequential(*[
            Block(
                dim=self.vf.embed_dim,
                num_heads=6,
            )
            for _ in range(self.phase_sequence_block_depth)])

    def forward(self, x):
        x = x.reshape(-1, 3, x.shape[3], x.shape[4])
        x = self.vf.forward_features(x)
        # x = x[:,1:,:]
        if self.model_name == 'vit_tiny_patch16_224':
            x = x.permute(0, 2, 1)  # for vit
            x = x.reshape(-1, x.shape[1] * 8, x.shape[2])  # for vit
            x = F.adaptive_avg_pool1d(x, 1)[:, :, 0]  # for vit
        if self.model_name == 'vit_small_patch16_224':
            x = x.permute(0, 2, 1)  # for vit
            fc_f = x.reshape(-1, x.shape[1] * 8, x.shape[2])  # for vit
            x = fc_f.permute(0,2,1)
            x = self.phase_sequence_map(x[:,1:,:])
            # fc_i = F.adaptive_avg_pool1d(fc_f, 1)[:, :, 0]  # for vit
            fc_i = fc_f[:,:,0]

        if self.model_name == 'vit_base_patch16_224':
            x = x.permute(0, 2, 1)  # for vit
            x = x.reshape(-1, x.shape[1] * 8, x.shape[2])  # for vit
            x = F.adaptive_avg_pool1d(x, 1)[:, :, 0]  # for vit
        elif self.model_name == 'resnet18':
            x = x.reshape(-1, x.shape[1] * 8, x.shape[2] * x.shape[3])  # for resnet
            x = F.adaptive_avg_pool1d(x, 1)[:, :, 0]
        fc_out = self.phase_fc(fc_i)
        x = self.phase_norm(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = x + self.phase_sequence_pos_embed
        x = self.phase_sequence_block(x)
        x = self.phase_sequence_norm(x)
        cls = x[:, 0, :]
        # x = F.adaptive_avg_pool1d(x.permute(0,2,1), 1)[:, :, 0]
        x = self.phase_sequence_fc(cls)
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
