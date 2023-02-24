import torch.nn as nn
import torch
import sys
sys.path.append("code/pytorch-image-models")
from timm_1.models.vision_transformer import VisionTransformer, _cfg, base_patch16_384_extractor
from timm_1.models.registry import register_model
from timm_1.models.layers import trunc_normal_
import math
import torch.nn.functional as F
from timm_1.models.resnet import resnet34
# from .vit_central_patch_extractor import *

class HybridEmbed(nn.Module):
    def __init__(self, backbone, img_size=96, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (img_size, img_size)
        self.img_size = img_size
        self.backbone = backbone
        o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
        if isinstance(o, (list, tuple)):
            o = o[-1]  
        feature_size = o.shape[-2:]
        feature_dim = o.shape[1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)
    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1] 
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        model = base_patch16_384_extractor(pretrained=True)
        backbone = resnet34(pretrained=True)
        backbone = torch.nn.Sequential(*(list(backbone.children())[0:8]))
        backbone[-1][-1] = torch.nn.Sequential(*(list(backbone[-1][-1].children())[0:-2]))
        model.patch_embed = HybridEmbed(backbone, img_size=96, embed_dim=768)
        pos_embed_old = model.pos_embed
        pos_embed_new = nn.Parameter(torch.zeros(1, model.patch_embed.num_patches + 1, model.embed_dim))
        ntok_new = pos_embed_new.shape[1]
        posemb_token, posemb_grid = pos_embed_old[:, :1], pos_embed_old[0, 1:]
        ntok_new -= 1
        gs_old = int(math.sqrt(len(posemb_grid)))
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(3, 3), mode='bilinear', align_corners=True)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, 9, -1)
        posemb = torch.cat([posemb_token, posemb_grid], dim=1)
        assert pos_embed_new.shape == posemb.shape
        model.pos_embed = nn.Parameter(posemb)
        self.part_cnn = model
    def forward(self, x):
        x, attn_w = self.part_cnn(x)
#         print("Meo meo hello world: ", x.size())
#         print("Gau gau hello world: ", attn_w[-1].size())
        return x, attn_w

