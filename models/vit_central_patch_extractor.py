import sys
sys.path.append("../code/")
from pytorch-image-models.timm.models.vision_transformer import VisionTransformer, _cfg
from pytorch-image-models.timm.models.registry import register_model
from pytorch-image-models.timm.models.layers import trunc_normal_
import math
from pytorch-image-models.timm.models.resnet import resnet34
import numpy as np

from email.policy import strict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

import cv2
import random

class VisionTransformer_Extractor(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.output1 = nn.Sequential(
#             nn.ReLU(),
            nn.Linear(768, 256)
        )
        self.output2 = nn.Sequential(
#             nn.ReLU(),
            nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)
    def forward_features(self, x):
        B = x.shape[0]
#         print("Batch: ", B)
        x = self.patch_embed(x)
#         print("Patch emb size: ", x.size())
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x1 = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
#         print("Output after module: ", x.size())
        for blk in self.blocks:
#             print(blk)
            x = blk(x)
#             print("Tmp out: ", x.size())
        x = x + x1
        x = self.norm(x)
        x = x[:, 1:]
        return x
    def forward(self, x):
        x = self.forward_features(x)
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        x = self.output1(x)
        x = x.reshape(batch_size, -1, x.shape[-1])
        x = x[:, 4]
        x = self.output2(x)
        return x

@register_model
def base_patch16_384_extractor(pretrained=False, **kwargs):
    model = VisionTransformer_Extractor(
        img_size=384, 
        patch_size=32, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('../code/pretrained_vit/jx_vit_base_p32_384-830016f5.pth')
        model.load_state_dict(checkpoint, strict=False)
    return model

