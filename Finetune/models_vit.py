# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import torch.nn.functional as F


### Yifan ###
from segm.model.blocks import Block, FeedForward
from segm.model.utils import init_weights
from timm.models.layers import trunc_normal_
from einops import rearrange
from segm.model.utils import padding, unpadding
import matplotlib.pyplot as plt
from timm.models.vision_transformer import PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, decoder_n_layers, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        #self.patch_embed = self.PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = PatchEmbed(kwargs['img_size'], kwargs['patch_size'], 3, kwargs['embed_dim'])
        
        num_patches_pos = 256
        embed_dim = kwargs['embed_dim']
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_pos + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches_pos**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
               
        decoder_cfg = {}
        decoder_cfg["n_cls"] = 1                               # n_cls 
        decoder_cfg["d_encoder"] = kwargs['embed_dim']         # encoder.d_model
        decoder_cfg["patch_size"] = kwargs['patch_size']       # encoder.patch_size
        decoder_cfg["n_layers"] = decoder_n_layers           
        
                
        decoder_cfg["drop_path_rate"] = 0
        decoder_cfg["dropout"] = 0.1

        dim = kwargs['embed_dim']
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)

        self.Decoder = decoder


        self.bias_GGG = nn.Conv2d(1, 3, kernel_size=1, bias=True)  # add bias layer with # levels
        nn.init.constant_(self.bias_GGG.weight, 1)  # in bias layer, no weights are needed
        nn.init.constant_(self.bias_GGG.bias, 0)  # initialize with zero
        self.bias_GGG.weight.requires_grad = False  # only update bias
        
        self.bias_PIRADS = nn.Conv2d(1, 2, kernel_size=1, bias=True)  # add bias layer with # levels
        nn.init.constant_(self.bias_PIRADS.weight, 1)  # in bias layer, no weights are needed
        nn.init.constant_(self.bias_PIRADS.bias, 0)  # initialize with zero
        self.bias_PIRADS.weight.requires_grad = False  # only update bias



        self.bias_GGG_score = nn.Linear(1, 3, bias=True)  # add bias layer with # levels
        nn.init.constant_(self.bias_GGG_score.weight, 1)  # in bias layer, no weights are needed
        nn.init.constant_(self.bias_GGG_score.bias, 0)  # initialize with zero
        self.bias_GGG_score.weight.requires_grad = False  # only update bias
        
        self.bias_PIRADS_score = nn.Linear(1, 2, bias=True)  # add bias layer with # levels
        nn.init.constant_(self.bias_PIRADS_score.weight, 1)  # in bias layer, no weights are needed
        nn.init.constant_(self.bias_PIRADS_score.bias, 0)  # initialize with zero
        self.bias_PIRADS_score.weight.requires_grad = False  # only update bias
    
    
    def patchify_mask(self, imgs):
        
        
        """ Yifan
        imgs: (N, H, W)
        x: (N, 1, patch_size**2 *1)
        """
       
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x


        
    def unpatchify(self, x, number):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """

        #x = torch.einsum('ntl->nlt', x)
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, number))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], number, h * p, h * p))
                
        
        #imgs = F.logsigmoid(imgs).exp()
        #soft = torch.nn.Sigmoid()
        #imgs = soft(imgs)
   
        return imgs
        
    
    def forward_features(self, x, mask):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ### X [1,5x,64,64]
        batch_size = x.shape[0]

        slices = int(x.shape[1]/3)

        
        x_em = torch.empty([batch_size, 16*slices, 64]).to(device)
        for ss in range(slices):
            x_em[:,ss*16:(ss+1)*16] = self.patch_embed(x[:,ss*3:(ss+1)*3,:,:])
            
        #x = self.patch_embed(x)

        ##### Forward position embding #####
        mask = self.patchify_mask(mask)                  ### [bs,225,256]
        mask,max_indices = torch.max(mask, dim=2)        ### [bs,225]
                      
        mask = torch.unsqueeze(mask, 2)
        Multi_mask = mask.repeat(1,1,self.pos_embed.shape[-1])

        self.Multi_pos_embed = self.pos_embed[:, 1:, :].repeat(batch_size,1,1)   ### [bs,225,768]  
        masksss = Multi_mask == 1                                                ### [bs,225,768]  
        self.crop_pos_embed = self.Multi_pos_embed[masksss]  
        self.crop_pos_embed = torch.reshape(self.crop_pos_embed,(batch_size,16,self.pos_embed.shape[-1]))

        #x = x + self.crop_pos_embed
        x = torch.empty([batch_size, 16*slices, 64]).to(device)
        for ss in range(slices):
            x[:,ss*16:(ss+1)*16,:] = x_em[:,ss*16:(ss+1)*16,:] + self.crop_pos_embed


              
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        
        
        #self.Multi_pos_embed[masksss]
        #x = x + self.pos_embed[:, 1:, :]    ### [bs,16,768]
        #x = x + self.crop_pos_embed

        
        #cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #x = torch.cat((cls_tokens, x), dim=1)
        
        
        #x = x + self.pos_embed
        #x = x + self.crop_pos_embed
        
        x = self.pos_drop(x)

        
        for blk in self.blocks:
            x = blk(x)
        return x   ### [bs, 226, 768] for base
    
    
        '''
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

                    
        print('ddddddddddddddddd',outcome.shape)
        return outcome
        '''   

    def forward(self, x, mask):

        #input_feature = x        
        #patched_feature = x.flatten(2).transpose(1, 2)
        #print(patched_feature.shape)


        x = self.forward_features(x,mask)
        
        x = x[:, 1:, :]
        #x = self.head(x)
        #masks_GGG,masks_PIRADS,scores_GGG,scores_PIRADS = self.Decoder(x, (64, 64))

        Score = self.Decoder(x, (64, 64))

        #masks_GGG = masks_GGG[:,:16,:]
        #masks_PIRADS = masks_PIRADS[:,:16,:]
        
        #masks_GGG = self.unpatchify(masks_GGG,1)
        #masks_PIRADS = self.unpatchify(masks_PIRADS,1)

        #masks_GGG = self.bias_GGG(masks_GGG)
        #masks_PIRADS = self.bias_PIRADS(masks_PIRADS)

        #scores_GGG = self.bias_GGG_score(scores_GGG)
        #scores_PIRADS = self.bias_PIRADS_score(scores_PIRADS)

        #masks = F.interpolate(masks, size=(16, 16), mode="bilinear")
        #masks = unpadding(masks, (240, 240))
         


        #original_image = self.unpatchify(patched_feature,1)

        #plt.imshow(original_image[0,0,:,:], cmap='gray',vmin=0, vmax = 1)
        #plt.show()

        #return torch.cat((masks_GGG, masks_PIRADS), dim=1), scores_GGG,scores_PIRADS
        return Score


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        #self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        #self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.prediction_decoder_GGG = nn.Linear(d_model, patch_size**2 * 1 , bias=True)
        self.prediction_decoder_PIRADS = nn.Linear(d_model, patch_size**2 * 1 , bias=True)

        self.prediction_decoder_GGG_score = nn.Linear(d_model, 1 , bias=True)   ## Should be true? ##
        self.prediction_decoder_PIRADS_score = nn.Linear(d_model, 1 , bias=True)

        
        
        ### Only needed ###
        self.prediction_decoder_score = nn.Linear(d_model, 1 , bias=True)

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)   ### [bs, 226,768] for base size ###
        
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
    
        #patches = patches @ self.proj_patch
        #cls_seg_feat = cls_seg_feat @ self.proj_classes

        #patches = patches / patches.norm(dim=-1, keepdim=True)
        #masks_GGG = self.prediction_decoder_GGG(patches)
        #masks_PIRADS = self.prediction_decoder_PIRADS(patches)

        #scores_GGG = self.prediction_decoder_GGG_score(cls_seg_feat)
        #scores_PIRADS = self.prediction_decoder_PIRADS_score(cls_seg_feat)

        score = self.prediction_decoder_score(cls_seg_feat)
        
        #cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        #masks = patches @ cls_seg_feat.transpose(1, 2)
        #masks_GGG = self.mask_norm(masks_GGG)
        #masks_PIRADS = self.mask_norm(masks_PIRADS)

        
        #masks = rearrange(patches, "b (h w) n -> b n h w", h=int(GS))
        #return masks_GGG,masks_PIRADS,scores_GGG,scores_PIRADS   ### patch_size**2 * 5
        return score


    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)





def vit_tiny_tiny_patch16(**kwargs):
    model = VisionTransformer(decoder_n_layers=1,
        img_size=64, patch_size=16, embed_dim=64, depth=6, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


'''
def vit_tiny_tiny_patch16(**kwargs):
    model = VisionTransformer(decoder_n_layers=1,
        img_size=64, patch_size=16, embed_dim=64, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
'''

def vit_tiny_patch16_6(**kwargs):
    model = VisionTransformer(decoder_n_layers=2,
        img_size=64, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patch16_12(**kwargs):
    model = VisionTransformer(decoder_n_layers=2,
        img_size=64, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(decoder_n_layers=2,
        img_size=64, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




def vit_small_patch16(**kwargs):
    model = VisionTransformer(decoder_n_layers=2,
        img_size=64, patch_size=16, embed_dim=348, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def vit_base_patch16(**kwargs):
    model = VisionTransformer(decoder_n_layers=2, 
        img_size=64, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model