import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from einops import rearrange

class FeatureRefineLoss(nn.Module):
    """Featrue refinement loss."""
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    
    def forward(self, feat_refine, gt_feat):
        loss = F.mse_loss(feat_refine, gt_feat, reduction='mean')
        return loss * self.loss_weight
    

class RefineModule(nn.Module):
    def __init__(self, channels):
        super(RefineModule, self).__init__()
        feature_size =  channels // 4
        num_heads = 4
        num_enc_layers = 2
        num_dec_layers = 1
        
        # hyper-parameters
        self.in_channels = channels
        self.out_channels = channels
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        
        # input projection
        self.input_proj = nn.Conv2d(channels, feature_size, kernel_size=1)
        
        # # Transformer encoder layers
        # self.encoder_layers = nn.ModuleList([
        #     nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads)
        #     for _ in range(num_enc_layers)
        # ])
        
        
        # # Transformer decoder layers， the output channel should be half of the input
        # self.decoder_layers = nn.ModuleList([
        #     nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads)
        #     for _ in range(num_dec_layers)
        # ])

        # # feature decomposition module
        # self.feature_decomposer = nn.Sequential(
        #     nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1), 
        #     nn.ReLU(),
        #     nn.Conv2d(feature_size, feature_size * 2, kernel_size=3, padding=1)  
        # )
        # self.cross_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4)
        # self.fusion_layer = nn.Conv2d(channels, channels, kernel_size=1) 
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            num_layers=num_enc_layers + num_dec_layers,
            norm=nn.LayerNorm(feature_size)
            )
        
        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads),
            num_layers=num_dec_layers,
            norm=nn.LayerNorm(feature_size)
            )

        # channel fusion to make [10, 256, 60, 108] to [10, 128, 60, 108]
        self.channel_fusion = nn.Conv2d(feature_size, feature_size*2, kernel_size=1)

        # output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size * 2, channels // 2, kernel_size=1)
        )
    
    def forward(self, local_feat, local_corr_feat):
        """
        Forward function of feat_refine Module.
        Arguments:
        - local_feat (torch.Tensor): local feature tensor of shape (b, l_t, c, h, w)
        - local_corr_feat (torch.Tensor): local correlation feature tensor of shape (b, l_t, c, h, w)
        Returns:
        - refined_feat (torch.Tensor): refined feature tensor of shape (b, l_t, c, h, w)
        """
        b, l_t, c, h, w = local_feat.size()
        local_feat = local_feat.reshape(b*l_t, c, h, w)
        local_corr_feat = local_corr_feat.reshape(b*l_t, c, h, w)
        # print('before refine, local feat:', local_feat.shape)
        # print('before refine, local_corr_feat:', local_corr_feat.shape)
        extra_feat = torch.cat([local_feat, local_corr_feat], dim=1)
        # print('before refine, extra_feat:', extra_feat.shape)

        # input projection
        feat = self.input_proj(extra_feat)
        
        # feat = self.feature_decomposer(feat)
        # feat_var, feat_inv = torch.chunk(feat, 2, dim=1)
        # feat_var_flat = feat_var.view(b, c, -1).permute(2, 0, 1)  # (HW, B, C)
        # feat_inv_flat = feat_inv.view(b, c, -1).permute(2, 0, 1)
        # feat, _ = self.cross_attention(feat_var_flat, feat_inv_flat, feat_inv_flat)
        # feat = feat.permute(1, 2, 0).view(b, c, h, w)
        # feat = self.fusion_layer(feat_inv + feat)
        
        
        # rearrange feature map
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        # pass through Transformer encoder and decoder
        # for i in range(self.num_enc_layers):
        #     feat = self.encoder_layers[i](feat)
        # for i in range(self.num_dec_layers):
        #     feat = self.decoder_layers[i](feat)
        feat = self.encoder(feat)
        memory = feat
        feat = self.decoder(
            tgt=feat,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # rearrange feature map
        feat = rearrange(feat, 'b (h w) c -> b c h w', h=h, w=w)

        # print("before skip connection, feat.size:", feat.shape)
        # print("before skip connection, local_feat.size:", local_feat.shape)
        # channel fusion
        feat = self.channel_fusion(feat)
        # print("after channel_fusion, feat.size:", feat.shape)

        # skip connection
        feat = feat + local_feat
        
        # project to output channels
        feat = self.output_proj(feat)
        # print('after output_proj, feat.size:', feat.shape)
        feat = feat.reshape(b, l_t, c, h, w)
        # print('after refine:', feat.shape)
        return feat
    
    def train_forward(self, local_feat, local_corr_feat, gt_local_feat):
        """
        Forward function of feat_refine Module.
        Arguments:
        - local_feat (torch.Tensor): local feature tensor of shape (b, l_t, c, h, w)
        - local_corr_feat (torch.Tensor): local correlation feature tensor of shape (b, l_t, c, h, w)
        Returns:
        - refined_feat (torch.Tensor): refined feature tensor of shape (b, l_t, c, h, w)
        """
        b, l_t, c, h, w = local_feat.size()
        local_feat = local_feat.reshape(b*l_t, c, h, w)
        local_corr_feat = local_corr_feat.reshape(b*l_t, c, h, w)
        extra_feat = torch.cat([local_feat, local_corr_feat], dim=1)

        gt_b, gt_l_t, gt_c, gt_h, gt_w = gt_local_feat.size()
        gt_local_feat = gt_local_feat.reshape(gt_b*gt_l_t, gt_c, gt_h, gt_w)
        gt_extra_feat = torch.cat([gt_local_feat, gt_local_feat], dim=1)
        
        # input projection
        feat = self.input_proj(extra_feat)
        gt_feat = self.input_proj(gt_extra_feat)
        
        feat = self.feature_decomposer(feat)
        feat_var, feat_inv = torch.chunk(feat, 2, dim=1)
        feat_var_flat = feat_var.view(b, c, -1).permute(2, 0, 1)  # (HW, B, C)
        feat_inv_flat = feat_inv.view(b, c, -1).permute(2, 0, 1)
        feat, _ = self.cross_attention(feat_var_flat, feat_inv_flat, feat_inv_flat)
        feat = feat.permute(1, 2, 0).view(b, c, h, w)
        feat = self.fusion_layer(feat_inv + feat)
        
        gt_feat = self.feature_decomposer(gt_feat)
        _, gt_feat_inv = torch.chunk(gt_feat, 2, dim=1)
        gt_feat = gt_feat_inv.view(gt_b, gt_l_t, gt_c, -1)
        
        # rearrange feature map
        feat = rearrange(feat, 'b c h w -> b (h w) c')
        # pass through Transformer encoder and decoder
        # for i in range(self.num_enc_layers):
        #     feat = self.encoder_layers[i](feat)
        # for i in range(self.num_dec_layers):
        #     feat = self.decoder_layers[i](feat)
        feat = self.encoder(feat)
        memory = feat
        feat = self.decoder(
            tgt=feat,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # rearrange feature map
        feat = rearrange(feat, 'b (h w) c -> b c h w', h=h, w=w)

        # print("before skip connection, feat.size:", feat.shape)
        # print("before skip connection, local_feat.size:", local_feat.shape)
        # channel fusion
        feat = self.channel_fusion(feat)
        # print("after channel_fusion, feat.size:", feat.shape)

        # skip connection
        feat = feat + local_feat
        
        # project to output channels
        feat = self.output_proj(feat)
        # print('after output_proj, feat.size:', feat.shape)
        feat = feat.reshape(b, l_t, c, h, w)
        # print('after refine:', feat.shape)
        return feat, gt_feat
