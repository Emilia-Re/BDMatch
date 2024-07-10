# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math

from semilearn.core.algorithmbase import OSAlgorithmBase
from semilearn.core.utils import OS_ALGORITHMS
from semilearn.os_algorithms.utils import SSL_Argument, str2bool
from .utils import BDMatchFixedThresholingHook, LAEMAHook

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, query_embed):
        B, N, C = x.shape
        K = query_embed.size(1)
        
        q = self.linear_q(query_embed).expand(B, -1, -1).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, K, C)
        x = self.proj_drop(x)
        return x
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class BDMatch_Net(nn.Module):
    def __init__(self, base, num_classes, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.,
                 init_values=None, drop_path=0., use_rot=False):
        super(BDMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        
        # Multi-head dot-product attention module to extract label-specific features
        self.attn = Attention(self.num_features, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(self.num_features, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(self.num_features, eps=1e-6)
        
        self.query_embed = nn.Parameter(torch.zeros(1, num_classes, self.num_features))
        nn.init.normal_(self.query_embed)
        
        self.classifier2 = nn.Linear(self.num_features, num_classes)
        nn.init.xavier_normal_(self.classifier2.weight.data)
        self.classifier2.bias.data.zero_()
        
        # Standard classifiers used in the dual-branch architecture
        self.st_classifier1 = nn.Linear(self.num_features, num_classes)
        nn.init.xavier_normal_(self.st_classifier1.weight.data)
        self.st_classifier1.bias.data.zero_()
        self.st_classifier2 = nn.Linear(self.num_features, num_classes)
        nn.init.xavier_normal_(self.st_classifier2.weight.data)
        self.st_classifier2.bias.data.zero_()
        
        if use_rot:
            self.rot_classifier = nn.Linear(self.num_features, 4, bias=False)
            nn.init.xavier_normal_(self.rot_classifier.weight.data)
    
    def forward(self, x, st_pred=False):
        feat = self.backbone.extract(x)
        out = F.adaptive_avg_pool2d(feat, 1)
        out = out.view(-1, 1, self.num_features)
        feat = feat.reshape((feat.size(0), feat.size(1), -1)).permute(0, 2, 1)
        feat = out + self.drop_path1(self.ls1(self.attn(feat, self.query_embed)))
        feat = self.norm(feat)
        
        logits = self.head_forward(feat)
        if not st_pred:
            return {'logits':logits, 'feat':feat}
        st_logits = self.head_forward_st(feat)
        return {'logits':logits, 'st_logits':st_logits, 'feat':feat}
    
    def head_forward(self, x):
        logits = torch.sum(x * self.backbone.classifier.weight, dim=-1) + self.backbone.classifier.bias
        logits2 = torch.sum(x * self.classifier2.weight, dim=-1) + self.classifier2.bias
        return logits - logits2
    
    def head_forward_st(self, x):
        logits = torch.sum(x * self.st_classifier1.weight, dim=-1) + self.st_classifier1.bias
        logits2 = torch.sum(x * self.st_classifier2.weight, dim=-1) + self.st_classifier2.bias
        return logits - logits2
    
    def rot_forward(self, x):
        feat = self.backbone(x, only_feat=True)
        logits = self.rot_classifier(feat)
        return logits

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher
    
    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd

@OS_ALGORITHMS.register('bdmatch')
class BDMatch(OSAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, p_cutoff_pos=args.p_cutoff_pos, p_cutoff_neg=args.p_cutoff_neg, hard_label=args.hard_label)
        
        self.ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.st_ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def init(self, T, p_cutoff_pos, p_cutoff_neg, hard_label=True):
        self.T = T
        self.p_cutoff_pos = p_cutoff_pos
        self.p_cutoff_neg = p_cutoff_neg
        self.use_hard_label = hard_label
        self.sup_remargin = - self.args.tau * math.log(self.num_classes - 1)
    
    def set_hooks(self):
        self.register_hook(BDMatchFixedThresholingHook(), "MaskingHook")
        self.register_hook(LAEMAHook(num_classes=self.num_classes,
                                     momentum=self.args.ema_p), "LAHook")
        super().set_hooks()
        
    def set_model(self):
        model = super().set_model()
        model = BDMatch_Net(model, self.num_classes, num_heads=self.args.feat_num_heads,
                            drop_path=self.args.feat_drop_path_ratio, use_rot=self.args.use_rot)
        return model
        
    def compute_prob(self, logits):
        return torch.sigmoid(logits)
    
    def compute_sup_loss(self, logits, targets):
        return F.multilabel_soft_margin_loss(logits, F.one_hot(targets.long(), num_classes=self.num_classes), reduction='mean')

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs, st_pred=True)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                st_logits_x_lb = outputs['st_logits'][:num_lb]
                st_logits_x_ulb_w, st_logits_x_ulb_s = outputs['st_logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb, st_pred=True) 
                logits_x_lb = outs_x_lb['logits']
                st_logits_x_lb = outs_x_lb['st_logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s, st_pred=True)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                st_logits_x_ulb_s = outs_x_ulb_s['st_logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w, st_pred=True)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    st_logits_x_ulb_w = outs_x_ulb_w['st_logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            
            
            ### Loss on the balanced branch of the dual-branch architecture
            # compute re-margining loss on labeled data
            sup_loss = self.ce_loss(logits_x_lb + self.sup_remargin,
                                    F.one_hot(y_lb.long(), num_classes=self.num_classes).float()).mean() * self.num_classes
            
            # obtain the adaptive proxy of potential class distribution of each binary unlabeled set
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            logits_adj = self.call_hook("get_logits_adj", "LAHook", probs_x_ulb=probs_x_ulb_w)
            
            adjusted_probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach() - self.args.tau * logits_adj)
            
            # compute mask (B x C) and generate pseudo label (B x C)
            mask, pseudo_label = self.call_hook("masking", "MaskingHook", logits_x_ulb=adjusted_probs_x_ulb_w, softmax_x_ulb=False)
            
            # compute consistency regularization on unlabeled data
            unsup_loss = self.ce_loss(logits_x_ulb_s, pseudo_label)
            unsup_loss = torch.mean(unsup_loss * mask) * self.num_classes
            
            
            ### Loss on the standard branch of the dual-branch architecture
            # compute binary cross-entropy loss on labeled data
            st_sup_loss = self.st_ce_loss(st_logits_x_lb,
                                          F.one_hot(y_lb.long(), num_classes=self.num_classes).float()).mean() * self.num_classes
            
            st_probs_x_ulb_w = self.compute_prob(st_logits_x_ulb_w.detach())
            
            # compute mask (B x C) and generate pseudo label (B x C)
            st_mask, st_pseudo_label = self.call_hook("masking", "MaskingHook", logits_x_ulb=st_probs_x_ulb_w, softmax_x_ulb=False)
            
            # compute consistency regularization on unlabeled data
            st_unsup_loss = self.st_ce_loss(st_logits_x_ulb_s, st_pseudo_label)
            st_unsup_loss = torch.mean(st_unsup_loss * st_mask) * self.num_classes
            
            # self-supervised rotation recognition task
            if self.args.use_rot:
                x_ulb_r = torch.cat(
                    [torch.rot90(x_ulb_w[:num_lb], i, [2, 3]) for i in range(4)], dim=0)
                y_ulb_r = torch.cat(
                    [torch.empty(x_ulb_w[:num_lb].size(0)).fill_(i).long() for i in range(4)], dim=0).to(x_ulb_r.device)
                self.bn_controller.freeze_bn(self.model)
                logits_rot = self.model.rot_forward(x_ulb_r)
                self.bn_controller.unfreeze_bn(self.model)
                rot_loss = F.cross_entropy(logits_rot, y_ulb_r, reduction='mean')
            else:
                rot_loss = torch.tensor(0).to(x_ulb_r.device)
            
            if self.epoch > 0:
                total_loss = sup_loss + self.lambda_u * unsup_loss + st_sup_loss + self.lambda_u * st_unsup_loss + rot_loss
            else:
                total_loss = sup_loss + st_sup_loss + rot_loss
        
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         st_sup_loss=st_sup_loss.item(),
                                         st_unsup_loss=st_unsup_loss.item(),
                                         rot_loss=rot_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         st_util_ratio=st_mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff_pos', float, 0.99),
            SSL_Argument('--p_cutoff_neg', float, 0.01),
            SSL_Argument('--feat_num_heads', int, 4),
            SSL_Argument('--feat_drop_path_ratio', float, 0.2),
        ]
