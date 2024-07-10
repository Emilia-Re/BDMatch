# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from semilearn.os_algorithms.hooks import MaskingHook
from semilearn.core.hooks import Hook
    
class BDMatchFixedThresholingHook(MaskingHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if softmax_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, pseudo_label = torch.max(probs_x_ulb, dim=-1)
        pos_mask = max_probs.ge(algorithm.p_cutoff_pos).unsqueeze(1)
        neg_mask = probs_x_ulb.le(algorithm.p_cutoff_neg)
        mask = (pos_mask | neg_mask).to(max_probs.dtype)
            
        pseudo_label = F.one_hot(pseudo_label.long(), num_classes=algorithm.num_classes).to(max_probs.dtype)
        pseudo_label *= pos_mask.to(max_probs.dtype)
        
        return mask, pseudo_label
    
class LAEMAHook(Hook):
    def __init__(self, num_classes, momentum=0.999):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        
        # p_model
        self.p_model = torch.ones((self.num_classes)) * 0.5
    
    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb):
        # check device
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        
    @torch.no_grad()
    def get_logits_adj(self, algorithm, probs_x_ulb):
        self.update_p(algorithm, probs_x_ulb)
        logits_adj = torch.log(self.p_model + 1e-8) - torch.log(1.0 - self.p_model + 1e-8)
        return logits_adj
