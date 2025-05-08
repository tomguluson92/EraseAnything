# coding: UTF-8
"""
    @date:    2024.11.30-12.2
    @ref:     https://github.com/ziqihuangg/ReVersion/blob/master/train.py#L537
"""


import os
import numpy as np

import torch
import torch.nn.functional as F

def calculate_steer_loss(unlearn_attn,
                         neg_attn,
                         irr_attn_lst,
                         temperature=0.07,
                         method="mean"):
    """L_steer"""
    # unlearn_attn   [bs, length, 1280] (nude)                       requires_grad = False
    # neg_attn       [bs, length, 1280] (同义词: naked, nudity, ...) requires_grad = False
    # irr_attn_lst   [[bs, length1, 1280], [bs, length2, 1280], ...],  requires_grad = True
    
    if method == "mean":
        unlearn_attn = unlearn_attn.mean(1).unsqueeze(1)
        neg_attn = neg_attn.mean(1).unsqueeze(1)
        irr_attn = None
        for item in irr_attn_lst:
            if irr_attn is None:
                irr_attn = item.mean(1).unsqueeze(1)
            else:
                irr_attn = torch.cat([irr_attn, item.mean(1).unsqueeze(1)], dim=1)
    # unlearn_attn [bs, 1, 1280]
    # irr_attn     [bs, N, 1280] 
        
    # stack positives(unlearn) and negatives(irrelevant) as a pn_block
    pn_embeds = torch.cat([unlearn_attn, irr_attn], dim=1)
    pn_embeds_normalized = F.normalize(
        pn_embeds, 
        p=2,
        dim=2)
    
    # compute malicious embeds
    neg_attn_normalized = F.normalize(neg_attn, p=2, dim=2)
    
    # print(neg_attn_normalized.shape, pn_embeds_normalized.shape)
    # compute Multi-Instance InfoNCE loss
    logits = torch.einsum('bnc,bmc->bnm',
                          [neg_attn_normalized, pn_embeds_normalized
                           ])  # (1, 1+N)
    
    # TODO: .
    # ref: https://www.zhihu.com/search?type=content&q=infoNCE
    logits /= temperature
    nominator = torch.logsumexp(logits[:, :, :neg_attn.shape[1]], dim=(1,2))
    denominator = torch.logsumexp(logits, dim=(1,2))
    
    return torch.mean(nominator - denominator)

    # ours(2024.12.13)
    # return torch.mean(denominator - nominator)
    # standard
    # return torch.mean(denominator - nominator)
    
    
if __name__ == "__main__":
    unlearn_attn = torch.randn(1, 2, 1280)
    neg_attn = torch.randn(1, 3, 1280)
    # K = ?
    irr_attn_lst = [torch.randn(1, 3, 1280), torch.randn(1, 4, 1280), torch.randn(1, 1, 1280)] # [bs, 1280, 2]
    aaa = calculate_steer_loss(unlearn_attn,
                         neg_attn,
                         irr_attn_lst)
    import pdb; pdb.set_trace()