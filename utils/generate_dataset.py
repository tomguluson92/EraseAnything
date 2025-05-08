# coding: UTF-8
"""
    @date:  2024.11.25  week48  Monday
    @func:  dataset generation.  
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import FluxPipeline
import torch
from matplotlib import pyplot as plt

from huggingface_hub import login
# login(token="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")

model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
model.to("cuda:0")

prompt_lst = ["a nude girl with beautiful hair and her breast open to see"]

os.makedirs("../image", exist_ok=True)

for idx in range(len(prompt_lst)):
    out, out_attn_maps = model(
        prompt=prompt_lst[idx],
        guidance_scale=0.0,
        height=512,
        width=512,
        num_inference_steps=28,
        max_sequence_length=256,
        num_images_per_prompt=1,
        return_dict=False
    )
    
    # import pdb; pdb.set_trace()
    for jdx, item in enumerate(out):
        item.save("/openbayes/home/sam_proj/vis/test_{}.png".format(jdx))
    
    
    for jdx, attn_map in enumerate(out_attn_maps):
        for bbb in range(attn_map.size(-1)):
            fig, axs = plt.subplots(6, 4, figsize=(32, 32))
            for head in range(24):
                # import pdb; pdb.set_trace()
                attn_sub = attn_map[head, 256:, bbb]   #attn_map.mean(0)
                attn_sub = attn_sub.reshape(32, 32).float()
                attn_sub = torch.flip(attn_sub, [0])
                attn_sub_np = attn_sub.detach().cpu().numpy()
                row = head // 4
                col = head % 4
                
                im = axs[row, col].imshow(attn_sub_np, cmap='viridis', origin='lower')
                axs[row, col].axis('off')  # 关闭坐标轴
                axs[row, col].set_title(f'Head {head}')
                fig.colorbar(im, ax=axs[row, col], orientation='vertical')
            
            plt.tight_layout()
            plt.savefig("/openbayes/home/sam_proj/vis/combined_heads_idx{}.png".format(bbb))
            plt.close()
    
