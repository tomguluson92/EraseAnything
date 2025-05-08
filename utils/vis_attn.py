# coding: UTF-8
import torch
from matplotlib import pyplot as plt

def save_attn_map(out_attn_maps, iters):
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
            plt.savefig("/openbayes/home/sam_proj/vis/heads_idx{0}_iter{1}.png".format(bbb, iters))
            plt.close()