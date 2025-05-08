import torch
from diffusers.utils.torch_utils import randn_tensor

from typing import Any, Callable, Dict, List, Optional, Union


def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def flux_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

@torch.no_grad()
def latent_sample(transformer, scheduler, batch_size, num_channels_latents, height, width, prompt_embeds, pooled_prompt_embeds, text_ids, guidance, timesteps, vae_scale_factor, latents=None, return_attn=False):
    """
        Sample the model
        ESD quick_sample_till_t
    """

    height = int(height) // 8  # self.vae_scale_factor
    width = int(width) // 8    # self.vae_scale_factor
    shape = (batch_size, num_channels_latents, height, width)
    
    # (A) generate random tensor
    if latents is None:
        latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)
    # print(latents.shape)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device, torch.bfloat16)
    
    # (B) retrieve prompt embed

    # (C) generate latents w.r.t text embedding
    scheduler.set_train_timesteps(timesteps, device=transformer.device)
    timesteps = scheduler.timesteps
    
    latents = latents.to(transformer.device).bfloat16()
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    text_ids = text_ids.bfloat16()
    
    attn_map_lst = []
    # Denoising loop
    for i, t in enumerate(timesteps):

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(torch.bfloat16)
        
        # print(latents.shape, timestep)
        # self.transformer.config.guidance_embeds False => guidance = None
        noise_pred, attn_maps = transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        attn_map_lst.append(attn_maps) 
        
    if return_attn:
        return latents, latent_image_ids, attn_map_lst
    else:
        return latents, latent_image_ids


def predict_noise(transformer, latent_code, prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids, guidance, timesteps, CPU_only=False):
    """
        ESD (apply_model)
    """
    
    if CPU_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:1")
    
    # print("PE 20241127",text_ids.shape, latent_image_ids.shape)
    
    model_pred, _ = transformer(
                    hidden_states=latent_code.to(device),
                    timestep= (timesteps / 1000).to(device),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds.to(device),
                    encoder_hidden_states=prompt_embeds.to(device),
                    txt_ids=text_ids.to(device),
                    img_ids=latent_image_ids.to(device),
                    return_dict=False,
                )
    
    # print("20241127 predict noise e0 en ep", model_pred.device, model_pred.shape)
    
    model_pred = flux_unpack_latents(
        model_pred,
        height=512,
        width=512,
        vae_scale_factor=8,
    )

    return model_pred