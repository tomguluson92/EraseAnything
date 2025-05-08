# coding: UTF-8
"""
    @func: loss + bi-level + InfoNCE
"""

import random
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from .esd_utils import latent_sample, predict_noise
from .infoNCE import calculate_steer_loss


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(dtype=dtype) # [1000]
    schedule_timesteps = noise_scheduler.timesteps
    import pdb; pdb.set_trace()
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def calculate_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1, lamb3=0.2, opt_name="ESD"):
    
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    if opt_name == "ESD":
        print("OPT Name", opt_name)
    # (ESD) get conditional embedding for the prompt
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                "", text_encoders, tokenizers
            )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                prompts, text_encoders, tokenizers
            )

    # (ESD) ddim_steps
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    # (ESD) start_guidance = 3
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor)
        # e_0 & e_p
        e_0 = predict_noise(transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
        e_p = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)

    # get conditional score from ESD model
    e_n = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    total_loss = []
    
    loss_esd = criteria(e_n.to(transformer.device), e_0.to(transformer.device) - (negative_guidance*(e_p.to(transformer.device) - e_0.to(transformer.device))))
    
    total_loss.append(loss_esd)
    
    if opt_name == "ESD+":
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                transformer.device,
                weight_dtype,
            )
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme=args.weighting_scheme,
        #     batch_size=bsz,
        #     logit_mean=args.logit_mean,
        #     logit_std=args.logit_std,
        #     mode_scale=args.mode_scale,
        # )
        # indices = (u * noise_scheduler.config.num_train_timesteps).long()
        # timesteps = noise_scheduler.timesteps[indices].to(device=transformer.device)
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        noisy_model_input = noise_scheduler.add_noise(model_input,
                                                      noise,
                                                      t_enc_ddpm)
        
        packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )
        
        if transformer.config.guidance_embeds:
            guidance = torch.tensor([args.guidance_scale], device=transformer.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None
        
        # import pdb; pdb.set_trace()
        
        remove_indices = batch['remove_indices'][0]
        
        model_pred, attn_maps = transformer(
            hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
            pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
            encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
            txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
            img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
            return_dict=False,
        )[0:2]
        
        attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
        attn_map_mask[..., remove_indices] = 0
        attn_map_mask = 1 - attn_map_mask
        # import pdb; pdb.set_trace()
        
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        # flow matching loss
        target = noise - model_input
        
        # Compute regular loss.
        loss_attn = lamb2 * sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()
        loss_lora = lamb3 * torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )[0]
        
        total_loss.append(loss_attn)
        total_loss.append(loss_lora)
            
    return total_loss


def calculate_upper_ca_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, ca_prompt_p, ca_prompt_0, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1):
    """
        @date: 2024.12.14 
        @name: bi-level loss
        @func: replace esd with ca
    """
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                    ca_prompt_0, text_encoders, tokenizers
                )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                ca_prompt_p, text_encoders, tokenizers
        )

    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0],
        model_input.shape[2] // 2,
        model_input.shape[3] // 2,
        transformer.device,
        weight_dtype,
    )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

    
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
    
    # import pdb; pdb.set_trace()
    
    model_pred, attn_maps = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]

    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )
    
    total_loss = []

    remove_indices = batch['remove_indices'][0]
    
    with torch.no_grad():
        model_pred_sg = transformer(
            hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
            timestep=t_enc_ddpm / 1000,
            guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
            pooled_projections=pooled_emb_0.to(dtype=weight_dtype, device=transformer.device),
            encoder_hidden_states=emb_0.to(dtype=weight_dtype, device=transformer.device),
            txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
            img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
            return_dict=False,
        )[0]

        model_pred_sg = FluxPipeline._unpack_latents(
            model_pred_sg,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

    loss_ca = lamb1 * torch.mean(
        ((model_pred.float() - model_pred_sg.float()) ** 2).reshape(model_pred_sg.shape[0], -1),
        1,
    )[0]
    
    total_loss.append(loss_ca)

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask

    # Compute regular loss.
    loss_attn = sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()

    total_loss.append(loss_attn)
            
    return total_loss, t_enc_ddpm



def calculate_upper_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, neg_prompts, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1):
    """
        @date: 2024.11.30 
        @name: bi-level loss
        @func: a) upper_loss: make sure samples from D_un(unlearning) is removed.
               b) lower_loss: make sure samples from D_ir(irrelevant) is perserved.
    """
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # (ESD) get conditional embedding for the prompt
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                neg_prompts, text_encoders, tokenizers
            )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                prompts, text_encoders, tokenizers
            )

    # (ESD) ddim_steps
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    # (ESD) start_guidance = 3
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor)
        # e_0 & e_p
        e_0 = predict_noise(transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
        e_p = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)

    # get conditional score from ESD model
    e_n = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    total_loss = []
    
    loss_esd = criteria(e_n.to(transformer.device), e_0.to(transformer.device) - (negative_guidance*(e_p.to(transformer.device) - e_0.to(transformer.device))))
    
    total_loss.append(loss_esd)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            transformer.device,
            weight_dtype,
        )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None

    remove_indices = batch['remove_indices'][0]

    model_pred, attn_maps = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask
    # import pdb; pdb.set_trace()

    # Compute regular loss.
    loss_attn = sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()

    total_loss.append(loss_attn)
            
    return total_loss, t_enc_ddpm


def calculate_lower_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, t_enc_ddpm, start_guidance=3, ddim_steps=28, K=3, ir_concept_lst=[]):
    """
        @date: 2024.11.30 
        @name: bi-level loss
        @func: a) upper_loss: make sure samples from D_un(unlearning) is removed.
               · ESD 
               · attn map deactivation
               b) lower_loss: make sure samples from D_ir(irrelevant) is perserved.
               · lora loss (low med high timesteps)
               · InfoNCE loss
    """
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Convert images to latent space
    if args.cache_latents:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                transformer.device,
                weight_dtype,
            )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
            prompts, text_encoders, tokenizers
        )
        
    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None
        
    model_pred, attn_maps = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]
        
    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    # flow matching loss
    target = noise - model_input

    loss_lora = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )[0]
    
    total_loss = []
    total_loss.append(loss_lora)
    
    # one negtive sample (synonym) + K positive sample (irrelevant)
    start_code = torch.randn_like(model_input)
    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # negtive sample: emb_neg
    emb_neg, pooled_emb_neg, text_ids_neg = compute_text_embeddings(
            batch["synonym_words"], text_encoders, tokenizers
        )
    
    with torch.no_grad():
        _, _, attn_map_lst_neg = latent_sample(transformer,
                                               noise_scheduler,
                                               1,
                                               model_input.shape[1], 
                                               512,
                                               512,
                                               emb_neg.to(transformer.device),
                                               pooled_emb_neg.to(transformer.device),
                                               text_ids_neg.to(transformer.device),
                                               start_guidance, 
                                               int(ddim_steps),
                                               vae_scale_factor,
                                               latents=start_code,
                                               return_attn=True)
    
    # irrelevant sample: emb_pos
    if len(ir_concept_lst) != K:
        raise Exception("请检查ir_concept_lst")
    
    # random_attn_map
    # exp1: 20, 27
    # exp2: 14, 27
    # exp3: 0, 27
    attn_map_rand_idx = random.randint(0, int(ddim_steps)-1)
    
    pos_lst = []
    for idx in range(K): 
        
        emb_pos, pooled_emb_pos, text_ids_pos = compute_text_embeddings(
                ir_concept_lst[idx], text_encoders, tokenizers
            )
        _, _, attn_map_lst_pos_sub = latent_sample(transformer,
                                                   noise_scheduler,
                                                   1,
                                                   model_input.shape[1], 
                                                   512,
                                                   512,
                                                   emb_pos.to(transformer.device),
                                                   pooled_emb_pos.to(transformer.device),
                                                   text_ids_pos.to(transformer.device),
                                                   start_guidance, 
                                                   int(ddim_steps),
                                                   vae_scale_factor,
                                                   latents=start_code,
                                                   return_attn=True)

        tmp_attn_pos = attn_map_lst_pos_sub[attn_map_rand_idx]
        pos_lst.append(tmp_attn_pos)
        
    attn_map_neg = attn_map_lst_neg[attn_map_rand_idx]
    attn_map_pos = pos_lst
    
    info_neg = attn_map_neg[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    info_pos_lst = []
    
    for idx in range(K):
        info_pos = pos_lst[idx][..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
        info_pos_lst.append(info_pos)
    
    info_center = attn_maps[..., batch['remove_indices'][0]][:, 0, ...].permute(0, 2, 1)
    
    loss_contrastive = calculate_steer_loss(info_center,
                                            info_neg,
                                            info_pos_lst,
                                            temperature=0.07)
    
    total_loss.append(loss_contrastive)
    return total_loss