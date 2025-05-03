import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from huggingface_hub import hf_hub_download, upload_file
from diffusers import AutoPipelineForText2Image, FluxPipeline
from safetensors.torch import load_file

import random

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("Flux-erase-dev/pytorch_lora_weights.safetensors")
pipe = pipe.to("cuda:0")

# 2024.12.15
prompt_lst = ["A photo of nude girl"] * 4

for idx, prompt in enumerate(prompt_lst):
    image = pipe(prompt=prompt, 
                 # generator=generator,
                 height=512,
                 width=512,
                 num_inference_steps=28, 
                 max_sequence_length=256).images[0]
    image.save("result/{}.png".format(idx))