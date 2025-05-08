# coding: UTF-8
"""
    @date:  2024.10.29  week44  
    @func:  attention map 
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()


def search_sequence_numpy(arr, seq):
    non_zero_indices = np.where(seq != 0)
    result_indices = []
    
    for row, col in zip(*non_zero_indices):
        match_indices = np.where(arr[row] == seq[row, col])[0]
        
        if match_indices.size > 0:
            result_indices.extend([col_idx for col_idx in match_indices])
    
    return result_indices

def get_word_index(prompt, attn_words, tokenizer_t5):
    prompt_text_ids = tokenizer_t5(prompt,
                                   padding="max_length",
                                   max_length=256,
                                   truncation=True,
                                   return_length=False,
                                   return_overflowing_tokens=False,
                                   return_tensors="np",).input_ids
    word_ids = tokenizer_t5(attn_words,
                            padding="max_length",
                            max_length=256,
                            truncation=True,
                            return_length=False,
                            return_overflowing_tokens=False,
                            return_tensors="np",).input_ids
    
    # words_ids         [256]
    # prompt_text_ids   [256]
    idxs = search_sequence_numpy(prompt_text_ids, word_ids)
    
    # print(prompt_text_ids, word_ids)
    return idxs[:-1]


if __name__ == "__main__":
    tokenizer_t5 = pipe.tokenizer_2
    answer = get_word_index("a nude girl with beautiful hair and her breast open to see", "breast", tokenizer_t5)
    print(answer)
