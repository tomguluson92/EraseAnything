# coding: UTF-8
"""
    @date:    2024.11.25  week48  Monday
    @readme:  propose dataset loader:
              (a) 1-1 prompt2image
              (b) prompt shuffle
              (c) other strengthen method
"""
import os
import argparse
import copy
import itertools
import logging
import math
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose

import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from utils.find_token import get_word_index
from diffusers import FluxPipeline


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)


class LoraDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        key_word,
        tokenizer_t5,
        class_prompt=None,
        train_text_encoder_ti=False,
        token_abstraction_dict=None,  # token mapping for textual inversion
        class_data_root=None,
        class_num=None,
        size=512,
        repeats=1,
        center_crop=False,
        random_flip=False,
    ):
        self.size = size
        self.center_crop = center_crop
        
        if key_word in instance_prompt:
            self.key_word = key_word
        else:
            self.key_word = None
        
        
        self.tokenizer_t5 = tokenizer_t5
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
        self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if center_crop:
                y1 = max(0, int(round((image.height - size) / 2.0)))
                x1 = max(0, int(round((image.width - size) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (size, size))
                image = crop(image, y1, x1, h, w)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                if self.train_text_encoder_ti:
                    # replace instances of --token_abstraction in caption with the new tokens: "<si><si+1>" etc.
                    for token_abs, token_replacement in self.token_abstraction_dict.items():
                        caption = caption.replace(token_abs, "".join(token_replacement))
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # the given instance prompt is used for all images
            example["instance_prompt"] = self.instance_prompt
            
        # randomly shuffle the instance_prompt
        if random.random() >= 0.1:
            tmp_words = example["instance_prompt"].split(" ")
            random.shuffle(tmp_words)
            shuffled_sentence = ' '.join(tmp_words)
            example["instance_prompt"] = shuffled_sentence
        
        synonym_nltk_lst = list(get_synonyms(self.key_word))
        if len(synonym_nltk_lst) > 0:
            synonym_word = random.choice(synonym_nltk_lst)
            example["synonym"] = synonym_word
        else:
            example["synonym"] = self.key_word
        
        # pick idx
        if self.key_word is not None:
            remove_index = get_word_index(example["instance_prompt"], self.key_word, self.tokenizer_t5)
            example["remove_indices"] = remove_index
            
            # synonym replacement.
            if random.random() >= 0.5:
                example["instance_prompt"] = example["instance_prompt"].replace(self.key_word, example["synonym"])
        else:
            example["remove_indices"] = None
            
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example

    
def collate_data_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    remove_indexes = [example["remove_indices"] for example in examples]
    synonym_words = [example["synonym"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts, "remove_indices": remove_indexes, "synonym_words": synonym_words}
    return batch
    
    
if __name__ == "__main__":
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    tokenizer_t5 = pipe.tokenizer_2
    train_dataset = LoraDataset(instance_data_root="image",
        instance_prompt="a nude girl with beautiful hair and her breast open to see",
        key_word="nude",
        tokenizer_t5=tokenizer_t5)
    
    # for item in data_test:
    #     import pdb; pdb.set_trace()
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda examples: collate_data_fn(examples, False),
        num_workers=0,
    )
    
    # word = "happy"
    # print(get_synonyms(word))
    
    for batch in train_dataloader:
        import pdb; pdb.set_trace()