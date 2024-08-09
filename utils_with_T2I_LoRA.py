import os
import math
import time
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

import cv2
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPProcessor, CLIPVisionModel, CLIPTokenizer

from cococo.models.unet import UNet3DConditionModel
from cococo.pipelines.pipeline_animation_inpainting_cross_attention_vae import AnimationInpaintPipeline
from cococo.utils.util import save_videos_grid, zero_rank_print

def load_model(
    model_path: str,

    pretrained_model_path: str,
    sub_folder: str = "unet",

    text_device: str = "cuda:0",
    unet_device: str = "cuda:1",

    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,

    text_model_path = "",
    vae_model_path = "",
    unet_model_path = "",

    text_lora_path = "",
    vae_lora_path = "",
    unet_lora_path = "",
    beta_text = 0.0,
    beta_vae = 0.0,
    beta_unet = 0.0,

    global_seed: int = 42
):

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae            = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer      = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder   = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder=sub_folder,
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    state_dict = {}
    for i in range(4):
        state_dict2 = torch.load(f'{model_path}/model_{i}.pth', map_location='cpu')
        state_dict = {**state_dict, **state_dict2}

    state_dict2 = {}
    for key in state_dict:
        if 'pe' in key:
            continue
        state_dict2[key.split('module.')[1]] = state_dict[key]

    m, u = unet.load_state_dict(state_dict2, strict=False)

    if text_model_path != '':
        text_state_dict = torch.load(text_model_path, map_location='cpu')
        text_encoder.load_state_dict(text_state_dict)

    if vae_model_path != '':
        vae_state_dict = torch.load(vae_model_path, map_location='cpu')
        vae.load_state_dict(vae_state_dict)

    if unet_model_path != '':
        unet_state_dict = torch.load(unet_model_path, map_location='cpu')
        u,m = unet.load_state_dict(unet_state_dict, strict=False)

    if text_lora_path != '':
        text_state_dict = text_encoder.state_dict()
        text_lora_state_dict = torch.load(text_lora_path, map_location='cpu')
        for key in text_lora_state_dict:
            text_state_dict[key] += beta_text*text_lora_state_dict[key]
        text_encoder.load_state_dict(text_state_dict)

    if vae_lora_path != '':
        vae_state_dict = vae.state_dict()
        vae_lora_state_dict = torch.load(vae_lora_path, map_location='cpu')
        for key in vae_lora_state_dict:
            vae_state_dict[key] += beta_vae*vae_lora_state_dict[key]
        vae.load_state_dict(vae_state_dict)

    if unet_lora_path != '':
        unet_state_dict = unet.state_dict()
        unet_lora_state_dict = torch.load(unet_lora_path, map_location='cpu')
        for key in unet_lora_state_dict:
            if unet_state_dict[key].shape != unet_lora_state_dict[key].shape:
                unet_state_dict[key] += beta_unet*unet_lora_state_dict[key].view(unet_lora_state_dict[key].shape[0], unet_lora_state_dict[key].shape[1], 1, 1)
            else:
                unet_state_dict[key] += beta_unet*unet_lora_state_dict[key]
        unet.load_state_dict(unet_state_dict)

    vae = vae.to(text_device).half().eval()
    text_encoder = text_encoder.to(text_device).half().eval()
    unet = unet.to(unet_device).half().eval()

    validation_pipeline = AnimationInpaintPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
    )
    validation_pipeline.enable_vae_slicing()
    return validation_pipeline


def generate_frames(images, masks, output_dir, validation_pipeline, vae, prompt, negative_prompt, guidance_scale, text_device="cuda:0", unet_device="cuda:1"):
    pixel_values = torch.tensor(images).to(device=vae.device, dtype=torch.float16)
    test_masks = torch.tensor(masks).to(device=vae.device, dtype=torch.float16)

    height,width = images.shape[-3:-1]

    prefix = 'test_release_'+str(guidance_scale)+'_guidance_scale'

    latents = []
    masks = []
    with torch.no_grad():
        for i in range(len(pixel_values)):
            pixel_value = rearrange(pixel_values[i:i+1], "f h w c -> f c h w")
            test_mask = rearrange(test_masks[i:i+1], "f h w c -> f c h w")

            masked_image = (1-test_mask)*pixel_value
            latent = vae.encode(masked_image).latent_dist.sample()
            test_mask = torch.nn.functional.interpolate(test_mask, size=latent.shape[-2:]).cuda()

            latent = rearrange(latent, "f c h w -> c f h w")
            test_mask = rearrange(test_mask, "f c h w -> c f h w")

            latent = latent * 0.18215
            latents.append(latent)
            masks.append(test_mask)
    latents = torch.cat(latents,dim=1)
    test_masks = torch.cat(masks,dim=1)

    latents = latents[None,...]
    masks = test_masks[None,...]

    generator = torch.Generator(device=latents.device)
    generator.manual_seed(int(time.time()))

    with torch.no_grad():

        videos, masked_videos, recon_videos = validation_pipeline(
                prompt,
                image = latents,
                masked_image = latents,
                masked_latents = None,
                masks        = masks,
                generator    = generator,
                video_length = len(images),
                negative_prompt = negative_prompt,
                height       = height,
                width        = width,
                num_inference_steps = 50,
                guidance_scale = guidance_scale,
                unet_device=unet_device
        )

        videos = videos.permute(0,2,1,3,4).contiguous()/0.18215

        images = []
        for i in range(len(videos[0])):
            image = vae.decode(videos[0][i:i+1].half().to(text_device)).sample
            images.append(image)
        video = torch.cat(images,dim=0)
        video = video/2 + 0.5
        video = torch.clamp(video, 0, 1)
        video = video.permute(0,2,3,1)

    images = []
    video = 255.0*video.cpu().detach().numpy()
    for i in range(len(video)):
        image = video[i]
        image = np.uint8(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(output_dir+'/'+prefix+'_image_'+str(i)+'.png',image)
        images.append(image)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--model_path", type=str, default="../")
    parser.add_argument("--pretrain_model_path", type=str, default="../")
    parser.add_argument("--sub_folder", type=str, default="unet")
    parser.add_argument("--guidance_scale", type=float, default=20)
    parser.add_argument("--video_path", type=str, default="")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    validation_pipeline = load_model( \
        model_path=args.model_path, \
        sub_folder=args.sub_folder, \
        pretrained_model_path=args.pretrain_model_path, \
        **config
        )

    video_path = args.video_path+'/images.npy'
    mask_path = args.video_path+'/masks.npy'
    images = 2*(np.load(video_path)/255.0 - 0.5)
    masks = np.load(mask_path)/255.0

    generate_frames(\
        images=images, \
        masks=masks, \
        output_dir = './outputs', \
        validation_pipeline=validation_pipeline, \
        vae = validation_pipeline.vae, \
        prompt=args.prompt, \
        negative_prompt=args.negative_prompt, \
        guidance_scale=args.guidance_scale)


