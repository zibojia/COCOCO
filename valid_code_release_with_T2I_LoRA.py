import os
import math
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

def main(
    name: str,
    use_wandb: bool,
    launcher: str,

    model_path: str,

    prompt: str,
    negative_prompt: str,
    guidance_scale: float,

    output_dir: str,
    pretrained_model_path: str,
    sub_folder: str = "unet",

    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,

    num_workers: int = 32,

    enable_xformers_memory_efficient_attention: bool = True,

    image_path: str = '',
    mask_path: str = '',

    text_model_path: str = '',
    vae_model_path: str = '',
    unet_model_path: str = '',

    text_lora_path: str = '',
    vae_lora_path: str = '',
    unet_lora_path: str = '',
    beta_text: float = 0.0,
    beta_vae: float = 0.0,
    beta_unet: float = 0.0,

    global_seed: int = 42,
    is_debug: bool = False,
):

    seed = global_seed
    torch.manual_seed(seed)

    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

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

    unet.load_state_dict(state_dict2, strict=False)

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

    vae = vae.cuda().half().eval()
    text_encoder = text_encoder.cuda().half().eval()
    unet = unet.cuda().half().eval()

    validation_pipeline = AnimationInpaintPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
    )
    validation_pipeline.enable_vae_slicing()

    video_path = image_path
    mask_path = mask_path
    images = 2*(np.load(video_path)/255.0 - 0.5)
    masks = np.load(mask_path)/255.0
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
    generator.manual_seed(0)

    for step in range(10):

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
                guidance_scale = guidance_scale
            )

        videos = videos.permute(0,2,1,3,4).contiguous()/0.18215

        with torch.no_grad():
            images = []
            for i in range(len(videos[0])):
                image = vae.decode(videos[0][i:i+1].half()).sample
                images.append(image)
            video = torch.cat(images,dim=0)
            video = video/2 + 0.5
            video = torch.clamp(video, 0, 1)
            video = video.permute(0,2,3,1)

        video = 255.0*video.cpu().detach().numpy()
        for i in range(len(video)):
            image = video[i]
            image = np.uint8(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_dir+'/'+prefix+'_'+str(step)+'_image_'+str(i)+'.png',image)

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
    parser.add_argument("--masks_path", type=str, default="")
    parser.add_argument("--text_lora_path", type=str, default="")
    parser.add_argument("--vae_lora_path", type=str, default="")
    parser.add_argument("--unet_lora_path", type=str, default="")
    parser.add_argument("--text_model_path", type=str, default="")
    parser.add_argument("--vae_model_path", type=str, default="")
    parser.add_argument("--unet_model_path", type=str, default="")
    parser.add_argument("--beta_text", type=float, default=1)
    parser.add_argument("--beta_vae", type=float, default=1)
    parser.add_argument("--beta_unet", type=float, default=1)
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, \
        launcher=None, \
        use_wandb=False, \
        prompt=args.prompt, \
        model_path=args.model_path, \
        sub_folder=args.sub_folder, \
        pretrained_model_path=args.pretrain_model_path, \
        negative_prompt=args.negative_prompt, \
        guidance_scale=args.guidance_scale, \
        image_path=args.video_path, \
        mask_path=args.masks_path, \
        text_lora_path = args.text_lora_path, \
        vae_lora_path = args.vae_lora_path, \
        unet_lora_path = args.unet_lora_path, \
        text_model_path = args.text_model_path, \
        vae_model_path = args.vae_model_path, \
        unet_model_path = args.unet_model_path, \
        beta_text = args.beta_text, \
        beta_vae = args.beta_vae, \
        beta_unet = args.beta_unet, \
        **config
        )
