import json
import torch
from safetensors.torch import load_file
import argparse

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments
parser.add_argument('--tensor_path', type=str, default='', help='')
parser.add_argument('--unet_path', type=str, default='', help='')
parser.add_argument('--text_encoder_path', type=str, default='', help='')
parser.add_argument('--vae_path', type=str, default='', help='')

parser.add_argument('--source_path', type=str, default='', help='')
parser.add_argument('--target_path', type=str, default='', help='')

parser.add_argument('--target_prefix', type=str, default='', help='')

# Parse the arguments
args = parser.parse_args()

unet_source_path = args.source_path + "/source.txt"
unet_target_path = args.target_path + "/target.txt"

text_encoder_source_path = args.source_path + "/text_source.txt"
text_encoder_target_path = args.target_path + "/text_target.txt"

vae_source_path = args.source_path + "/vae_source.txt"
vae_target_path = args.target_path + "/vae_target.txt"


tensor_dict = load_file(args.tensor_path)

state_dict = torch.load(args.unet_path, map_location='cpu')
text_state_dict = torch.load(args.text_encoder_path, map_location='cpu')
vae_state_dict = torch.load(args.vae_path, map_location='cpu')
state_dict = {**vae_state_dict, **state_dict, **text_state_dict}

# Convert diffusion model
f = open(unet_source_path,'r')
source = f.readlines()
f.close()

f = open(unet_target_path,'r')
target = f.readlines()
f.close()

state_dict2 = {}
for source_key, target_key in zip(source, target):
    source_key = source_key.strip()
    target_key = target_key.strip()

    if tensor_dict[source_key].shape == state_dict[target_key].shape and source_key != 'model.diffusion_model.input_blocks.0.0.weight':
        state_dict2[target_key] = tensor_dict[source_key] - state_dict[target_key]
    elif source_key == 'model.diffusion_model.input_blocks.0.0.weight':
        delta_weight = torch.cat([tensor_dict[source_key] - state_dict[target_key], torch.zeros([320,5,3,3])], dim=1)
        state_dict2[target_key] = delta_weight

torch.save(state_dict2, f'{args.target_prefix}_unet_delta.pth')

# Convert text encoder model
f = open(text_encoder_source_path,'r')
source = f.readlines()
f.close()

f = open(text_encoder_target_path,'r')
target = f.readlines()
f.close()

state_dict2 = {}
for source_key, target_key in zip(source, target):
    source_key = source_key.strip()
    target_key = target_key.strip()

    if tensor_dict[source_key].shape == state_dict[target_key].shape:
        state_dict2[target_key] = tensor_dict[source_key] - state_dict[target_key]
    else:
        print(source_key, tensor_dict[source_key].shape, state_dict[target_key].shape)

torch.save(state_dict2, f'{args.target_prefix}_text_encoder_delta.pth')

# Convert vae model
f = open(vae_source_path,'r')
source = f.readlines()
f.close()

f = open(vae_target_path,'r')
target = f.readlines()
f.close()

state_dict2 = {}
for source_key, target_key in zip(source, target):
    source_key = source_key.strip()
    target_key = target_key.strip()

    if tensor_dict[source_key].shape == state_dict[target_key].shape:
        state_dict2[target_key] = tensor_dict[source_key] - state_dict[target_key]
    else:
        state_dict2[target_key] = tensor_dict[source_key].squeeze() - state_dict[target_key]
        print(source_key, tensor_dict[source_key].shape, state_dict[target_key].shape)

torch.save(state_dict2, f'{args.target_prefix}_vae_delta.pth')
