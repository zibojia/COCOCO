import re
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

parser.add_argument('--regulation_path', type=str, default='', help='')
parser.add_argument('--target_prefix', type=str, default='', help='')

# Parse the arguments
args = parser.parse_args()

tensor_dict = load_file(args.tensor_path)

state_dict = torch.load(args.unet_path, map_location='cpu')
text_state_dict = torch.load(args.text_encoder_path, map_location='cpu')
vae_state_dict = torch.load(args.vae_path, map_location='cpu')
state_dict = {**vae_state_dict, **state_dict, **text_state_dict}

json_list = json.load(open(f"{args.regulation_path}","r"))
for item in json_list:
    key = item["start"]
    value = item["end"]

state_dict2 = {}
for key in tensor_dict:
    print(key)
    org_key = key
    for it in json_list:
        reg = it["regression"]
        key2 = it["start"]
        value = it["end"]
        key = key.replace(key2, value)
        if reg:
            key = re.sub(key2, value, key)

    key = key.replace(".lora_up","")
    key = key.replace(".lora_down", "")

    if 'lora_up' in org_key:
        alpha_ = tensor_dict[org_key.split('.lora_up')[0]+'.alpha']
        lora_down = org_key.split('.lora_up')[0]+'.lora_down.weight'
        rank = tensor_dict[org_key].shape[1]
        if len(tensor_dict[org_key].shape) == 4:
            lora1 = tensor_dict[org_key].float()
            w1,_,w3,w4 = lora1.shape
            lora1 = lora1.permute(0,2,3,1).contiguous()
            lora1 = lora1.view(-1, rank)
            lora2 = tensor_dict[lora_down].float()
            _,w2,w3,w4 = lora2.shape
            lora2 = lora2.view(rank, -1)
            weight = alpha_*lora1@lora2/rank
            weight = weight.view(w1,w2,w3,w4)
        else:
            lora1 = tensor_dict[org_key].float()
            lora2 = tensor_dict[lora_down].float()
            weight = lora1@lora2
        state_dict2[key] = weight

    if key.endswith('.alpha'):
        continue
    if key in state_dict:
        continue
    else:
        print('### The key doesn\'t match!')
        exit(0)

text_state_dict = {}
vae_state_dict = {}
unet_state_dict = {}
for key in state_dict2:
    if 'text' in key:
        text_state_dict[key] = state_dict2[key]
    elif key.startswith('encoder.') or key.startswith('decoder.'):
        vae_state_dict[key] = state_dict2[key]
    else:
        unet_state_dict[key] = state_dict2[key]

if len(text_state_dict) > 0:
    torch.save(text_state_dict, f'./text_{args.target_prefix}_delta.pth')
if len(vae_state_dict) > 0:
    torch.save(vae_state_dict, f'./vae_{args.target_prefix}_delta.pth')
if len(unet_state_dict):
    torch.save(unet_state_dict, f'./unet_{args.target_prefix}_delta.pth')
    
