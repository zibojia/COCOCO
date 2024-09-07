# CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility
<a href='https://cococozibojia.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/pdf/2403.12035'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>


**[Bojia Zi<sup>1</sup>](https://scholar.google.fi/citations?user=QrMKIkEAAAAJ&hl=en), [Shihao Zhao<sup>2</sup>](https://scholar.google.com/citations?user=dNQiLDQAAAAJ&hl=en), [Xianbiao Qi<sup>*4</sup>](https://scholar.google.com/citations?user=odjSydQAAAAJ&hl=en), [Jianan Wang<sup>4</sup>](https://scholar.google.com/citations?user=mt5mvZ8AAAAJ&hl=en), [Yukai Shi<sup>3</sup>](https://scholar.google.com/citations?user=oQXfkSQAAAAJ&hl=en), [Qianyu Chen<sup>1</sup>](https://scholar.google.com/citations?user=Kh8FoLQAAAAJ&hl=en), [Bin Liang<sup>1</sup>](https://scholar.google.com/citations?user=djpQeLEAAAAJ&hl=en), [Kam-Fai Wong<sup>1</sup>](https://scholar.google.com/citations?user=fyMni2cAAAAJ&hl=en), [Lei Zhang<sup>4</sup>](https://scholar.google.com/citations?user=fIlGZToAAAAJ&hl=en)**

<sup>1</sup>The Chinese University of Hong Kong    <sup>2</sup>The University of Hong Kong    <sup>3</sup>Tsinghua University    <sup>4</sup>International Digital Economy Academy

\* is corresponding author.

*This is the inference code for our paper CoCoCo.*


<p align="center">
  <img src="https://github.com/zibojia/COCOCO/blob/main/__asset__/COCOCO.PNG" alt="COCOCO" style="width: 60%;"/>
</p>

<table>
    <tr>
    <td><img src="__asset__/sea_org.gif"></td>
    <td><img src="__asset__/sea1.gif"></td>
    <td><img src="__asset__/sea2.gif"></td>
    </tr>
    <tr>
    <td> Orginal </td>
    <td> The ocean, the waves ...  </td>
    <td> The ocean, the waves ...  </td>
    </tr>
    
</table>

<table>
    <tr>
    <td><img src="__asset__/river_org.gif"></td>
    <td><img src="__asset__/river1.gif"></td>
    <td><img src="__asset__/river2.gif"></td>
    </tr>
    <tr>
    <td> Orginal </td>
    <td> The river with ice ...  </td>
    <td> The river with ice ...  </td>
    </tr>
</table>

<table>
    <tr>
    <td><img src="__asset__/sky_org.gif"></td>
    <td><img src="__asset__/sky1.gif"></td>
    <td><img src="__asset__/sky2.gif"></td>
    </tr>
    <tr>
    <td> Orginal </td>
    <td> Meteor streaking in the sky ...  </td>
    <td> Meteor streaking in the sky ...  </td>
    </tr>
</table>

## Table of Contents <!-- omit in toc -->

- [Installation](#Installation)
- [Usage](#Usage)
- [Download pretrained models](#1.-download-pretrained-models.)
+ []

### Installation

#### Installation Checklist
*Before install the dependencies, you should check the following requirements to overcome the installation failure.*
- [x] You have a GPU with at least 24G GPU memory.
- [x] Your CUDA version is greater than 12.0.
- [x] Your Pytorch version is greater than 2.4.
- [x] Your gcc version is greater than 9.4.
- [x] Your diffusers version is 0.11.1.

*If you update your enviroments successfully, then try to install the dependencies by pip.*

```
pip3 install -r requirements.txt
pip3 install -e .
```


## Usage
### 1. Download pretrained models. 

***Note that our method requires both parameters of SD1.5 inpainting and cococo.***

 * **The pretrained image inpainting model ([Stable Diffusion Inpainting](https://huggingface.co/benjamin-paine/stable-diffusion-v1-5-inpainting).)**

 * **The CoCoCo [Checkpoints](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155203591_link_cuhk_edu_hk/EoXyViqDi8JEgBDCbxsyPY8BCg7YtkOy73SbBY-3WcQ72w?e=cDZuXM).**

[1]. The image models are put in [sd_folder_name]. 

For example, we can use the scripts to create a folder, and put the model to this folder.

```
mkdir [sd_folder_name]; cd [sd_folder_name]; wget [sd_download_link];
```

[2]. The video inpainting models are put in [cococo_folder_name].

```
mkdir [cococo_folder_name]; cd [cococo_folder_name]; wget [cococo_download_link];
```

### 2. Prepare the mask

**You can obtain mask by [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) or [Track-Anything](https://github.com/gaomingqi/Track-Anything) or our [Demo Code](https://github.com/zibojia/COCOCO?tab=readme-ov-file#5-cococo-inference-with-sam2), or draw masks by yourself.**


### 3. Run our validation script.
```run_code

python3 valid_code_release.py --config ./configs/code_release.yaml \
--prompt "Trees. Snow mountains. best quality." \
--negative_prompt "worst quality. bad quality." \
--guidance_scale 10 \ # the cfg number, higher means more powerful text controlability
--video_path ./images/ \ # the path that store the video, the format is the images.npy and masks.npy
--model_path [cococo_folder_name] \
--pretrain_model_path [sd_folder_name] \ # the path that store the pretrained stable inpainting model, e.g. stable-diffusion-inpainting
--sub_folder unet # set the subfolder of pretrained stable inpainting model to get the unet checkpoints

```

### 4. Personalized Video Inpainting (Optional)

*We give a method to allow users to compose their own personlized video inpainting model by using personalized T2Is* **WITHOUT TRAINING**. There are two steps in total:

1. Transform the personalized image diffusion to personliazed inpainting diffusion. Substract the weights of personalized image diffusion from SD1.5, and add them on inpainting model. Surprisingly, this method can get a personalized image inpainting model, and it works well:)
   
2. Add the weight of personalized inpainting model to our CoCoCo.

<table>
    <tr>
    <td><img src="__asset__/gibuli_lora_org.gif"></td>
    <td><img src="__asset__/gibuli_merged1.gif"></td>
    <td><img src="__asset__/gibuli_merged2.gif"></td>
    </tr>
</table>





<table>
    <tr>
    <td><img src="__asset__/unmbrella_org.gif"></td>
    <td><img src="__asset__/unmbrella1.gif"></td>
    <td><img src="__asset__/unmbrella2.gif"></td>
    </tr>
</table>

<table>
    <tr>
    <td><img src="__asset__/gibuli.gif"></td>
    <td><img src="__asset__/bocchi1.gif"></td>
    <td><img src="__asset__/bocchi2.gif"></td>
    </tr>
</table>

Our idea is based on the [task vector](https://arxiv.org/abs/2212.04089).

Surprisingly, we found that inpainting model is compatiable with T2I model, even the first convlutional channel is mimatched. 

<p align="center">
  <img src="https://github.com/zibojia/COCOCO/blob/main/__asset__/task.PNG" alt="COCOCO" style="width: 60%;"/>
</p>



**1. For the model using different key, we use the following script to process opensource T2I model.**

For example, the [epiCRealism](https://civitai.com/models/25694?modelVersionId=134065), it is different from the key of the StableDiffusion.

```
model.diffusion_model.input_blocks.1.1.norm.bias
model.diffusion_model.input_blocks.1.1.norm.weight
model.diffusion_model.input_blocks.1.1.proj_in.bias
model.diffusion_model.input_blocks.1.1.proj_in.weight
model.diffusion_model.input_blocks.1.1.proj_out.bias
model.diffusion_model.input_blocks.1.1.proj_out.weight
```

Therefore, we develope a tool to convert this type model to the delta of weight.

```
cd task_vector;
python3 convert.py --tensor_path [safetensor_path] --unet_path [unet_path] --text_encoder_path [text_encoder_path] --vae_path [vae_path] --source_path ./resources --target_path ./resources --target_prefix [prefix];
```

**2. For the model using same key and trained by LoRA.**

For example, the [Ghibli](https://civitai.com/models/54233/ghiblibackground) LoRA.

```
lora_unet_up_blocks_3_resnets_0_conv1.lora_down.weight
lora_unet_up_blocks_3_resnets_0_conv1.lora_up.weight
lora_unet_up_blocks_3_resnets_0_conv2.lora_down.weight
lora_unet_up_blocks_3_resnets_0_conv2.lora_up.weight
```

```
python3 convert_lora.py \
--tensor_path [tensor_path] \
--unet_path [unet_path] \
--text_encoder_path [text_encoder_path] \
--vae_path [vae_path] \
--regulation_path ./lora.json \
--target_prefix [target_prefix]
```

**3. You can use customized T2I or LoRA to create vision content in the masks.**

```
python3 valid_code_release_with_T2I_LoRA.py \
--config ./configs/code_release.yaml --guidance_scale 10 \
--video_path [video_path] \
--masks_path [masks_path] \
--model_path [model_path] \
--pretrain_model_path [pretrain_model_path] \
--sub_folder [sub_folder] \
--unet_lora_path [unet_lora_path] \
--beta_unet 0.75 \
--text_lora_path [text_lora_path] \
--beta_text 0.75 \
--unet_model_path [unet_model_path] \
--text_model_path [text_model_path] \
--vae_model_path [vae_model_path] \
--prompt [prompt] \
--negative_prompt [negative_prompt]
```

### 5. COCOCO INFERENCE with SAM2

<p align="center">
  <img src="https://github.com/zibojia/COCOCO/blob/main/__asset__/DEMO.PNG" alt="DEMO" style="width: 80%;"/>
</p>

**Try our demo with original COCOCO**
```
CUDA_VISIBLE_DEVICES=0,1 python3 app.py \
--config ./configs/code_release.yaml \
--model_path [model_path] \
--pretrain_model_path [pretrain_model_path] \
--sub_folder [sub_folder]
```

**Try our demo with LoRA and checkpoint**

The checkpoint is [available](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155203591_link_cuhk_edu_hk/EpuCr0azYKxJg7QJ71Mln9UBYDLzoFm6GQWYN9UwCauhYg?e=rwPAhY).

The LoRA is [available](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155203591_link_cuhk_edu_hk/EiqYrc8lKUhFkpEb-DC8CV8BJPbqkJsyvz66cjXOCnDS1Q?e=hAgbi9).

```
CUDA_VISIBLE_DEVICES=0,1 python3 app_with_T2I_LoRA.py \
--config ./configs/code_release.yaml \
--text_lora_path [text_lora_path] \
--unet_lora_path [unet_lora_path] \
--beta_text [beta_text] \
--beta_vae [beta_vae] \
--beta_unet [beta_unet] \
--text_model_path [text_model_path] \
--unet_model_path [unet_model_path] \
--vae_model_path [vae_model_path]  \
--model_path [model_path] \
--pretrain_model_path [pretrain_model_path] \
--sub_folder [sub_folder]
```

### TO DO

---------------------------------------

[1]. *We will use larger dataset with high-quality videos to produce a more powerful video inpainting model soon.*


[2]. *The training code is under preparation.*



### Citation

---------------------------------------

```bibtex
@article{Zi2024CoCoCo,
  title={CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility},
  author={Bojia Zi and Shihao Zhao and Xianbiao Qi and Jianan Wang and Yukai Shi and Qianyu Chen and Bin Liang and Kam-Fai Wong and Lei Zhang},
  journal={ArXiv},
  year={2024},
  volume={abs/2403.12035},
  url={https://arxiv.org/abs/2403.12035}
}
```

### Acknowledgement
This code is based on [AnimateDiff](https://github.com/guoyww/AnimateDiff), [Segment-Anything-2](https://github.com/facebookresearch/segment-anything-2) and [propainter](https://github.com/sczhou/ProPainter).


