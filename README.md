# CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility

**[Bojia Zi<sup>1</sup>](https://scholar.google.fi/citations?user=QrMKIkEAAAAJ&hl=en), [Shihao Zhao<sup>2</sup>](https://scholar.google.com/citations?user=dNQiLDQAAAAJ&hl=en), [Xianbiao Qi<sup>*4</sup>](https://scholar.google.com/citations?user=odjSydQAAAAJ&hl=en), [Jianan Wang<sup>4</sup>](https://scholar.google.com/citations?user=mt5mvZ8AAAAJ&hl=en), [Yukai Shi<sup>3</sup>](https://scholar.google.com/citations?user=oQXfkSQAAAAJ&hl=en), [Qianyu Chen<sup>1</sup>](https://scholar.google.com/citations?user=Kh8FoLQAAAAJ&hl=en), [Bin Liang<sup>1</sup>](https://scholar.google.com/citations?user=djpQeLEAAAAJ&hl=en), [Kam-Fai Wong<sup>1</sup>](https://scholar.google.com/citations?user=fyMni2cAAAAJ&hl=en), [Lei Zhang<sup>4</sup>](https://scholar.google.com/citations?user=fIlGZToAAAAJ&hl=en)**

<sup>1</sup>The Chinese University of Hong Kong    <sup>2</sup>The University of Hong Kong    <sup>3</sup>Tsinghua University    <sup>4</sup>International Digital Economy Academy

\* is corresponding author.

*This is the inference code for our paper CoCoCo.*


<p align="center">
  <img src="https://github.com/zibojia/COCOCO/blob/main/__asset__/COCOCO.PNG" alt="COCOCO" style="width: 60%;"/>
</p>

---------------------------------------


## Usage

### 1. Download pretrained models. 

**Note that our method requires both parameters of sd1.5 inpainting and cococo.**

**The pretrained image inpainting model is [available](https://huggingface.co/runwayml/stable-diffusion-inpainting).**

**The pretrained video inpainting model is [available](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155203591_link_cuhk_edu_hk/EoXyViqDi8JEgBDCbxsyPY8BCg7YtkOy73SbBY-3WcQ72w?e=cDZuXM).**

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

**You can obtain mask by [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) or [Track-Anything](https://github.com/gaomingqi/Track-Anything), or draw masks by yourself.**


### 3. Run our validation script.
```run_code

python3 valid_code_release.py --config ./configs/code_release.yaml --prompt "Trees. Snow mountains. best quality." --negative_prompt "worst quality. bad quality." --guidance_scale 10 --video_path ./images/ --model_path [cococo_folder_name] --pretrain_model_path [sd_folder_name] --sub_folder unet

```

### Using our Inpainting model with T2Is

Our idea is based on the [task vector](https://arxiv.org/abs/2212.04089).

Surprisingly, we found that inpainting model is compatiable with T2I model, even the first convlutional channel is mimatch. 

<p align="center">
  <img src="https://github.com/zibojia/COCOCO/blob/main/__asset__/task.PNG" alt="COCOCO" style="width: 60%;"/>
</p>



**For the model using different key**

For example, the epiCRealism.

```
model.diffusion_model.input_blocks.1.1.norm.bias
model.diffusion_model.input_blocks.1.1.norm.weight
model.diffusion_model.input_blocks.1.1.proj_in.bias
model.diffusion_model.input_blocks.1.1.proj_in.weight
model.diffusion_model.input_blocks.1.1.proj_out.bias
model.diffusion_model.input_blocks.1.1.proj_out.weight
```

We develope a tool to convert this type model to the delta of weight.

```
cd task_vector;
python3 convert.py --tensor_path [safetensor_path] --unet_path [unet_path] --text_encoder_path [text_encoder_path] --vae_path [vae_path] --source_path [source_path] --target_path [target_path] --target_prefix [prefix];
```


#### TO DO

---------------------------------------

[1]. *We will use larger dataset with high-quality videos to produce a more powerful video inpainting model soon.*

[2]. *We will provide interactive UI to process videos.*

[3]. *The code for LoRA will be given in 24 hours.*

[4]. *The training code is under preparation.*

#### Citation

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


