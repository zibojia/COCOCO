# CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility

**[Bojia Zi<sup>1</sup>](https://scholar.google.fi/citations?user=QrMKIkEAAAAJ&hl=en), [Shihao Zhao<sup>2</sup>](https://scholar.google.com/citations?user=dNQiLDQAAAAJ&hl=en), [Xianbiao Qi<sup>*4</sup>](https://scholar.google.com/citations?user=odjSydQAAAAJ&hl=en), [Jianan Wang<sup>4</sup>](https://scholar.google.com/citations?user=mt5mvZ8AAAAJ&hl=en), [Yukai Shi<sup>3</sup>](https://scholar.google.com/citations?user=oQXfkSQAAAAJ&hl=en), [Qianyu Chen<sup>1</sup>](https://scholar.google.com/citations?user=Kh8FoLQAAAAJ&hl=en), [Bin Liang<sup>1</sup>](https://scholar.google.com/citations?user=djpQeLEAAAAJ&hl=en), [Kam-Fai Wong<sup>1</sup>](https://scholar.google.com/citations?user=fyMni2cAAAAJ&hl=en), [Lei Zhang<sup>4</sup>](https://scholar.google.com/citations?user=fIlGZToAAAAJ&hl=en)**

<sup>1</sup>The Chinese University of Hong Kong    <sup>2</sup>The University of Hong Kong    <sup>3</sup>Tsinghua University    <sup>4</sup>International Digital Economy Academy

\* is corresponding author.

*This is the inference code for our paper CoCoCo.*

---------------------------------------


## Usage

### Download pretrained models

**The pretrained image inpainting model is available [release](https://huggingface.co/runwayml/stable-diffusion-inpainting).**

**The pretrained video inpainting model is available [release](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155203591_link_cuhk_edu_hk/EoXyViqDi8JEgBDCbxsyPY8BCg7YtkOy73SbBY-3WcQ72w?e=cDZuXM).**

[1]. The image models are put in /path/to/StableDiffusion.

[2]. The video models are put in /path/to/Video_inpainting.

### Prepare the mask

**You can obtain mask by GroundingDINO+Track-Anything, or draw masks by yourselves.**

```run_code

python3 valid_code_release.py --config ./configs/code_release.yaml --prompt "Trees. Snow mountains. best quality." --negative_prompt "worst quality. bad quality." --guidance_scale 10 --video_path ./images/

```



#### TO DO

---------------------------------------

[1]. *We will use larger dataset with high-quality videos to produce a more powerful video inpainting model soon.*

[2]. *We will provide interactive UI to process videos.*

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


