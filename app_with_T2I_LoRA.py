import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import scipy
from collections import OrderedDict
import requests
import json
import torchvision
import torch
import psutil
from omegaconf import OmegaConf
import time

from decord import VideoReader

from utils_with_T2I_LoRA import load_model, generate_frames

from sam2.build_sam import build_sam2_video_predictor

parser = argparse.ArgumentParser()
parser.add_argument("--config",   type=str, required=True)
parser.add_argument("--text_lora_path", type=str, default="")
parser.add_argument("--vae_lora_path", type=str, default="")
parser.add_argument("--unet_lora_path", type=str, default="")
parser.add_argument("--text_model_path", type=str, default="")
parser.add_argument("--vae_model_path", type=str, default="")
parser.add_argument("--unet_model_path", type=str, default="")
parser.add_argument("--beta_text", type=float, default=1)
parser.add_argument("--beta_vae", type=float, default=1)
parser.add_argument("--beta_unet", type=float, default=1)
parser.add_argument("--model_path", type=str, default="../")
parser.add_argument("--pretrain_model_path", type=str, default="../")
parser.add_argument("--sub_folder", type=str, default="unet")
args = parser.parse_args()

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

config = OmegaConf.load(args.config)
validation_pipeline = load_model(model_path=args.model_path, \
    sub_folder=args.sub_folder, \
    pretrained_model_path=args.pretrain_model_path, \
    text_lora_path = args.text_lora_path, \
    vae_lora_path = args.vae_lora_path, \
    unet_lora_path = args.unet_lora_path, \
    text_model_path = args.text_model_path, \
    vae_model_path = args.vae_model_path, \
    unet_model_path = args.unet_model_path, \
    beta_text = args.beta_text, \
    beta_vae = args.beta_vae, \
    beta_unet = args.beta_unet, \
    **config)

def init_state(
        offload_video_to_cpu=False,
        offload_state_to_cpu=False
        ):
    inference_state = {}
    inference_state["images"] = torch.zeros([1,3,100,100])
    inference_state["num_frames"] = 1
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = 100
    inference_state["video_width"] = 100
    inference_state["device"] = torch.device("cuda")
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = torch.device("cuda")
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state = gr.State(inference_state)
    return inference_state

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already. Try click the image for adding targets to track and inpaint.","Normal")]
    vr = VideoReader(video_path)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    inference_state = predictor.init_state(images=frames)
    fps = 30
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "ann_obj_id": 0
        }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)
    return gr.update(visible=True), gr.update(visible=True), \
    inference_state, video_state, \
    video_info, video_state["origin_images"][0], \
    gr.update(visible=True, maximum=len(frames), value=1), \
    gr.update(visible=True, maximum=len(frames), value=len(frames)), \
    gr.update(visible=True), gr.update(visible=True), \
    gr.update(visible=True), gr.update(visible=True), \
    gr.update(visible=True), gr.update(visible=True), \
    gr.update(visible=True), gr.update(visible=True, value=operation_log)

# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider
    operation_log = [("",""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider),"Normal")]

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Set the tracking finish at frame {}".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log

def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state

# use sam to get the mask
def sam_refine(inference_state, video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):

    ann_obj_id = 0
    ann_frame_idx = video_state["select_frame_number"]

    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    prompt = get_prompt(click_state=click_state, click_input=coordinate)
    points=np.array(prompt["input_point"])
    labels=np.array(prompt["input_label"])
    height, width = video_state["origin_images"][0].shape[0:2]

    for i in range(len(points)):
        points[i,0] = int(points[i,0]/width*1024)
        points[i,1] = int(points[i,1]/height*1024)

    frame_idx, obj_ids, mask = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    )

    mask_ = mask.cpu().squeeze().detach().numpy()
    print(mask.shape)
    mask_[mask_<=0] = 0
    mask_[mask_>0] = 1
    org_image = video_state["origin_images"][video_state["select_frame_number"]]
    mask_ = cv2.resize(mask_, (width, height))
    mask_ = mask_[:, :, None]
    mask_[mask_>0.5] = 1
    mask_[mask_<=0.5] = 0
    color = 63*np.ones((height, width, 3)) * np.array([[[np.random.randint(5),np.random.randint(5),np.random.randint(5)]]])
    painted_image = np.uint8((1-0.5*mask_)*org_image + 0.5*mask_*color)

    video_state["masks"][video_state["select_frame_number"]] = mask_
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("",""), ("Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment","Normal")]
    return painted_image, video_state, interactive_state, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log

def clear_click(inference_state, video_state, click_state):
    predictor.reset_state(inference_state)
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Clear points history and refresh the image.","Normal")]
    return inference_state, template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all mask, please add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("Select {} for tracking or inpainting".format(mask_dropdown),"Normal")]
    return select_frame, operation_log

# tracking vos
def vos_tracking_video(inference_state, video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Track the selected masks, and then you can select the masks for inpainting.","Normal")]
    height, width = video_state["origin_images"][0].shape[0:2]

    masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = np.zeros([1024, 1024, 1])
        for i in range(len(out_mask_logits)):
            out_mask = out_mask_logits[i].cpu().squeeze().detach().numpy()
            out_mask[out_mask>0] = 1
            out_mask[out_mask<=0] = 0
            out_mask = out_mask[:,:,None]
            mask += out_mask
        mask = cv2.resize(mask, (width, height))
        mask = mask[:,:,None]
        mask[mask>0.5] = 1
        mask[mask<1] = 0
        mask = scipy.ndimage.binary_dilation(mask, iterations=12)
        masks.append(mask)
    masks = np.array(masks)

    painted_images = None
    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        org_images = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
        color = 255*np.ones((1, org_images.shape[-3], org_images.shape[-2], 3)) * np.array([[[[0,1,1]]]])
        painted_images = np.uint8((1-0.5*masks)*org_images + 0.5*masks*color)
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"] = masks
        org_images = video_state["origin_images"]
        color = 255*np.ones((1, org_images.shape[-3], org_images.shape[-2], 3)) * np.array([[[[0,1,1]]]])
        painted_images = np.uint8((1-0.5*masks)*org_images + 0.5*masks*color)
        video_state["painted_images"] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/track/{}".format(video_state["video_name"]), fps=video_state["fps"]) # import video_input to name the output video
    interactive_state["inference_times"] += 1
    
    return inference_state, video_output, video_state, interactive_state, operation_log

# inpaint 
def inpaint_video(video_state, text_pos_input, text_neg_input, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Removed the selected masks.","Normal")]

    validation_images = video_state["origin_images"][list(range(0, len(video_state["origin_images"]),2))]
    validation_images = [cv2.resize(validation_images[i], (640, 360)) for i in range(len(validation_images))]
    #validation_images = [cv2.cvtColor(validation_images[i], cv2.COLOR_BGR2RGB) for i in range(len(validation_images))]
    validation_images = np.array(validation_images)
    validation_images = validation_images/127.5 - 1.0
    validation_masks = video_state["masks"][list(range(0, len(video_state["origin_images"]),2))]
    validation_masks = np.float32(validation_masks)
    validation_masks = [cv2.resize(validation_masks[i], (640, 360)) for i in range(len(validation_masks))]
    validation_masks = np.array(validation_masks)
    validation_masks = validation_masks[:,:,:,None]
    validation_masks[validation_masks<1]=0

    print(validation_images.shape)
    print(validation_masks.shape)

    print(str(text_pos_input))
    print(str(text_neg_input))

    images = generate_frames(\
    images=validation_images, \
    masks=validation_masks, \
    output_dir = './outputs', \
    validation_pipeline=validation_pipeline, \
    vae = validation_pipeline.vae, \
    prompt=str(text_pos_input), \
    negative_prompt=str(text_neg_input), \
    guidance_scale=14)

    #frames = np.asarray(video_state["origin_images"])

    video_output = generate_video_from_frames(images, output_path="./result/inpaint/{}".format(video_state["video_name"]), fps=12) # import video_input to name the output video

    return video_output, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def echo_text(text1, text2):
    print(f"你输入的文本是：pos {text1}, neg {text2}")
    return f"你输入的文本是：pos {text1}, neg {text2}"

title = """<p><h1 align="center">COCOCO Inference with SAM2</h1></p>
    """
description = """<p>Gradio demo for COCOCO</p>"""


with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": False,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1,
    }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30,
        "ann_obj_id": 0
        }
    )
    inference_state = init_state()
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row():
                video_input = gr.Video()#autosize=True)
                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
                    resize_info = gr.Textbox(value="If you want to use the inpaint function, it is best to git clone the repo and use a machine with more VRAM locally. \
                                            Alternatively, you can use the resize ratio slider to scale down the original image to around 360P resolution for faster processing.", label="Tips for running this demo.")
                    resize_ratio_slider = gr.Slider(minimum=0.02, maximum=1, step=0.02, value=1, label="Resize ratio", visible=True)
          

            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 

                     # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point prompt",
                                interactive=True,
                                visible=False)
                            clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)#.style(height=160)
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False)#.style(height=360)
                    with gr.Row():
                        image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                        track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                    text_pos_input = gr.Textbox(label="Positive Prompt", placeholder="positive prompt...", interactive=True, visible=False)
                    text_neg_input = gr.Textbox(label="Negative Prompt", placeholder="negative prompt...", interactive=True, visible=False)
            
                with gr.Column():
                    run_status = gr.HighlightedText(value=[("Text","Error"),("to be","Label 2"),("highlighted","Label 3")], visible=False)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                    video_output = gr.Video(visible=False)#gr.Video(autosize=True, visible=False)#.style(height=360)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(value="Tracking", visible=False)
                        inpaint_video_predict_button = gr.Button(value="Inpainting", visible=False)

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state
        ],
        outputs=[text_pos_input, text_neg_input, inference_state, video_state, video_info, template_frame,
                 image_selection_slider, track_pause_number_slider, 
                 point_prompt, clear_button_click, template_frame, tracking_video_predict_button, 
                 video_output, mask_dropdown, inpaint_video_predict_button, 
                 run_status]
    )   

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state, interactive_state], 
                                   outputs=[template_frame, video_state, interactive_state, run_status], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number, 
                                   inputs=[track_pause_number_slider, video_state, interactive_state], 
                                   outputs=[template_frame, interactive_state, run_status], api_name="end_image")
    resize_ratio_slider.release(fn=get_resize_ratio, 
                                   inputs=[resize_ratio_slider, interactive_state], 
                                   outputs=[interactive_state], api_name="resize_ratio")
    
    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[inference_state, video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[inference_state, video_state, interactive_state, mask_dropdown],
        outputs=[inference_state, video_output, video_state, interactive_state, run_status]
    )

    # inpaint video from select image and mask
    inpaint_video_predict_button.click(
        fn=inpaint_video,
        inputs=[video_state, text_pos_input, text_neg_input, interactive_state, mask_dropdown],
        outputs=[video_output, run_status]
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status]
    )
    
    # clear input
    video_input.clear(
        lambda: (
        gr.update(visible=False), 
        gr.update(visible=False),
        init_state(),
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30,
        "ann_obj_id": 0
        },
        {
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": False,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": 0,
        "resize_ratio": 1
        },
        [[],[]],
        None,
        None,
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=[]), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)           
        ),
        [],
        [ 
            text_pos_input,
            text_neg_input,
            inference_state,
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider , track_pause_number_slider, point_prompt, 
            clear_button_click, template_frame, tracking_video_predict_button, video_output, 
            mask_dropdown, inpaint_video_predict_button, run_status
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [inference_state, video_state, click_state],
        outputs = [inference_state, template_frame, click_state, run_status],
    )
#iface.queue()#concurrency_count=1)
iface.launch(server_port=8000, server_name="0.0.0.0")
# iface.launch(debug=True, enable_queue=True)
