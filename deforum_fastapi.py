import subprocess, time, gc, os, sys
import logging

#@markdown **Environment Setup**
def setup_environment():
    try:
        sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src', 'src'])
        print("..skipping setup")
    except Exception as e:
        logging.error(f"Error in setup_environment: {e}")

setup_environment()

import torch
import random
import clip
from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts
import datetime

from deforumargs import DeforumArgs, DeforumAnimArgs
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi import Request
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
log_file = f'logging/app_{current_time}.log'
logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="test_api"), name="static")
templates = Jinja2Templates(directory="videos")

@app.get("/")
async def hello():
    return {"message": "Hello World"}

#@markdown **Path Setup**
def PathSetup():
    try:
        models_path = "models" #@param {type:"string"}
        configs_path = "configs" #@param {type:"string"}
        output_path = "outputs" #@param {type:"string"}
        mount_google_drive = True #@param {type:"boolean"}
        models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
        output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}
        return locals()
    except Exception as e:
        logging.error(f"Error in PathSetup: {e}")

#@markdown **Model Setup**
def ModelSetup():
    try:
        map_location = "cuda" #@param ["cpu", "cuda"]
        model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
        model_checkpoint =  "Protogen_V2.2.ckpt" #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
        custom_config_path = "" #@param {type:"string"}
        custom_checkpoint_path = "" #@param {type:"string"}
        return locals()
    except Exception as e:
        logging.error(f"Error in ModelSetup: {e}")



root = SimpleNamespace(**PathSetup())
root.models_path, root.output_path = get_model_output_paths(root)
root.__dict__.update(ModelSetup())
root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

"""
# Settings
"""
# prompts
# prompts

stored_prompts = {
    "pos_prompts": {},
    "neg_prompts": {}
}

title_name = 'video_0925'
class PromptsModel(BaseModel):
    positive_prompts: Dict[int, str] = Field(..., example={0: "a beautiful lake", 30: "a cyberpunk character", 60: "dystopian future"})
    negative_prompts: Dict[int, str] = Field(..., example={0: "mountain, snow, fire"})
    
@app.post("/set_prompts")
async def set_prompts(prompts_data: PromptsModel):
    stored_prompts["pos_prompts"] = prompts_data.positive_prompts
    stored_prompts["neg_prompts"] = prompts_data.negative_prompts
    
    # Here you would use the prompts and neg_prompts in your application, 
    # perhaps saving them to a database or another data store.
    # For this example, I'm just returning them.
    return {"prompts": stored_prompts["pos_prompts"], "neg_prompts": stored_prompts["neg_prompts"]}

#@markdown **Load Settings**
# fill this setting_args function to return args_dict and anim_args_dict
def setting_args(root):
    args_dict = DeforumArgs(title_name)
    args = SimpleNamespace(**args_dict)
    
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    
    # Load clip model if using clip guidance
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
        root.clip_model = clip.load(args.clip_name, fit=False)[0].eval().requires_grad_(False).to(root.device)
        if (args.aesthetics_scale > 0):
            root.aesthetics_model = load_aesthetics_model(args, root)
            
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0
    anim_args_dict = DeforumAnimArgs()
    anim_args = SimpleNamespace(**anim_args_dict)
    
    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True
    
    return args, anim_args

# clean up unused memory
gc.collect()
torch.cuda.empty_cache()


def generate_frames():
    # get prompts
    cond, uncond = Prompts(prompt=stored_prompts["pos_prompts"],neg_prompt=stored_prompts["neg_prompts"]).as_dict()

    # args = setting_args(root)
    # anim_args = setting_anim_args()
    
    args, anim_args = setting_args(root)
    
    # dispatch to appropriate renderer
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        render_animation(root, anim_args, args, cond, uncond)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(root, anim_args, args, cond, uncond)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(root, anim_args, args, cond, uncond)
    else:
        render_image_batch(root, args, cond, uncond)
        

@app.post("/generate")
async def trigger_generate():
    args, _ = setting_args(root)
    generate_frames()
    create_video(args)
    return {"message": "Generating frames is done."}


"""
# Create Video From Frames
"""
def create_video(args):
    #@markdown **New Version**
    # skip_video_for_run_all = True #@param {type: 'boolean'}
    skip_video_for_run_all = False #@param {type: 'boolean'}
    # create_gif = False #@param {type: 'boolean'}
    create_gif = True #@param {type: 'boolean'}

    if skip_video_for_run_all == True:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
    else:

        from helpers.ffmpeg_helpers import get_extension_maxframes, get_auto_outdir_timestring, get_ffmpeg_path, make_mp4_ffmpeg, make_gif_ffmpeg, patrol_cycle

        def ffmpegArgs():
            ffmpeg_mode = "auto" #@param ["auto","manual","timestring"]
            ffmpeg_outdir = "" #@param {type:"string"}
            ffmpeg_timestring = "" #@param {type:"string"}
            ffmpeg_image_path = "" #@param {type:"string"}
            ffmpeg_mp4_path = "" #@param {type:"string"}
            ffmpeg_gif_path = "" #@param {type:"string"}
            ffmpeg_extension = "png" #@param {type:"string"}
            ffmpeg_maxframes = 200 #@param
            ffmpeg_fps = 15 #@param

            # determine auto paths
            if ffmpeg_mode == 'auto':
                ffmpeg_outdir, ffmpeg_timestring = get_auto_outdir_timestring(args,ffmpeg_mode)
            if ffmpeg_mode in ["auto","timestring"]:
                ffmpeg_extension, ffmpeg_maxframes = get_extension_maxframes(args,ffmpeg_outdir,ffmpeg_timestring)
                ffmpeg_image_path, ffmpeg_mp4_path, ffmpeg_gif_path = get_ffmpeg_path(ffmpeg_outdir, ffmpeg_timestring, ffmpeg_extension)
            return locals()

        ffmpeg_args_dict = ffmpegArgs()
        ffmpeg_args = SimpleNamespace(**ffmpeg_args_dict)
        make_mp4_ffmpeg(ffmpeg_args, display_ffmpeg=True, debug=False)
        if create_gif:
            make_gif_ffmpeg(ffmpeg_args, debug=False)
        #patrol_cycle(args,ffmpeg_args)

# Display test_api/2023-09/FastAPI video Test/20230922230904.mp4
# 2023-09 : YYYY-MM
# FastAPI video Test : args.batch_name
# 20230922230904 : args.timestring
@app.get("/display_video/{filename}")
async def display_video(request: Request, filename: str):
    args, _ = setting_args(root)
    video_path = Path(f"test_api/2023-09/{args.batch_name}/{filename}.mp4")
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    else:
        return {"message": "Video not found"}


# TODO: add upscale video
@app.post("/upscale-video")
def upscale_video(args, upscale=True):
    if upscale:
        create_video(args)
    return {"message": "Upscaling video is done."}
