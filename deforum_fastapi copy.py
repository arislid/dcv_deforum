
import subprocess, time, gc, os, sys
#@markdown **Environment Setup**
def setup_environment():
    sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src', 'src'])
    print("..skipping setup")

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

from fastapi import FastAPI, Depends



#@markdown **Path Setup**

def PathSetup():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}
    return locals()

root = SimpleNamespace(**PathSetup())
root.models_path, root.output_path = get_model_output_paths(root)

#@markdown **Model Setup**

def ModelSetup():
    map_location = "cuda" #@param ["cpu", "cuda"]
    model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
    model_checkpoint =  "Protogen_V2.2.ckpt" #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = "" #@param {type:"string"}
    return locals()

root.__dict__.update(ModelSetup())
root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

"""
# Settings
"""


# prompts
prompts = {
    0: "a beautiful lake by Asher Brown Durand, trending on Artstation",   
    30: "a portrait for a cyberpunk handsome male character who is a mob boss with a dark cunning intelect, \
        include cyberware and a cybernetic eye, \
        trending on Artstation",
    60: "black and white sketch of a stoic, wisdom, strength, eye catching" 
}

neg_prompts = {
    0: "mountain",
}

#@markdown **Load Settings**
# override_settings_with_file = False #@param {type:"boolean"}
# settings_file = "custom" #@param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
# custom_settings_file = "/content/drive/MyDrive/Settings.txt"#@param {type:"string"}


from deforumargs import DeforumArgs, DeforumAnimArgs

args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

# if override_settings_with_file:
#     load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

# fill this setting_args function to return args_dict and anim_args_dict
def setting_args():
    
    return locals()

args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

args.timestring = time.strftime('%Y%m%d%H%M%S')
args.strength = max(0.0, min(1.0, args.strength))

# Load clip model if using clip guidance
if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
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

if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

# clean up unused memory
gc.collect()
torch.cuda.empty_cache()

# get prompts
cond, uncond = Prompts(prompt=prompts,neg_prompt=neg_prompts).as_dict()

# dispatch to appropriate renderer
if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
    render_animation(root, anim_args, args, cond, uncond)
elif anim_args.animation_mode == 'Video Input':
    render_input_video(root, anim_args, args, cond, uncond)
elif anim_args.animation_mode == 'Interpolation':
    render_interpolation(root, anim_args, args, cond, uncond)
else:
    render_image_batch(root, args, cond, uncond)


"""
# Create Video From Frames
"""
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
