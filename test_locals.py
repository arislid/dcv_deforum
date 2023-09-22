
import subprocess, time, gc, os, sys
#@markdown **Environment Setup**
def setup_environment():
    sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src', 'src'])
    print("..skipping setup")

setup_environment()

from deforumargs import DeforumArgs, DeforumAnimArgs
from types import SimpleNamespace
import clip
import random
from helpers.aesthetics import load_aesthetics_model

args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

def setting_args(args_dict, anim_args_dict):
    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    # How to deal with root variable?
    # Load clip model if using clip guidance
    # if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    #     root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    #     if (args.aesthetics_scale > 0):
    #         root.aesthetics_model = load_aesthetics_model(args, root)

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
    return locals()

print(setting_args(args_dict, anim_args_dict)['args_dict'])
print(setting_args(args_dict, anim_args_dict)['anim_args_dict'])