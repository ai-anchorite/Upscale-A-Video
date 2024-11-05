import gradio as gr
import torch
import os
import warnings
import imageio
import numpy as np
from einops import rearrange
from torch.nn import functional as F
import transformers
transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)

from models_video.RAFT.raft_bi import RAFT_bi
from models_video.propagation_module import Propagation
from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
from models_video.unet_video import UNetVideoModel
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline
from models_video.scheduling_ddim import DDIMScheduler
from models_video.color_correction import wavelet_reconstruction, adaptive_instance_normalization
from llava.llava_agent import LLavaAgent
from utils import read_frame_from_videos, VIDEO_EXTENSIONS
from configs.CKPT_PTH import LLAVA_MODEL_PATH

def load_models():
    # Load models with the same setup as in the original script
    if torch.cuda.device_count() >= 2:
        UAV_device = 'cuda:0'
        LLaVA_device = 'cuda:1'
    elif torch.cuda.device_count() == 1:
        UAV_device = 'cuda:0'
        LLaVA_device = 'cuda:0'
    else:
        raise ValueError('Currently support CUDA only.')
    
    pipeline = VideoUpscalePipeline.from_pretrained("./pretrained_models/upscale_a_video", torch_dtype=torch.float16)
    
    # Load VAE
    pipeline.vae = AutoencoderKLVideo.from_config("./pretrained_models/upscale_a_video/vae/vae_3d_config.json")
    pipeline.vae.load_state_dict(torch.load("./pretrained_models/upscale_a_video/vae/vae_3d.bin", map_location="cpu"))
    
    # Load UNet
    pipeline.unet = UNetVideoModel.from_config("./pretrained_models/upscale_a_video/unet/unet_video_config.json")
    pipeline.unet.load_state_dict(torch.load("./pretrained_models/upscale_a_video/unet/unet_video.bin", map_location="cpu"))
    pipeline.unet = pipeline.unet.half()
    pipeline.unet.eval()
    
    # Load scheduler
    pipeline.scheduler = DDIMScheduler.from_config("./pretrained_models/upscale_a_video/scheduler/scheduler_config.json")
    
    # Load LLaVA
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device)
    
    # Load RAFT and Propagator
    raft = RAFT_bi("./pretrained_models/upscale_a_video/propagator/raft-things.pth")
    propagator = Propagation(4, learnable=False)
    
    pipeline = pipeline.to(UAV_device)
    return pipeline, llava_agent, raft, propagator, UAV_device

def upscale_video(video_path, use_llava, additional_prompt, negative_prompt, noise_level, 
                  guidance_scale, inference_steps, propagation_steps, color_fix):
    # Convert propagation steps string to list
    if propagation_steps.strip():
        prop_steps = [int(x.strip()) for x in propagation_steps.split(',')]
    else:
        prop_steps = []
    
    # Load models if not already loaded
    if not hasattr(upscale_video, 'models_loaded'):
        upscale_video.pipeline, upscale_video.llava_agent, upscale_video.raft, \
        upscale_video.propagator, upscale_video.device = load_models()
        upscale_video.models_loaded = True
    
    # Process video
    vframes, fps, size, video_name = read_frame_from_videos(video_path)
    
    # Generate caption with LLaVA if requested
    if use_llava:
        with torch.no_grad():
            video_img0 = vframes[0]
            w, h = video_img0.shape[-1], video_img0.shape[-2]
            fix_resize = 512
            _upsacle = fix_resize / min(w, h)
            w *= _upsacle
            h *= _upsacle
            w0, h0 = round(w), round(h)
            video_img0 = F.interpolate(video_img0.unsqueeze(0).float(), size=(h0, w0), mode='bicubic')
            video_img0 = (video_img0.squeeze(0).permute(1, 2, 0)).cpu().numpy().clip(0, 255).astype(np.uint8)
            video_caption = upscale_video.llava_agent.gen_image_caption([Image.fromarray(video_img0)])[0]
    else:
        video_caption = ""
    
    prompt = video_caption + " " + additional_prompt
    
    # Prepare video frames
    vframes = (vframes/255. - 0.5) * 2  # T C H W [-1, 1]
    vframes = vframes.to(upscale_video.device)
    
    h, w = vframes.shape[-2:]
    if h >= 1280 and w >= 1280:
        vframes = F.interpolate(vframes, (int(h//4), int(w//4)), mode='area')
    
    vframes = vframes.unsqueeze(dim=0)  # 1 T C H W
    vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()  # 1 C T H W
    
    # Process flows if needed
    if prop_steps:
        flows_forward, flows_backward = upscale_video.raft.forward_slicing(vframes)
        flows_bi = [flows_forward, flows_backward]
        upscale_video.pipeline.propagator = upscale_video.propagator
    else:
        flows_bi = None
        upscale_video.pipeline.propagator = None
    
    # Generate output
    generator = torch.Generator(device=upscale_video.device).manual_seed(10)
    
    with torch.no_grad():
        output = upscale_video.pipeline(
            prompt,
            image=vframes,
            flows_bi=flows_bi,
            generator=generator,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            propagation_steps=prop_steps,
        ).images
    
    # Apply color correction if requested
    if color_fix in ['AdaIn', 'Wavelet']:
        vframes = rearrange(vframes.squeeze(0), 'c t h w -> t c h w').contiguous()
        output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
        vframes = F.interpolate(vframes, scale_factor=4, mode='bicubic')
        if color_fix == 'AdaIn':
            output = adaptive_instance_normalization(output, vframes)
        elif color_fix == 'Wavelet':
            output = wavelet_reconstruction(output, vframes)
    else:
        output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
    
    # Prepare output video
    output = output.cpu()
    upscaled_video = (output / 2 + 0.5).clamp(0, 1) * 255
    upscaled_video = rearrange(upscaled_video, 't c h w -> t h w c').contiguous()
    upscaled_video = upscaled_video.cpu().numpy().astype(np.uint8)
    
    # Save and return result
    output_path = f"results/upscaled_{os.path.basename(video_path)}"
    os.makedirs("results", exist_ok=True)
    imageio.mimwrite(output_path, upscaled_video, fps=fps, quality=8)
    
    return output_path, f"Generated caption: {video_caption}" if use_llava else None


# Create Gradio interface
with gr.Blocks(title="Video Upscaling with Upscale-A-Video") as interface:
    gr.Markdown("""
    # Upscale-A-Video
    Upload a video to upscale its resolution using the Upscale-A-Video model.
    """)
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            use_llava = gr.Checkbox(label="Use LLaVA for Caption Generation", value=False)
            additional_prompt = gr.Textbox(label="Additional Prompt", 
                                        value="best quality, extremely detailed")
            negative_prompt = gr.Textbox(label="Negative Prompt", 
                                       value="blur, worst quality")
            
        with gr.Column():
            noise_level = gr.Slider(label="Noise Level", minimum=0, maximum=200, 
                                  value=120, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=20, 
                                     value=6, step=0.1)
            inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, 
                                      value=30, step=1)
            propagation_steps = gr.Textbox(label="Propagation Steps (comma-separated)", 
                                         value="", placeholder="e.g., 24,26,28")
            color_fix = gr.Radio(label="Color Fix", choices=["None", "AdaIn", "Wavelet"], 
                               value="None")
    
    with gr.Row():
        submit_btn = gr.Button("Upscale Video")
    
    with gr.Row():
        output_video = gr.Video(label="Upscaled Video")
        caption_output = gr.Textbox(label="LLaVA Caption")
    
    submit_btn.click(
        fn=upscale_video,
        inputs=[
            input_video, use_llava, additional_prompt, negative_prompt,
            noise_level, guidance_scale, inference_steps, propagation_steps, color_fix
        ],
        outputs=[output_video, caption_output]
    )
    

interface.launch(share=False)
