import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', FutureWarning)

import os
import gc
import gradio as gr
import torch
import imageio
import numpy as np
from einops import rearrange
from torch.nn import functional as F
from torch.cuda import empty_cache
import transformers
transformers.logging.set_verbosity_error()

from models_video.RAFT.raft_bi import RAFT_bi
from models_video.propagation_module import Propagation
from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
from models_video.unet_video import UNetVideoModel
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline
from models_video.scheduling_ddim import DDIMScheduler
from models_video.color_correction import wavelet_reconstruction, adaptive_instance_normalization
from utils import read_frame_from_videos, VIDEO_EXTENSIONS

def clear_memory():
    """Aggressive memory clearing"""
    empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class MemoryEfficientVideoUpscaler:
    def __init__(self, max_frames_per_chunk=4, vae_batch_size=1):
        self.max_frames_per_chunk = max_frames_per_chunk
        self.vae_batch_size = vae_batch_size
        self.models_loaded = False
        
    def load_models(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Load pipeline with memory efficient settings
        print('Loading Upscale-A-Video pipeline...')
        self.pipeline = VideoUpscalePipeline.from_pretrained(
            "./pretrained_models/upscale_a_video", 
            torch_dtype=torch.float16
        )
        
        # Configure UNet for memory efficiency
        self.pipeline.unet = UNetVideoModel.from_config(
            "./pretrained_models/upscale_a_video/unet/unet_video_config.json"
        )
        self.pipeline.unet.load_state_dict(
            torch.load("./pretrained_models/upscale_a_video/unet/unet_video.bin", map_location="cpu")
        )
        self.pipeline.unet = self.pipeline.unet.half()
        self.pipeline.unet.eval()
        
        # Enable memory efficient attention and gradient checkpointing
        if hasattr(self.pipeline.unet, 'enable_gradient_checkpointing'):
            self.pipeline.unet.enable_gradient_checkpointing()
        
        # Load VAE in eval mode
        self.pipeline.vae = AutoencoderKLVideo.from_config(
            "./pretrained_models/upscale_a_video/vae/vae_3d_config.json"
        )
        self.pipeline.vae.load_state_dict(
            torch.load("./pretrained_models/upscale_a_video/vae/vae_3d.bin", map_location="cpu")
        )
        self.pipeline.vae.eval()
        
        # Load scheduler
        self.pipeline.scheduler = DDIMScheduler.from_config(
            "./pretrained_models/upscale_a_video/scheduler/scheduler_config.json"
        )
        
        self.pipeline = self.pipeline.to(self.device)
        self.raft = None
        self.propagator = None
        
        self.models_loaded = True

    def decode_latents_in_chunks(self, latents):
        """Memory efficient VAE decoding"""
        b, c, t, h, w = latents.shape
        chunk_size = self.vae_batch_size
        decoded_chunks = []
        
        for i in range(0, t, chunk_size):
            chunk = latents[:, :, i:i+chunk_size]
            with torch.no_grad():
                decoded_chunk = self.pipeline.vae.decode(chunk).sample
            decoded_chunks.append(decoded_chunk.cpu())
            clear_memory()
            
        return torch.cat(decoded_chunks, dim=2)

    def process_chunk(self, chunk, prompt, flows_bi, **kwargs):
        """Process a chunk of frames with memory optimization"""
        clear_memory()
        
        with torch.no_grad():
            # Process through pipeline
            output = self.pipeline(
                prompt,
                image=chunk,
                flows_bi=flows_bi,
                **kwargs
            ).images
                
        clear_memory()
        return output

    def upscale_video(self, video_path, prompt, negative_prompt,
                      noise_level, guidance_scale, inference_steps, propagation_steps, color_fix):
        """Main video processing function"""
        if not self.models_loaded:
            self.load_models()
        
        # Process propagation steps
        if propagation_steps.strip():
            prop_steps = [int(x.strip()) for x in propagation_steps.split(',')]
            if self.raft is None:
                print('Loading RAFT and propagator...')
                self.raft = RAFT_bi("./pretrained_models/upscale_a_video/propagator/raft-things.pth")
                self.propagator = Propagation(4, learnable=False)
        else:
            prop_steps = []
            self.raft = None
            self.propagator = None
            
        self.pipeline.propagator = self.propagator
        
        # Read and prepare video
        vframes, fps, size, video_name = read_frame_from_videos(video_path)
        vframes = (vframes/255. - 0.5) * 2
        vframes = vframes.to(self.device)
        
        # Resize if needed
        h, w = vframes.shape[-2:]
        if h >= 1280 and w >= 1280:
            vframes = F.interpolate(vframes, (int(h//4), int(w//4)), mode='area')
            
        vframes = vframes.unsqueeze(0)
        vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()
        
        # Calculate flows if needed
        if self.raft is not None:
            flows_forward, flows_backward = self.raft.forward_slicing(vframes)
            flows_bi = [flows_forward, flows_backward]
        else:
            flows_bi = None
            
        # Process in chunks
        b, c, t, h, w = vframes.shape
        chunk_size = min(t, self.max_frames_per_chunk)
        output_chunks = []
        
        for i in range(0, t, chunk_size):
            print(f'Processing frames {i} to {min(i+chunk_size, t)}...')
            chunk = vframes[:, :, i:i+chunk_size]
            
            if flows_bi is not None:
                flows_bi_chunk = [
                    flows_bi[0][:, :, i:i+chunk_size],
                    flows_bi[1][:, :, i:i+chunk_size]
                ]
            else:
                flows_bi_chunk = None
                
            generator = torch.Generator(device=self.device).manual_seed(10)
            
            chunk_output = self.process_chunk(
                chunk,
                prompt=prompt,
                flows_bi=flows_bi_chunk,
                generator=generator,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                noise_level=noise_level,
                negative_prompt=negative_prompt,
                propagation_steps=prop_steps
            )
            
            output_chunks.append(chunk_output)
            clear_memory()
            
        # Combine chunks and apply color correction
        output = torch.cat(output_chunks, dim=2)
        
        if color_fix in ['AdaIn', 'Wavelet']:
            vframes = rearrange(vframes.squeeze(0), 'c t h w -> t c h w').contiguous()
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
            vframes = F.interpolate(vframes, scale_factor=4, mode='bicubic')
            
            # Process color correction in chunks if needed
            if color_fix == 'AdaIn':
                output = adaptive_instance_normalization(output, vframes)
            elif color_fix == 'Wavelet':
                output = wavelet_reconstruction(output, vframes)
        else:
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
            
        # Save result
        output = output.cpu()
        upscaled_video = (output / 2 + 0.5).clamp(0, 1) * 255
        upscaled_video = rearrange(upscaled_video, 't c h w -> t h w c').contiguous()
        upscaled_video = upscaled_video.numpy().astype(np.uint8)
        
        os.makedirs("results", exist_ok=True)
        output_path = f"results/upscaled_{os.path.basename(video_path)}"
        
        # Save video in chunks to avoid memory spike
        imageio.mimwrite(output_path, upscaled_video, fps=fps, quality=8)
        clear_memory()
        
        return output_path


# Use smaller chunk sizes for memory efficiency
upscaler = MemoryEfficientVideoUpscaler(max_frames_per_chunk=4, vae_batch_size=1)
    
with gr.Blocks(title="Memory-Efficient Video Upscaling") as interface:
    gr.Markdown("""
    # Memory-Efficient Upscale-A-Video
    Upload a video to upscale its resolution using a memory-optimized version of Upscale-A-Video.
    """)
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            prompt = gr.Textbox(
                label="Prompt",
                value="best quality, extremely detailed"
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blur, worst quality"
            )
            
        with gr.Column():
            noise_level = gr.Slider(
                label="Noise Level",
                minimum=0,
                maximum=200,
                value=120,
                step=1
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0,
                maximum=20,
                value=6,
                step=0.1
            )
            inference_steps = gr.Slider(
                label="Inference Steps",
                minimum=1,
                maximum=100,
                value=30,
                step=1
            )
            propagation_steps = gr.Textbox(
                label="Propagation Steps (comma-separated)",
                value="",
                placeholder="e.g., 24,26,28"
            )
            color_fix = gr.Radio(
                label="Color Fix",
                choices=["None", "AdaIn", "Wavelet"],
                value="None"
            )
    
    with gr.Row():
        submit_btn = gr.Button("Upscale Video")
        
    with gr.Row():
        output_video = gr.Video(label="Upscaled Video")
        
    submit_btn.click(
        fn=upscaler.upscale_video,
        inputs=[
            input_video, prompt, negative_prompt,
            noise_level, guidance_scale, inference_steps, propagation_steps, color_fix
        ],
        outputs=output_video
    )
    
interface.launch(share=False)
