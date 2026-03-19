import torch
import time
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
from DeepCache import DeepCacheSDHelper

print("Loading Stable Video Diffusion (SVD-XT) onto the RTX A5000...")
print("This is a massive ~9GB model, the download will take a few minutes!")

# 1. Load the SVD Pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")

# 2. Wrap the pipeline
helper = DeepCacheSDHelper(pipe=pipe)

# For video, the paper recommends interval=3 or 4, and branch=0 for maximum speed
helper.set_params(cache_interval=3, cache_branch_id=0)

# 3. Load your exact SDXL Cyberpunk Tiger image!
# SVD XT expects a specific aspect ratio, so we resize it slightly
print("\nLoading your Cyberpunk Tiger image...")
init_image = Image.open("sdxl_baseline.png").convert("RGB")
init_image = init_image.resize((1024, 576)) 

SEED = 42

# ---------------------------------------------------------
# TRUE A/B BENCHMARK FOR VIDEO
# ---------------------------------------------------------
print("\n--- Running Video Baseline (This will be slow!) ---")
generator = torch.Generator(device="cuda").manual_seed(SEED)
start_time = time.time()
# decode_chunk_size=8 prevents VRAM overflow on the A5000
frames_base = pipe(init_image, decode_chunk_size=8, generator=generator).frames[0]
print(f"Video Baseline Time: {time.time() - start_time:.2f} seconds")
export_to_video(frames_base, "video_baseline.mp4", fps=7)

print("\n--- Running Video with DeepCache ---")
generator = torch.Generator(device="cuda").manual_seed(SEED)
helper.enable()
start_time = time.time()
frames_cache = pipe(init_image, decode_chunk_size=8, generator=generator).frames[0]
print(f"Video DeepCache Time: {time.time() - start_time:.2f} seconds")
export_to_video(frames_cache, "video_deepcache.mp4", fps=7)
helper.disable()

print("\nSaved video_baseline.mp4 and video_deepcache.mp4")
