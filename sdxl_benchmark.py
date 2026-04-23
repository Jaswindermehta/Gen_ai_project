import torch
import time
from diffusers import StableDiffusionXLPipeline
from DeepCache import DeepCacheSDHelper

print("Loading SDXL onto the RTX A5000...")
print("(Note: SDXL is a massive model, so the initial download will take a few minutes)")

# 1. Load the SDXL Pipeline
# We strictly use fp16 and safetensors to ensure it fits efficiently in your 24GB VRAM
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to("cuda")

# 2. Wrap the pipeline with DeepCache
helper = DeepCacheSDHelper(pipe=pipe)

# We use branch_id=0 for maximum speedup on this massive model
helper.set_params(cache_interval=3, cache_branch_id=0)

prompt = "A majestic cyberpunk tiger walking through a neon-lit futuristic alleyway, 8k resolution, highly detailed, photorealistic"
SEED = 42

# ---------------------------------------------------------
# TRUE A/B BENCHMARK FOR SDXL
# ---------------------------------------------------------
print("\n--- Running SDXL Baseline (Standard) ---")
generator = torch.Generator(device="cuda").manual_seed(SEED)
start_time = time.time()
# SDXL defaults to 50 steps, generating a 1024x1024 image
image_base = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
print(f"SDXL Baseline Time: {time.time() - start_time:.2f} seconds")
image_base.save("sdxl_baseline.png")

print("\n--- Running SDXL with DeepCache ---")
generator = torch.Generator(device="cuda").manual_seed(SEED)
helper.enable()
start_time = time.time()
image_cache = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
print(f"SDXL DeepCache Time: {time.time() - start_time:.2f} seconds")
image_cache.save("sdxl_deepcache.png")
helper.disable()

print("\nSaved sdxl_baseline.png and sdxl_deepcache.png")
