import torch
import time
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")

helper = DeepCacheSDHelper(pipe=pipe)
# We will use branch_id=1 this time. It is slightly slower than 0, 
# but preserves much better visual quality for complex landscapes.
helper.set_params(cache_interval=3, cache_branch_id=1)

prompt = "a photo of an astronaut on a moon"

# ---------------------------------------------------------
# THE MAGIC FIX: We lock the starting noise with a Seed
# ---------------------------------------------------------
SEED = 12345 

print("\n--- Running Baseline ---")
generator = torch.Generator(device="cuda").manual_seed(SEED)
start_time = time.time()
image_base = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
print(f"Baseline Time: {time.time() - start_time:.2f} seconds")
image_base.save("true_astronaut__baseline.png")

print("\n--- Running DeepCache ---")
# We reset the exact same seed so we start from the exact same noise!
generator = torch.Generator(device="cuda").manual_seed(SEED)
helper.enable()
start_time = time.time()
image_cache = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
print(f"DeepCache Time: {time.time() - start_time:.2f} seconds")
image_cache.save("true_astronaut_deepcache.png")
helper.disable()

print("\nSaved true_as_baseline.png and true_as_deepcache.png")
