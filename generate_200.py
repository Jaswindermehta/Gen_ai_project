import os
import torch
import time
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

output_dir = "parti_200_outputs"
os.makedirs(output_dir, exist_ok=True)

print("Downloading PartiPrompts...")
dataset = load_dataset("nateraw/parti-prompts", split="train")
# Grab 200 prompts!
test_prompts = dataset["Prompt"][:200]

print("\nLoading Stable Diffusion v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")

# CRITICAL FOR BENCHMARKING: Disable the safety checker to prevent black images
pipe.safety_checker = None
pipe.requires_safety_checker = False

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)

SEED = 42
generator = torch.Generator(device="cuda").manual_seed(SEED)

print(f"\nGenerating {len(test_prompts)} images with DeepCache...")
helper.enable()
start_time = time.time()

for i, prompt in enumerate(test_prompts):
    # Print progress every 10 images so your terminal isn't flooded
    if i % 10 == 0:
        print(f"Processing image {i}/{len(test_prompts)}...")
        
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    image.save(os.path.join(output_dir, f"img_{i}.png"))

total_time = time.time() - start_time
helper.disable()

print(f"\nDone! Generated 200 images in {total_time:.2f} seconds.")
print(f"Average: {total_time / 200:.2f} seconds per image.")
