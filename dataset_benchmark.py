import os
import torch
import time
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

# 1. Create a folder to hold our dataset outputs
output_dir = "partiprompts_outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the PartiPrompts Dataset used in the DeepCache Paper
print("Downloading PartiPrompts dataset from Hugging Face...")
# nateraw/parti-prompts contains the 1600+ prompts used by researchers
dataset = load_dataset("nateraw/parti-prompts", split="train")

# Grab a batch of 10 complex prompts from the benchmark
test_prompts = dataset["Prompt"][:10]

print(f"\nLoaded {len(test_prompts)} prompts. Examples:")
print(f"1. {test_prompts[0][:80]}...")
print(f"2. {test_prompts[1][:80]}...")

# 3. Load the Model
print("\nLoading Stable Diffusion v1.5 onto the RTX A5000...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")

# 4. Wrap with DeepCache
helper = DeepCacheSDHelper(pipe=pipe)
# Using the standard configuration to replicate the paper's claims
helper.set_params(cache_interval=3, cache_branch_id=0)

SEED = 42
generator = torch.Generator(device="cuda").manual_seed(SEED)

# ---------------------------------------------------------
# RUNNING PARTIPROMPTS WITH DEEPCACHE
# ---------------------------------------------------------
print("\nStarting DeepCache Batch Generation on PartiPrompts...")
helper.enable()

start_time = time.time()

for i, prompt in enumerate(test_prompts):
    print(f"Generating image {i+1}/10...")
    
    # Generate the image
    image = pipe(prompt, num_inference_steps=50, generator=generator).images[0]
    
    # Save it with a clean filename
    filename = os.path.join(output_dir, f"deepcache_parti_{i}.png")
    image.save(filename)

end_time = time.time()
helper.disable()

total_time = end_time - start_time
print(f"\nPartiPrompts Run Complete!")
print(f"Total time for 10 images: {total_time:.2f} seconds")
print(f"Average time per image: {total_time / 10:.2f} seconds")
