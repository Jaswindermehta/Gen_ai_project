import os
import torch
import time
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

# --- CONFIGURATION ---
NUM_IMAGES = 5000  
BATCH_SIZE = 4     
SEED = 42

base_dir = f"coco_baseline_{NUM_IMAGES}"
cache_dir = f"coco_deepcache_{NUM_IMAGES}"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

print("Loading Text-Only COCO Dataset...")
dataset = load_dataset("Lakonik/t2i-prompts-coco-10k", split="test")
dataset = dataset.shuffle(seed=SEED).select(range(NUM_IMAGES))
prompts = dataset["prompt"]

print("\nLoading Stable Diffusion v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16,
    variant="fp16" # Keeping the space-saving variant
).to("cuda")
pipe.safety_checker = None
pipe.requires_safety_checker = False

print(f"\n--- Generating {NUM_IMAGES} Baseline Images (Batched) ---")
start_time = time.time()
for i in range(0, NUM_IMAGES, BATCH_SIZE):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    
    # --- RESUME LOGIC: Check if batch already exists ---
    skip_batch = True
    for j in range(len(batch_prompts)):
        if not os.path.exists(os.path.join(base_dir, f"img_{i+j}.png")):
            skip_batch = False
            break
            
    if skip_batch:
        if i % 100 == 0: 
            print(f"Skipping Baseline: {i}/{NUM_IMAGES} (Already exists)...")
        continue
    # ---------------------------------------------------
    
    generators = [torch.Generator(device="cuda").manual_seed(SEED + i + j) for j in range(len(batch_prompts))]
    images = pipe(batch_prompts, num_inference_steps=50, generator=generators).images
    
    for j, img in enumerate(images):
        img.save(os.path.join(base_dir, f"img_{i+j}.png"))
        
    if i % 100 == 0: 
        print(f"Baseline: {i}/{NUM_IMAGES}...")

print(f"\n--- Generating {NUM_IMAGES} DeepCache Images (Batched) ---")
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)
helper.enable()

for i in range(0, NUM_IMAGES, BATCH_SIZE):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    
    # --- RESUME LOGIC: Check if batch already exists ---
    skip_batch = True
    for j in range(len(batch_prompts)):
        if not os.path.exists(os.path.join(cache_dir, f"img_{i+j}.png")):
            skip_batch = False
            break
            
    if skip_batch:
        if i % 100 == 0: 
            print(f"Skipping DeepCache: {i}/{NUM_IMAGES} (Already exists)...")
        continue
    # ---------------------------------------------------
    
    generators = [torch.Generator(device="cuda").manual_seed(SEED + i + j) for j in range(len(batch_prompts))]
    images = pipe(batch_prompts, num_inference_steps=50, generator=generators).images
    
    for j, img in enumerate(images):
        img.save(os.path.join(cache_dir, f"img_{i+j}.png"))
        
    if i % 100 == 0: 
        print(f"DeepCache: {i}/{NUM_IMAGES}...")

helper.disable()
print(f"\nGeneration Complete in {(time.time() - start_time)/60:.2f} minutes!")