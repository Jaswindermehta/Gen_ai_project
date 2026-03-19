import torch
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.io import read_image

print("Loading Inception-v3 Model for Formal Relative FID...")
# Using the strict academic 2048 feature layer
fid = FrechetInceptionDistance(feature=2048).to("cuda")

# Pointing to the newly generated randomized folders
# Update these to match your actual folder names!
base_dir = "coco_baseline_5000"  # Assuming you named the baseline folder this
cache_dir = "coco_deepcache_5000"

# --- THE BIG UPGRADE ---
num_images = 5000 
batch_size = 50  # Batched to protect your A5000 VRAM

print(f"\nProcessing Randomized Baseline Images ({num_images} images)...")
base_images = []
valid_base_count = 0
for i in range(num_images):
    img_path = os.path.join(base_dir, f"img_{i}.png")
    if os.path.exists(img_path):
        base_images.append(read_image(img_path))
        valid_base_count += 1
        
    if len(base_images) == batch_size or i == num_images - 1:
        if len(base_images) > 0:
            batch_tensor = torch.stack(base_images).to("cuda")
            fid.update(batch_tensor, real=True)
            base_images = [] 
            # Tweak: Only print every 500 so we don't spam your terminal
            if valid_base_count % 500 == 0:
                print(f"  Passed {valid_base_count}/{num_images} baseline images through Inception...")

print(f"\nProcessing Randomized DeepCache Images ({num_images} images)...")
cache_images = []
valid_cache_count = 0
for i in range(num_images):
    img_path = os.path.join(cache_dir, f"img_{i}.png")
    if os.path.exists(img_path):
        cache_images.append(read_image(img_path))
        valid_cache_count += 1
        
    if len(cache_images) == batch_size or i == num_images - 1:
        if len(cache_images) > 0:
            batch_tensor = torch.stack(cache_images).to("cuda")
            fid.update(batch_tensor, real=False)
            cache_images = [] 
            # Tweak: Only print every 500 so we don't spam your terminal
            if valid_cache_count % 500 == 0:
                print(f"  Passed {valid_cache_count}/{num_images} DeepCache images through Inception...")

print("\nCalculating Final Covariance Matrix Distance...")
final_fid = fid.compute()

print("\n=========================================")
print(f"  RANDOMIZED RELATIVE FID (5000 imgs)")
print(f"=========================================")
print(f"Total Image Pairs Evaluated: {valid_cache_count}")
print(f"Relative FID Score: {final_fid.item():.4f}")
print(f"=========================================")