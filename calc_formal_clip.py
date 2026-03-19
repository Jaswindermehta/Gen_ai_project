import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

print("Loading Official High-Resolution CLIP Model (ViT-L/14)...")
# UPGRADE: This is the heavy academic standard for final publication scores
model_id = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to("cuda")

print("Loading the exact same randomized 200 text prompts...")
dataset = load_dataset("Lakonik/t2i-prompts-coco-10k", split="test")
# We use the exact same seed so the evaluator reads the exact same 200 random prompts!
dataset = dataset.shuffle(seed=42).select(range(200))
test_prompts = dataset["prompt"]

def calculate_folder_clip(folder_name):
    total_clip_score = 0.0
    valid_images = 0
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            img_path = os.path.join(folder_name, f"img_{i}.png")
            
            if not os.path.exists(img_path):
                continue
                
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to("cuda")
                outputs = model(**inputs)
                
                score = outputs.logits_per_image.item()
                total_clip_score += score
                valid_images += 1
                
                if valid_images % 50 == 0:
                    print(f"  Evaluated {valid_images}/200...")
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                
    return total_clip_score / valid_images if valid_images > 0 else 0

print("\n--- Scoring Baseline Images (The Control) ---")
baseline_score = calculate_folder_clip("coco_baseline_random")

print("\n--- Scoring DeepCache Images (The Experiment) ---")
deepcache_score = calculate_folder_clip("coco_deepcache_random")

delta = baseline_score - deepcache_score

print(f"\n=========================================")
print(f" ACADEMIC COCO 2017 CLIP SCORES (ViT-L/14)")
print(f"=========================================")
print(f"Baseline Stable Diffusion: {baseline_score:.2f}")
print(f"DeepCache Accelerated:     {deepcache_score:.2f}")
print(f"-----------------------------------------")
print(f"Score Difference (Delta):  {delta:.2f}")
print(f"=========================================")
