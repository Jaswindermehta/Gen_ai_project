import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

print("Loading OpenAI CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
# Using our safetensors fix to bypass the security block!
model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to("cuda")

print("Loading Text-Only COCO 2017 Prompts...")
# Using the exact same dataset and split we used for generation
dataset = load_dataset("Lakonik/t2i-prompts-coco-10k", split="test")
test_prompts = dataset["prompt"][:200]

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
                
                # Print progress so we know it hasn't frozen
                if valid_images % 50 == 0:
                    print(f"  Evaluated {valid_images}/200...")
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                
    return total_clip_score / valid_images if valid_images > 0 else 0

print("\n--- Scoring Baseline Images (The Control) ---")
baseline_score = calculate_folder_clip("coco_baseline")

print("\n--- Scoring DeepCache Images (The Experiment) ---")
deepcache_score = calculate_folder_clip("coco_deepcache")

delta = baseline_score - deepcache_score

print(f"\n=========================================")
print(f"   FINAL COCO 2017 CLIP SCORES (200 imgs)")
print(f"=========================================")
print(f"Baseline Stable Diffusion: {baseline_score:.2f}")
print(f"DeepCache Accelerated:     {deepcache_score:.2f}")
print(f"-----------------------------------------")
print(f"Score Difference (Delta):  {delta:.2f}")
print(f"=========================================")
