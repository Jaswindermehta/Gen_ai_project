import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

print("Loading OpenAI CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to("cuda")

print("Loading Dataset Prompts...")
dataset = load_dataset("nateraw/parti-prompts", split="train")
test_prompts = dataset["Prompt"][:200]

output_dir = "parti_200_outputs"
total_clip_score = 0.0
valid_images = 0

print("Calculating CLIP Scores...")
# We use torch.no_grad() because we are only evaluating, not training
with torch.no_grad():
    for i, prompt in enumerate(test_prompts):
        img_path = os.path.join(output_dir, f"img_{i}.png")
        
        # Make sure the image actually exists
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Process text and image through CLIP
            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to("cuda")
            outputs = model(**inputs)
            
            # Extract the cosine similarity score
            score = outputs.logits_per_image.item()
            total_clip_score += score
            valid_images += 1
            
            if valid_images % 20 == 0:
                print(f"Evaluated {valid_images}/200...")
                
        except Exception as e:
            print(f"Error processing image {i}: {e}")

average_score = total_clip_score / valid_images
print(f"\n--- FINAL RESULTS ---")
print(f"Total Images Evaluated: {valid_images}")
print(f"Average DeepCache CLIP Score: {average_score:.2f}")
