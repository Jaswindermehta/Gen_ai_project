from datasets import load_dataset

# Load the exact same dataset
dataset = load_dataset("nateraw/parti-prompts", split="train")

print("--- The First 10 PartiPrompts ---")
for i in range(10):
    # We pull the exact prompt text for each image index
    prompt_text = dataset["Prompt"][i]
    category = dataset["Category"][i] # The dataset also tags what type of challenge it is!
    
    print(f"\nImage {i} (deepcache_parti_{i}.png):")
    print(f"Category: {category}")
    print(f"Prompt: {prompt_text}")
