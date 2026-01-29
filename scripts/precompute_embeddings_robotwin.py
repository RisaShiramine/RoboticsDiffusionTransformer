
import os
import json
import hashlib
import torch
import fnmatch
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer

# Configuration
DATASET_ROOT = "data/datasets/robotwin" # This is the link we created
INSTRUCTION_DIR = "/mnt/hdd/RoboTwin/data/stack_blocks_three/demo_clean/instructions" # Fallback/Original path
OUTPUT_DIR = "data/embeddings/robotwin"

def get_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main():
    # 1. Gather all unique instructions
    print("Gathering instructions...")
    unique_instructions = set()
    
    # Try using the instruction dir relative to dataset link first
    scan_dir = INSTRUCTION_DIR
    if not os.path.exists(scan_dir):
        # Try to deduce from dataset root
        scan_dir = os.path.abspath(os.path.join(DATASET_ROOT, "../instructions"))
    
    if not os.path.exists(scan_dir):
        print(f"Error: Instruction directory not found at {scan_dir} or {INSTRUCTION_DIR}")
        return

    json_files = []
    for root, _, files in os.walk(scan_dir):
        for filename in fnmatch.filter(files, '*.json'):
            json_files.append(os.path.join(root, filename))
    
    print(f"Found {len(json_files)} instruction files.")
    
    for jf in tqdm(json_files, desc="Parsing JSONs"):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                if 'seen' in data:
                    unique_instructions.update(data['seen'])
                if 'unseen' in data:
                    unique_instructions.update(data['unseen'])
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    print(f"Found {len(unique_instructions)} unique instructions.")
    if len(unique_instructions) == 0:
        print("No instructions found? Exiting.")
        return

    # 2. Load Model (Quantized for 4080/16GB VRAM)
    print("Loading T5-XXL model (8-bit)...")
    model_name = "google/t5-v1_1-xxl"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        model = T5EncoderModel.from_pretrained(
            model_name, 
            load_in_8bit=True, 
            device_map="auto",
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Make sure you have bitsandbytes and accelerate installed: pip install bitsandbytes accelerate")
        return

    # 3. Compute and Save Embeddings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = {}
    
    model.eval()
    
    print("Encoding instructions...")
    with torch.no_grad():
        for text in tqdm(unique_instructions, desc="Encoding"):
            text_hash = get_hash(text)
            save_path = os.path.join(OUTPUT_DIR, f"{text_hash}.pt")
            
            # Map text to absolute path
            mapping[text] = os.path.abspath(save_path)
            
            if os.path.exists(save_path):
                continue
                
            # Tokenize and Encode
            tokens = tokenizer(
                text, 
                return_tensors="pt", 
                padding="longest", 
                truncation=True,
                max_length=128 # Reasonable length for instructions
            )
            
            input_ids = tokens.input_ids.to(model.device)
            attn_mask = tokens.attention_mask.to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            embeddings = outputs.last_hidden_state.detach().cpu() # (1, Seq, Dim)
            
            # Remove batch dim
            embeddings = embeddings[0] # (Seq, Dim)
            
            # Save
            torch.save(embeddings, save_path)
            
    # 4. Save Mapping
    mapping_path = os.path.join(OUTPUT_DIR, "mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
        
    print(f"Done! Saved {len(unique_instructions)} embeddings to {OUTPUT_DIR}")
    print(f"Mapping saved to {mapping_path}")

if __name__ == "__main__":
    main()
