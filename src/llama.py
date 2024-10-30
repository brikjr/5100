import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Constants remain the same
INPUT_CSV = '../data/Books.csv'
OUTPUT_CSV = '../data/Books_with_descriptions.csv'
CHECKPOINT_INTERVAL = 100
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def clean_description(text):
    """Enhanced cleaning of model output."""
    # Remove common prefixes
    prefixes_to_remove = [
        r'Here is a one-sentence summary of.*?:',
        r'Sure! Here is a one-sentence.*?:',
        r'Here\'s a one-sentence description.*?:',
        r'\[INST\].*?\[/INST\]',
        r'This book is',
        r'The book is',
        r'This is',
    ]
    
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Clean up quotes and formatting
    text = re.sub(r'^["\']\s*|\s*["\']$', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove numbered indices
    text = re.sub(r'\s+\d+\s*$', '', text)
    
    return text.strip()

def generate_description(book_title, tokenizer, model, device):
    """Generate description with improved prompt."""
    prompt = f"[INST] Generate a single sentence describing '{book_title}' and its genre. Do not include any introductory phrases or quotes. [/INST]"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        description = clean_description(text)
        
        # Verify the description is meaningful
        if len(description) < 20 or description.lower() in ['description unavailable.', '']:
            # Retry with alternative prompt
            retry_prompt = f"[INST] Describe the content and genre of '{book_title}' in one clear sentence. [/INST]"
            inputs = tokenizer(retry_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                retry_outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True
                )
            retry_text = tokenizer.decode(retry_outputs[0], skip_special_tokens=True)
            description = clean_description(retry_text)
        
        return description if len(description) >= 20 else "Description unavailable."
        
    except Exception as e:
        print(f"Error generating description for '{book_title}': {e}")
        return "Description unavailable."

def setup_tokenizer_and_model():
    """Initialize tokenizer and model with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer, model = setup_tokenizer_and_model()
    
    try:
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if 'description' not in df.columns:
                df['description'] = ""
            mask = (df['description'].isna()) | (df['description'] == "") | (df['description'] == "Description unavailable.")
            df = df[mask].copy()
        else:
            df = pd.read_csv(INPUT_CSV)
            if 'Book-Title' not in df.columns:
                raise ValueError("Missing 'Book-Title' column")
            df['description'] = ""
        
        df.reset_index(drop=True, inplace=True)
        
        total_rows = len(df)
        print(f"Processing {total_rows} rows")
        
        for idx in tqdm(range(total_rows), desc="Generating Descriptions"):
            title = df.at[idx, 'Book-Title']
            description = generate_description(title, tokenizer, model, device)
            df.at[idx, 'description'] = description
            
            if (idx + 1) % CHECKPOINT_INTERVAL == 0 or idx == total_rows - 1:
                print(f"\nSaving checkpoint at row {idx + 1}/{total_rows}")
                if os.path.exists(OUTPUT_CSV):
                    existing_df = pd.read_csv(OUTPUT_CSV)
                    existing_df.loc[df.index, 'description'] = df['description']
                    existing_df.to_csv(OUTPUT_CSV, index=False)
                else:
                    df.to_csv(OUTPUT_CSV, index=False)
        
        print("Processing completed")
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()