import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Constants
INPUT_CSV = '../data/Books.csv'
OUTPUT_CSV = '../data/Books_with_descriptions.csv'
CHECKPOINT_INTERVAL = 100  # Save after every 100 rows

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-Guard-3-8B"

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map model to available devices
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True  # Optimize memory usage
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def generate_description_llama(book_title):
    prompt = (
        f"Provide a concise description for the book titled \"{book_title}\". "
        "Include the genre and a one-line summary of its content."
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,  # Enable sampling for more varied outputs
                top_p=0.9,       # Nucleus sampling
                top_k=50         # Top-K sampling
            )
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the generated part after the prompt
        description = description[len(prompt):].strip()
        return description
    except Exception as e:
        print(f"Error generating description for '{book_title}': {e}")
        return "Description unavailable."

def main():
    # Check if the output CSV exists to resume processing
    if os.path.exists(OUTPUT_CSV):
        print(f"Resuming from existing output file: '{OUTPUT_CSV}'")
        df_output = pd.read_csv(OUTPUT_CSV)
        # Ensure that the 'description' column exists
        if 'description' not in df_output.columns:
            df_output['description'] = ""
        # Identify rows that need processing
        mask = df_output['description'].isna() | (df_output['description'] == "")
        df = df_output[mask].copy()
        # Reset index for iteration
        df.reset_index(drop=True, inplace=True)
    else:
        print(f"Starting fresh with input file: '{INPUT_CSV}'")
        try:
            df = pd.read_csv(INPUT_CSV)
        except FileNotFoundError:
            print(f"Input CSV file '{INPUT_CSV}' not found.")
            exit(1)
        except Exception as e:
            print(f"Error reading '{INPUT_CSV}': {e}")
            exit(1)

        # Rename columns for consistency (optional)
        df.rename(columns=lambda x: x.strip(), inplace=True)  # Remove any leading/trailing whitespace

        # Check if 'Book-Title' column exists
        if 'Book-Title' not in df.columns:
            print("The CSV file does not contain a 'Book-Title' column.")
            exit(1)

        # Initialize 'description' column if it doesn't exist
        if 'description' not in df.columns:
            df['description'] = ""

    total_rows = df.shape[0]
    print(f"Total rows to process: {total_rows}")

    # Iterate over the DataFrame and generate descriptions
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Generating Descriptions"):
        title = row['Book-Title']
        if pd.isna(row['description']) or row['description'] == "":
            description = generate_description_llama(title)
            df.at[idx, 'description'] = description

        # Checkpointing
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            # Save the current progress
            print(f"\nCheckpoint reached at row {idx + 1}. Saving progress...")
            # If output CSV exists, append; else, create it
            if os.path.exists(OUTPUT_CSV):
                # Read the existing output CSV
                df_existing = pd.read_csv(OUTPUT_CSV)
                # Append the newly processed rows
                df_existing = df_existing.append(df.iloc[:idx + 1], ignore_index=True)
                # Save back to CSV
                df_existing.to_csv(OUTPUT_CSV, index=False)
            else:
                df.to_csv(OUTPUT_CSV, index=False)
            print(f"Progress saved to '{OUTPUT_CSV}'.\n")

    # Save the remaining rows after the loop completes
    print("Finalizing and saving all progress...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"All descriptions added and saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()
