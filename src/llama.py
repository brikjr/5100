import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
    input_csv = '../data/Books.csv'
    output_csv = '../data/Books_with_descriptions.csv'

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Input CSV file '{input_csv}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading '{input_csv}': {e}")
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

    # Iterate over the DataFrame and generate descriptions
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Descriptions"):
        title = row['Book-Title']
        if pd.isna(row['description']) or row['description'] == "":
            description = generate_description_llama(title)
            df.at[idx, 'description'] = description

    # Save the updated DataFrame to a new CSV
    try:
        df.to_csv(output_csv, index=False)
        print(f"Descriptions added and saved to '{output_csv}'.")
    except Exception as e:
        print(f"Error writing to '{output_csv}': {e}")

if __name__ == "__main__":
    main()
