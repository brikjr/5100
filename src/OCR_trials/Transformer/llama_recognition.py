from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# Load model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

processor = AutoProcessor.from_pretrained(model_id)

# Load image and preprocess
image_path = "5BooksSameDirections.jpg"
image = Image.open(image_path).convert("RGB")

# Tie weights if necessary
model.tie_weights()

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Provide the title and the author of each book in this image"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device).to("cuda" if torch.cuda.is_available() else "cpu")

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))