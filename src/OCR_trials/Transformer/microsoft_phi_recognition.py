from transformers import AutoModelForCausalLM, AutoProcessor, modeling_utils
import torch
import os
from unittest.mock import patch
from PIL import Image
from transformers.dynamic_module_utils import get_imports

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model and processor
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)  

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