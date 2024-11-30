import ollama
import base64
from PIL import Image
import io

def encode_image_to_base64(image_bytes):
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def generate_resopnse(image_file):
    """Generate response from Ollama model using uploaded file."""
    try:
        # Read image bytes from StreamlitUploadedFile
        image_bytes = image_file.getvalue()
        
        # Convert to base64
        base64_image = encode_image_to_base64(image_bytes)
        
        response = ollama.chat(model='llama2-vision', messages=[
            {
                'role': 'user',
                'content': 'List the titles of all the books in the image. List the book titles in quotes. Do not include any other text in the response, only the book titles in quotes.',
                'images': [base64_image]
            },
        ])
        response = response['message']['content']
        return response
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        raise

def titles_to_list(response):
    """Convert response string to list of titles."""
    # Setup a flag to account for if a quotation has been seen
    quote_seen = 0
    # Setup an empty string for the title and an empty list for the return value
    current_title = ""
    title_list = []
    # For each character in the response string
    for char in response:
        # If no starting quote has been seen yet and the current char is a quote
        if quote_seen == 0 and char == '"':
            # Set the quote_seen to true to start recording
            quote_seen = 1
        # If the starting quote has been seen and the current char is a quote
        elif quote_seen == 1 and char == '"':
            # Set the quote seen to false, add the title to the list and reset the string
            quote_seen = 0
            title_list.append(current_title)
            current_title = ""
        # If the starting quote is seen then append the current char to the title string
        elif quote_seen == 1:
            current_title += char
    # Remove duplicates using a set
    unique_list = list(set(title_list))
    return unique_list

def processImageGetList(image_file):
    """Process image file and return list of book titles."""
    try:
        if image_file is None:
            return []
            
        response = generate_resopnse(image_file)
        return titles_to_list(response)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []