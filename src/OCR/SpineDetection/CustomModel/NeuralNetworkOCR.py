import cv2
import easyocr
from ultralytics import YOLO

def crop_books(image, results):
    """
    Crop detected book regions from the input image.
    Args:
        image: The original input image.
        results: YOLO detection results.
    Returns:
        List of cropped images corresponding to detected books.
    """
    crops = []
    for result in results:
        for box in result.boxes:  # Iterate over each detected box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            cropped = image[y1:y2, x1:x2]  # Crop the region
            crops.append(cropped)

    # Save crops for debugging
    for i, crop in enumerate(crops):
        print("Writing")
        cv2.imwrite(f"cropped/crop_{i}.jpg", crop)

    return crops

def extract_text(crops):
    """
    Extract text from cropped images using EasyOCR.
    Args:
        crops: List of cropped images.
    Returns:
        List of text strings extracted from the crops.
    """
    reader = easyocr.Reader(['en'])  # Initialize the OCR reader
    titles_and_authors = []
    for crop in crops:
        results = reader.readtext(crop)
        text = " ".join([result[1] for result in results])  # Combine detected text
        titles_and_authors.append(text.strip())
    return titles_and_authors

# Load the trained object detection model
model = YOLO('bookrec.pt')

# Load the image
image = cv2.imread('5BooksSameDirections.jpg')

# Detect books in the image
results = model.predict(image, conf=0.75)  # Adjust confidence threshold as needed

# Crop detected books
crops = crop_books(image, results)

# Extract text from the cropped book spines
texts = extract_text(crops)

# Print the extracted text
for i, text in enumerate(texts, start=1):
    print(f"Book {i}: {text}")
