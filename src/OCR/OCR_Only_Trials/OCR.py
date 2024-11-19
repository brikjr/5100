import cv2
import numpy as np
import easyocr

def preprocess_image(image_path):
    """Preprocess the image to emphasize spine divisions."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image at {image_path}")
    
    # Step 1: Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Step 2: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)  # Use only the lightness channel for enhancement

    # Step 3: Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced_l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Step 4: Morphological operations to connect divisions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 5: Detect edges
    edges = cv2.Canny(closed, 50, 150)

    cv2.imshow(f"IMG", edges)
    cv2.waitKey(0)

    return image, edges

def detect_spines(image, edges):
    """Detect book spines using contour detection."""
    # Find contours from the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spines = []
    min_width, min_height = 50, 100  # Minimum size for valid spines
    max_width = image.shape[1] // 3  # Avoid overly large regions
    
    for contour in contours:
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours by size and aspect ratio
        if w >= min_width and h >= min_height and w < max_width and h / w > 2:
            spine = image[y:y+h, x:x+w]
            spines.append(spine)
    
    return spines

def extract_text_from_spines(spines):
    """Use OCR to extract text from each detected spine."""
    reader = easyocr.Reader(['en'])
    book_data = []

    for idx, spine in enumerate(spines):
        gray_spine = cv2.cvtColor(spine, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray_spine)

        combined_text = " ".join([text for (_, text, confidence) in results if confidence > 0.5])
        if ' by ' in combined_text.lower():
            title, author = combined_text.split(' by ', maxsplit=1)
            book_data.append({'Title': title.strip(), 'Author': author.strip()})
        else:
            book_data.append({'Title': combined_text.strip(), 'Author': 'Unknown'})

    return book_data

def process_bookshelf(image_path):
    """Process an image of a bookshelf to extract book titles and authors."""
    # Step 1: Preprocess the image
    image, edges = preprocess_image(image_path)

    # Step 2: Detect spines using contours
    spines = detect_spines(image, edges)

    # Step 3: Extract text from spines using OCR
    book_data = extract_text_from_spines(spines)

    return book_data

# Example usage
image_path = "5BooksSameDirections.jpg"  # Path to your image
book_data = process_bookshelf(image_path)

# Display the results
for book in book_data:
    print(f"Title: {book['Title']}, Author: {book['Author']}")

def visualize_spines(image, spines):
    for idx, spine in enumerate(spines):
        cv2.imshow(f"Spine {idx}", spine)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize detected spines
image, edges = preprocess_image(image_path)
spines = detect_spines(image, edges)
#visualize_spines(image, spines)
