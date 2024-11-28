from ultralytics import YOLO
import cv2

def visualize_predictions(image_path, model_path, output_path):
    """
    Visualize bounding boxes on the provided image using a trained YOLO model.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained YOLO model (e.g., 'best.pt').
        output_path (str): Path to save the output image with bounding boxes.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)
    
    # Load the input image
    image = cv2.imread(image_path)
    
    # Make predictions
    results = model.predict(image, conf=0.8)  # Adjust confidence threshold as needed

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            conf = box.conf[0]  # Confidence score
            label = f"Book: {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

# Example usage
image_path = "5BooksSameDirections.jpg"          # Path to your input image
model_path = "bookrec.pt"                # Path to your trained model
output_path = "output_bookshelf.jpg"  # Path to save the output image
visualize_predictions(image_path, model_path, output_path)
