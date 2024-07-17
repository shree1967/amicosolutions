import cv2
import numpy as np
from keras.models import load_model

# Function to preprocess input image
def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize image
    return image

# Function to postprocess detected boxes
def postprocess_boxes(pred_boxes, img_shape):
    # Implement Non-Max Suppression (NMS) if necessary
    return pred_boxes

# Load the trained model
model_path = 'yolo_model_coco.h5'  # Path to your trained YOLO model
model = load_model(model_path, compile=False)

# Function to perform object detection on an image
def detect_objects(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image, (416, 416))
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    
    # Predict bounding boxes
    pred_boxes = model.predict(processed_image)
    
    # Postprocess boxes
    boxes = postprocess_boxes(pred_boxes, image.shape)
    
    # Draw bounding boxes on the image
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        xmin = int(xmin * image.shape[1])
        ymin = int(ymin * image.shape[0])
        xmax = int(xmax * image.shape[1])
        ymax = int(ymax * image.shape[0])
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    # Display the image with bounding boxes
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage: Detect objects in an image
image_path = 'image.jpeg'  # Replace with your image path
detect_objects(image_path)
