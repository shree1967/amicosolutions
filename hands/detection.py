import cv2
import numpy as np
from keras.models import load_model

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  
    return image


def detect_hand_state(model, image_path, input_shape):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image, input_shape[:2])
    
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Predict the class
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Map prediction to label
    if predicted_class == 0:
        return "Open hand"
    elif predicted_class == 1:
        return "Closed hand"
    else:
        return "Hand movement not detected"


if __name__ == "__main__":
    model_path = 'hand_detection_model.h5'  
    image_path = 'testimg1.jpg'  
    input_shape = (128, 128, 3)  
    
    # Load the trained model
    model = load_model(model_path)
    
    # Detect hand state
    result = detect_hand_state(model, image_path, input_shape)
    print(f"Detected: {result}")

