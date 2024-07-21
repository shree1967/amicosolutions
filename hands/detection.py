

import cv2
import numpy as np
from keras.models import load_model

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  
    return image


def detect_hand_state(model, image_path, input_shape):
    
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image, input_shape[:2])
    
    
    processed_image = np.expand_dims(processed_image, axis=0)
    

    prediction = model.predict(processed_image)
    
    
    if prediction > 0.5:
        return "Open hand"
    else:
        return "Closed hand"


if __name__ == "__main__":
    model_path = 'hand_detection_model.h5'  
    image_path = 'testimg1.jpg'  
    input_shape = (128, 128, 3)  
    
    
    model = load_model(model_path)
    
    
    result = detect_hand_state(model, image_path, input_shape)
    print(f"Detected: {result}")
