

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam


def create_hand_detection_model(input_shape):
    model = Sequential()
    
   
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    return model


def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    return image


def train_hand_detection_model(X_train, y_train, input_shape, epochs=10, batch_size=16, model_save_path='hand_detection_model.h5'):
    
    model = create_hand_detection_model(input_shape)
    model.summary()
    
    
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    
    
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    
    X_train = np.random.rand(100, 128, 128, 3)  
    y_train = np.random.randint(0, 2, size=(100,)) 
    
    input_shape = (128, 128, 3)  
    epochs = 10
    batch_size = 16
    model_save_path = 'hand_detection_model.h5' 
    
    
    train_hand_detection_model(X_train, y_train, input_shape, epochs, batch_size, model_save_path)
