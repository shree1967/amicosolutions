import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical


def create_hand_detection_model(input_shape):
    model = Sequential()
    
    # Add Convolutional Layers with MaxPooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten and add Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 classes: Open hand, Closed hand, No hand movement
    
    return model


def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    return image


def train_hand_detection_model(X_train, y_train, input_shape, epochs=10, batch_size=16, model_save_path='hand_detection_model.h5'):
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=3)
    
    # Create and compile the model
    model = create_hand_detection_model(input_shape)
    model.summary()
    
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    # Example training data
    X_train = np.random.rand(100, 128, 128, 3)  
    y_train = np.random.randint(0, 3, size=(100,))  # Randomly generated labels (0: Open, 1: Closed, 2: No hand)
    
    input_shape = (128, 128, 3)  
    epochs = 10
    batch_size = 16
    model_save_path = 'hand_detection_model.h5' 
    
    # Train the model
    train_hand_detection_model(X_train, y_train, input_shape, epochs, batch_size, model_save_path)

