import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, label, target_size):
    images = []
    labels = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is not None:
            image = preprocess_image(image, target_size)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize the image
    return image

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
    model.add(Dense(2, activation='softmax'))  # 2 classes: Open hand, Closed hand
    return model

def train_hand_detection_model(X_train, y_train, input_shape, epochs=10, batch_size=16, model_save_path='hand_detection_model.h5'):
    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train, num_classes=2)
    
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
    folder_path_open = 'hands/train_open'  # Folder containing open hand images
    folder_path_closed = 'hands/train_closed'  # Folder containing closed hand images
    target_size = (128, 128)
    input_shape = (128, 128, 3)  # Image shape
    
    # Load images and labels from the dataset
    X_open, y_open = load_images_from_folder(folder_path_open, label=1, target_size=target_size)  # Label 1 for open hands
    X_closed, y_closed = load_images_from_folder(folder_path_closed, label=0, target_size=target_size)  # Label 0 for closed hands
    
    # Combine open and closed hand images and labels
    X_train = np.concatenate((X_open, X_closed), axis=0)
    y_train = np.concatenate((y_open, y_closed), axis=0)
    
    # Split data into training and validation sets (optional)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    epochs = 10
    batch_size = 16
    model_save_path = 'hand_detection_model.h5' 
    
    # Train the model
    train_hand_detection_model(X_train, y_train, input_shape, epochs, batch_size, model_save_path)


