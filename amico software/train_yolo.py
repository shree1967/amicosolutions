import os
import json
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Concatenate, add
from keras.optimizers import Adam
import tensorflow as tf

# Function to build YOLO-like model
def create_yolo_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    def conv_block(x, filters, kernel_size, strides=1):
        if strides == 2:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(filters, kernel_size, strides=strides, padding='valid' if strides == 2 else 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x
    
    def residual_block(x, filters):
        shortcut = x
        x = conv_block(x, filters // 2, 1)
        x = conv_block(x, filters, 3)
        x = add([shortcut, x])
        return x
    
    def darknet53_body(x):
        x = conv_block(x, 32, 3)
        x = conv_block(x, 64, 3, strides=2)
        for _ in range(1):
            x = residual_block(x, 64)
        x = conv_block(x, 128, 3, strides=2)
        for _ in range(2):
            x = residual_block(x, 128)
        x = conv_block(x, 256, 3, strides=2)
        for _ in range(8):
            x = residual_block(x, 256)
        x = conv_block(x, 512, 3, strides=2)
        for _ in range(8):
            x = residual_block(x, 512)
        x = conv_block(x, 1024, 3, strides=2)
        for _ in range(4):
            x = residual_block(x, 1024)
        return x
    
    def yolo_head(x, num_filters):
        x = conv_block(x, num_filters, 1)
        x = conv_block(x, num_filters * 2, 3)
        x = conv_block(x, num_filters, 1)
        x = conv_block(x, num_filters * 2, 3)
        x = conv_block(x, num_filters, 1)
        return x
    
    x = darknet53_body(inputs)
    
    y1 = yolo_head(x, 512)
    y2 = UpSampling2D(2)(y1)
    y2 = Concatenate()([y2, x])
    y2 = yolo_head(y2, 256)
    
    y3 = UpSampling2D(2)(y2)
    y3 = Concatenate()([y3, x])
    y3 = yolo_head(y3, 128)
    
    outputs = [y1, y2, y3]
    
    model = Model(inputs, outputs)
    return model

# Custom loss function for YOLO
def yolo_loss(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))

# Function to load COCO dataset
def load_coco_data(images_path, annotations_path):
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    image_data = []
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        image_info = next(img for img in annotations['images'] if img['id'] == image_id)
        image_path = os.path.join(images_path, image_info['file_name'])
        
        image = cv2.imread(image_path)
        bbox = ann['bbox']  # [xmin, ymin, width, height]
        
        # Normalize bounding box
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        bbox_normalized = [xmin / image_info['width'], ymin / image_info['height'], xmax / image_info['width'], ymax / image_info['height']]
        
        image_data.append((image, bbox_normalized, ann['category_id']))
    
    return image_data

# Function to prepare data batches for training
def prepare_batches(data, batch_size, input_shape, num_classes):
    while True:
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            images = []
            bboxes = []
            for image, bbox, category_id in batch_data:
                image_resized = cv2.resize(image, input_shape[:2])
                bbox = np.array(bbox)
                bbox = np.expand_dims(bbox, axis=0)
                images.append(image_resized)
                bboxes.append(bbox)
            images = np.array(images) / 255.0
            bboxes = np.array(bboxes)
            yield images, bboxes

# Paths to COCO dataset
coco_images_path = 'data/coco_images/train2017/'
coco_annotations_path = 'data/coco_annotations/instances_train2017.json'

# Hyperparameters
batch_size = 16
input_shape = (416, 416, 3)
num_classes = 80  # COCO has 80 classes

# Load COCO dataset
data = load_coco_data(coco_images_path, coco_annotations_path)

# Build YOLO-like model
model = create_yolo_model(input_shape, num_classes)
model.summary()

# Compile model
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss=yolo_loss)

# Training loop
epochs = 50

train_generator = prepare_batches(data, batch_size, input_shape, num_classes)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(train_generator, steps_per_epoch=len(data) // batch_size, epochs=1)

# Save model
model.save('yolo_model_coco.h5')
