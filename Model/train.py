import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import PIL
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_pnet
import time


with open('Data/filtered_annotations.txt', 'r') as file:
    lines = file.readlines()

#Leave as Positives for testing for now
image_folder = '\Data\Positives'

def load_and_preprocess_image(image_path, target_size=(12, 12)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return image

images, face_classes, bboxes, landmarks = [], [], [], []

for idx, line in enumerate(lines):
    if idx > 25000:
        break
    parts = line.strip().split()
    image_path = parts[0]
    
    try:
        image = load_and_preprocess_image(image_path)
        images.append(image)
    except Exception as e:
        continue  # Skip corrupted images
    
    face_class = 1 if int(parts[1]) >= 0 else 0
    face_classes.append(face_class)
    
    try:
        bbox = tuple(map(float, parts[3].strip('()').replace(" ", "").split(',')))
    except (ValueError, IndexError):
        bbox = (0, 0, 0, 0)  # Default bbox if parsing fails
    bboxes.append(bbox)
    
    if len(parts) > 4:
        try:
            landmark = tuple(map(float, parts[4].strip('()').replace(" ", "").split(',')))
        except ValueError:
            landmark = (0,)*10
        landmarks.append(landmark)
    else:
        landmarks.append((0,)*10)

images = np.array(images, dtype=np.float32)
print(f"Total Training size: {len(images)}")
face_classes = np.array(face_classes, dtype=np.int32)
# Convert lists of tuples to numpy arrays, ensuring they are float32 and properly shaped
bboxes = np.array([list(b) if b else [0, 0, 0, 0] for b in bboxes], dtype=np.float32)
landmarks = np.array([list(l) if l else [0]*10 for l in landmarks], dtype=np.float32)

print("Made lists", flush=True)

X_train, X_val, y_train_face, y_val_face, y_train_bbox, y_val_bbox, y_train_landmark, y_val_landmark = train_test_split(
    images, face_classes, bboxes, landmarks, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train_face, y_train_bbox, y_train_landmark)))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, (y_val_face, y_val_bbox, y_val_landmark)))

print("Made Datasets", flush=True)

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Made Batch", flush=True)

pnet_model = create_pnet()

loss_weights = {
    'face_class': 1.0,  # Prioritize classification
    'bbox_reg_reshaped': 0.5,  # Moderate importance
    'landmark_reg_reshaped': 0.5  # Moderate importance
}

pnet_model.compile(optimizer='adam',
                   loss={
                       'face_class': 'binary_crossentropy',
                       'bbox_reg_reshaped': 'mean_squared_error',
                       'landmark_reg_reshaped': 'mean_squared_error'
                   }, 
                   #Added params
                #    loss_weights=loss_weights,
                #    metrics=['accuracy'])
                   metrics={
                       'face_class': 'accuracy',
                       'bbox_reg_reshaped': 'mse',
                       'landmark_reg_reshaped': 'mse'
                   })

pnet_model.summary()

time_start = time.time()
print("Starting training arc!", flush=True)

history = pnet_model.fit(train_dataset,
                         validation_data=val_dataset,
                         epochs=10,
                         verbose=2)

pnet_model.save('Model/model.h5')
end_time = time.time()
print(f"Duration of Training: {end_time-time_start}")

# Load a test image
def preprocess_test_image(image_path, target_size=(12, 12)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

test_image_path = 'Data/Parts/0_Parade_marchingband_1_5.jpg_6.jpg' 
test_image = preprocess_test_image(test_image_path)

# Make a prediction
face_pred, bbox_pred, landmark_pred = pnet_model.predict(test_image)

print("Face classification prediction:", face_pred)
print("Bounding box prediction:", bbox_pred)
print("Landmark prediction:", landmark_pred)


