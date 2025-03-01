import tensorflow as tf
import pandas as pd
import numpy as np
import PIL
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_pnet

with open('Data/filtered_annotations.txt', 'r') as file:
    lines = file.readlines()

#Leave as Positives for testing for now
image_folder = '\Data\Positives'

def load_and_preprocess_image(image_path, target_size=(12, 12)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return image

images = []
face_classes = []
bboxes = []
landmarks = []
data = []
converted_bboxes = []
for idx, line in enumerate(lines):
    if idx> 50:
        break
    parts = line.strip().split()
    image_path = parts[0]
    
    # Load image and append
    try:
        image = load_and_preprocess_image(image_path)
        images.append(image)  # Append actual image data, not just paths
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue  # Skip corrupted images
    
    face_class = 1 if int(parts[1]) >= 0 else 0
    face_classes.append(face_class)
    
    # Process bbox
    try:
        bbox = tuple(map(float, filter(None, parts[2].strip('()').replace(" ", "").split(','))))
    except ValueError:
        bbox = ()  # Handle missing bbox gracefully
    bboxes.append(bbox)

    # Process landmarks if available
    if len(parts) > 3:
        try:
            landmark = tuple(map(float, filter(None, parts[3].strip('()').replace(" ", "").split(','))))
        except ValueError:
            landmark = ()
        landmarks.append(landmark)
    else:
        landmarks.append(())  # Ensure consistent array lengths

images = np.array(images, dtype=np.float32)
face_classes = np.array(face_classes, dtype=np.int32)
# Convert lists of tuples to numpy arrays, ensuring they are float32 and properly shaped
bboxes = np.array([list(b) if b else [0, 0, 0, 0] for b in bboxes], dtype=np.float32)
landmarks = np.array([list(l) if l else [0]*10 for l in landmarks], dtype=np.float32)

print("Made lists")

X_train, X_val, y_train_face, y_val_face, y_train_bbox, y_val_bbox, y_train_landmark, y_val_landmark = train_test_split(
    images, face_classes, bboxes, landmarks, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train_face, y_train_bbox, y_train_landmark)))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, (y_val_face, y_val_bbox, y_val_landmark)))

print("Made Datasets")

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Made Batch")

pnet_model = create_pnet()
pnet_model.compile(optimizer='adam',
                   loss={
                       'face_class': 'sparse_categorical_crossentropy',
                       'conv4-2': 'mean_squared_error',  # Match output layer names
                       'conv4-3': 'mean_squared_error'   # Match output layer names
                   },
                   metrics={
                       'face_class': 'accuracy',
                       'conv4-2': 'mse',
                       'conv4-3': 'mse'
                   })


print("Starting training arc!")

history = pnet_model.fit(train_dataset,
                         validation_data=val_dataset,
                         epochs=10,
                         verbose=1)
