import tensorflow as tf
import pandas as pd
import numpy as np
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
for line in lines:
    parts = line.strip().split()
    image_path = parts[0]
    face_class = int(parts[1])
    bbox = tuple(map(int, parts[2].strip('()').split(',')))
    landmarks = None
    if len(parts) > 3:  # Check if landmarks are present
        landmarks = tuple(map(float, parts[3].strip('()').split(',')))
    data.append((image_path, face_class, bbox, landmarks))

images = np.array(images)
face_classes = np.array(face_classes)
bboxes = np.array(bboxes)
landmarks = np.array(landmarks)

X_train, X_val, y_train_face, y_val_face, y_train_bbox, y_val_bbox, y_train_landmark, y_val_landmark = train_test_split(
    images, face_classes, bboxes, landmarks, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train_face, y_train_bbox, y_train_landmark)))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, (y_val_face, y_val_bbox, y_val_landmark)))

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

pnet_model = create_pnet()
pnet_model.compile(optimizer='adam',
                   loss={
                       'face_class': 'sparse_categorical_crossentropy',
                       'bbox_reg': 'mean_squared_error',
                       'landmark_reg': 'mean_squared_error'
                   },
                   metrics={
                       'face_class': 'accuracy',
                       'bbox_reg': 'mse',
                       'landmark_reg': 'mse'
                   })

history = pnet_model.fit(train_dataset,
                         validation_data=val_dataset,
                         epochs=10,
                         verbose=1)