import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_pnet
import re

from collections import deque

import random
from collections import deque

#Leave as Positives for testing for now
image_folder = '\Data\Positives'

# def load_and_preprocess_image(image_path, target_size=(12, 12)):
#     image = load_img(image_path, target_size=target_size)
#     image = img_to_array(image) / 255.0  # Normalize to [0, 1]
#     reformated_image = np.expand_dims(reformated_image, axis=0)
#     return image

def load_and_preprocess_image_cv2(image_path, target_size=(12, 12), normalize=True):
    # Load image using OpenCV (BGR format by default)
    image = cv2.imread(image_path)
    
    # Resize the image to the target size (if necessary)
    image = cv2.resize(image, target_size)
    
    # Convert image to RGB (because OpenCV loads images as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize image to [0, 1] by dividing by 255 (only if required)
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image

def generate_batch(batch_size=64):
    images, face_classes, bboxes, landmarks = [], [], [], []

    # Check if there are enough lines in the queue to form a batch
    for _ in range(batch_size):
        if lines_queue:  # Make sure there are still lines in the queue
            line = lines_queue.popleft()  # Pop the first line from the queue
            #print("line: " + line)
            pattern = r'\([^)]*\)|\S+'
    
            # Find all parts that match the pattern (either inside parentheses or other space-separated parts)
            parts = re.findall(pattern, line)
            #print(parts)
            image_path = parts[0]

            try:
                # Preprocess the image
                image = load_and_preprocess_image_cv2(image_path)
                images.append(image)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue  # Skip corrupted images

            # Classify face presence
            face_class = 1 if int(parts[1]) >= 0 else 0
            #print(face_class)
            face_classes.append(face_class)

            try:
                # Extract bounding box info (x1, y1, width, height)
                if(face_class == 0):
                    bbox = (0, 0, 12, 12)
                    #print(bbox)
                else: 
                    bbox = tuple(map(float, parts[3].strip('()').replace(" ", "").split(',')))
                    #print(bbox)
            except (ValueError, IndexError):
                bbox = (0, 0, 0, 0)  # Default bbox if parsing fails
            bboxes.append(bbox)

            if len(parts) > 4:
                try:
                    # Extract landmark info (10 values)
                    landmark = tuple(map(float, parts[4].strip('()').replace(" ", "").split(',')))
                    print(landmark)
                    # Ensure landmarks have exactly 10 points
                    landmark = landmark + (0,) * (10 - len(landmark))  # Padding if less than 10 landmarks
                except ValueError:
                    landmark = (0,) * 10  # Default to 10 zeros in case of failure
                landmarks.append(landmark)
            else:
                landmarks.append((0,) * 10)  # Default landmarks if missing

    # Convert lists to numpy arrays, ensuring the correct data type
    images = np.array(images, dtype=np.float32)
    face_classes = np.array(face_classes, dtype=np.int32)
    bboxes = np.array([list(b) if b else [0, 0, 12, 12] for b in bboxes], dtype=np.float32)
    landmarks = np.array([list(l) if l else [0] * 10 for l in landmarks], dtype=np.float32)

    print(f"Created batch with {len(images)} samples.", flush=True)
    return images, face_classes, bboxes, landmarks


pnet_model = create_pnet()

loss_weights = {
    'face_class': 1.0,  # Prioritize classification
    'bbox_reg_reshaped': 0.5,  # Moderate importance
    'landmark_reg_reshaped': 0.5  # Moderate importance
}


# def custom_loss(face_class_weight=1.0, bbox_weight=0.5, landmark_weight=0.5):
#     """
#     Custom loss function that adjusts the contribution of the bounding box loss
#     based on the class of the sample (face_class).
#     """
    
#     def loss(y_true, y_pred):
#         # Extract true values from the tuple (not dictionary)
#         face_class_true = y_true[0]  # This should correspond to the face class
#         bbox_true = y_true[1]  # This should correspond to the bounding boxes
#         landmark_true = y_true[2]  # This should correspond to landmarks
        
#         # Extract predicted values from the model's predictions
#         face_class_pred = y_pred[0]
#         bbox_pred = y_pred[1]
#         landmark_pred = y_pred[2]
        
#         # Classification loss (binary crossentropy)
#         face_class_loss = K.binary_crossentropy(face_class_true, face_class_pred)
        
#         # Bounding box loss (mean squared error) - only if the face_class is positive (1)
#         bbox_loss = K.mean(K.square(bbox_true - bbox_pred), axis=-1)
#         # Mask the bbox loss for negative samples (face_class == 0)
#         bbox_loss = bbox_loss * face_class_true
        
#         # Landmark loss (mean squared error)
#         landmark_loss = K.mean(K.square(landmark_true - landmark_pred), axis=-1)
        
#         # Apply the weights
#         total_loss = (face_class_weight * face_class_loss +
#                       bbox_weight * bbox_loss +
#                       landmark_weight * landmark_loss)
        
#         return total_loss
    
#     return loss

# pnet_model.compile(optimizer='adam',
#                    loss=custom_loss(face_class_weight=1.0, bbox_weight=0.5, landmark_weight=0.5),
#                    metrics={'face_class': 'accuracy',
#                             'bbox_reg_reshaped': 'mse',
#                             'landmark_reg_reshaped': 'mse'})

def euclidean_loss(y_true, y_pred):
    """Compute Euclidean loss only for positive samples (face_class == 1)."""
    # Extract face classification labels (assuming the first value represents the label)
    face_labels = y_true[:, 0]  # Assumes y_true format: [face_class, bbox_targets]

    # Compute squared differences
    squared_diff = tf.square(y_true - y_pred)

    # Mask loss where face_labels == 1 (only positive face samples)
    mask = tf.cast(tf.equal(face_labels, 1), tf.float32)[:, tf.newaxis]  # Expand dims for broadcasting
    masked_loss = tf.reduce_sum(mask * squared_diff, axis=-1)

    return tf.reduce_mean(masked_loss)

pnet_model.compile(optimizer='adam',
                   loss={
                       'face_class': 'binary_crossentropy',
                       'bbox_reg_reshaped': euclidean_loss,
                       'landmark_reg_reshaped': 'mean_squared_error'
                   }, 
                   #Added params
                   #loss_weights=loss_weights,
                   #metrics=['accuracy'])
                   metrics={
                       'face_class': 'accuracy',
                       'bbox_reg_reshaped': 'mse',
                       'landmark_reg_reshaped': 'mse'
                   })

pnet_model.summary()

print("Starting training arc!", flush=True)

batch_size = 256
epochs = 1

def print_first_and_last(queue, label):
    queue_list = list(queue)
    print(f"\n{label}")
    print("First 10 elements:")
    print(queue_list[:10])
    print("\nLast 10 elements:")
    print(queue_list[-10:])

for epoch in range(epochs):
    # Load all lines from the file into a queue (deque for efficiency)
    lines_queue = deque()

    # Read the file and fill the queue
    with open('Data/filtered_annotations.txt', 'r') as file:
        lines_queue.extend(file.readlines())
        # lines = file.readlines()  # Read all lines
        # lines_queue.extend(lines[:200000])  # Take only the first x lines

    #print_first_and_last(lines_queue, "Before Shuffling")

    queue_list = list(lines_queue)

    # Step 2: Shuffle the list
    random.shuffle(queue_list)

    # Step 3: Convert back to deque
    lines_queue = deque(queue_list)

    #print_first_and_last(lines_queue, "After Shuffling")

    print(f"Epoch {epoch + 1}/{epochs}")
    #np.random.shuffle(lines_queue)  # Shuffle the dataset for randomness in training
    #print(f"Epoch {epoch + 1}/{epochs}")

    total_batches = len(lines_queue) // batch_size  # Calculate the total number of batches for the epoch
    batch_count = 0  # Initialize the batch counter
    
    # Continue until the queue is empty
    while lines_queue:  # Continue until the queue is empty
        images_batch, face_classes_batch, bboxes_batch, landmarks_batch = generate_batch(batch_size)
        
        # Now, you can use these batches in training
        loss_values = pnet_model.train_on_batch(images_batch, 
                                  {'face_class': face_classes_batch, 
                                   'bbox_reg_reshaped': bboxes_batch, 
                                   'landmark_reg_reshaped': landmarks_batch})
        batch_count += 1

        # Print progress for the current batch
        print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_count + 1}/{total_batches}")
        print(loss_values)
        if batch_count % 1000 == 0:
            print("Save model")
            pnet_model.save('Model/model.h5')

    print(f"Epoch {epoch + 1} completed.")
    

# Save the model

pnet_model.save('Model/model.h5')

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


