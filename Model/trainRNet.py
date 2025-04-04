import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_rnet
import re

from collections import deque

import random
from collections import deque

#Leave as Positives for testing for now
image_folder = 'Model\RNetData\Positives'
def random_brightness(image, max_delta=0.2):
    """Randomly adjust the brightness of the image."""
    delta = random.uniform(-max_delta, max_delta)
    image = np.clip(image + delta, 0, 1)
    return image

def random_contrast(image, lower=0.5, upper=1.5):
    """Randomly adjust the contrast of the image."""
    factor = random.uniform(lower, upper)
    image = np.clip((image - 0.5) * factor + 0.5, 0, 1)
    return image

def random_gamma(image, lower=0.5, upper=1.5):
    """Randomly adjust the gamma of the image."""
    gamma = random.uniform(lower, upper)
    image = np.clip(image ** gamma, 0, 1)
    return image

def random_grayscale(image, probability=0.1):
    """Randomly convert image to grayscale with a specified probability."""
    if random.random() < probability:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
    return image

def random_augmentation(image, untouched_prob=0.4, grayscale_prob=0.1):
    """Apply random augmentations to the image."""
    if random.random() < untouched_prob:
        # No augmentation for 30-40% of the images
        return image
    
    # Apply other augmentations
    if random.random() < 0.5:
        image = random_brightness(image)
    if random.random() < 0.5:
        image = random_contrast(image)
    if random.random() < 0.5:
        image = random_gamma(image)
    
    # Grayscale has a fixed probability of being applied
    image = random_grayscale(image, probability=grayscale_prob)
    
    return image

def load_and_preprocess_image_cv2(image_path, target_size=(24, 24), normalize=True):
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
                image = random_augmentation(image)
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
                    bbox = (0, 0, 24, 24)
                    #print(bbox)
                else: 
                    bbox = tuple(map(float, parts[3].strip('()').strip('[]').replace(" ", "").split(',')))
                    # print("Box: ")
                    # print(bbox)
            except (ValueError, IndexError):
                bbox = (0, 0, 0, 0)  # Default bbox if parsing fails
            bboxes.append(bbox)
            landmarks.append((0,) * 10)  # Default landmarks if missing

    # Convert lists to numpy arrays, ensuring the correct data type
    images = np.array(images, dtype=np.float32)
    face_classes = np.array(face_classes, dtype=np.int32)
    bboxes = np.array([list(b) if b else [0, 0, 24, 24] for b in bboxes], dtype=np.float32)
    landmarks = np.array([list(l) if l else [0] * 10 for l in landmarks], dtype=np.float32)

    print(f"Created batch with {len(images)} samples.", flush=True)
    return images, face_classes, bboxes, landmarks


rnet_model = create_rnet()

loss_weights = {
    'face_class': 1.0,  # Prioritize classification
    'bbox_reg': 1.0,  # Moderate importance
    'landmark_reg': 0.5  # Moderate importance
}


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

# ✅ Dynamic learning rate scheduling
initial_lr = 0.0001

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Optimizer with weight decay (AdamW)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=initial_lr,
    weight_decay=3e-5  # Regularization to prevent overfitting
)

rnet_model.compile(optimizer=optimizer,
                   loss={
                       'face_class': 'binary_crossentropy',
                       'bbox_reg': euclidean_loss,
                       'landmark_reg': 'mean_squared_error'
                   }, 
                   #Added params
                   #loss_weights=loss_weights,
                   #metrics=['accuracy'])
                   metrics={
                       'face_class': 'accuracy',
                       'bbox_reg': 'mse',
                       'landmark_reg': 'mse'
                   })

rnet_model.summary()

print("Starting training arc!", flush=True)

batch_size = 184
epochs = 2

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
    with open('RNetData/annotations.txt', 'r') as file:
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
        loss_values = rnet_model.train_on_batch(images_batch, 
                                  {'face_class': face_classes_batch, 
                                   'bbox_reg': bboxes_batch, 
                                   'landmark_reg': landmarks_batch})
        batch_count += 1

        # Print progress for the current batch
        print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_count + 1}/{total_batches}")
        print(loss_values)

    print(f"Epoch {epoch + 1} completed.")
    
    print("✅ Saving model checkpoint...")
    rnet_model.save(f'Model/RNetAugmentedModels/rnet_epoch_{epoch}.h5')
    batch_size += 20
    

# Save the model

rnet_model.save('Model/RNetAugmentedModels/rnet_model.h5')
print("Model Saved")


