import tensorflow as tf
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_pnet

# Load annotation file
with open('Data/filtered_annotations.txt', 'r') as file:
    lines = file.readlines()

# Set image directory
image_folder = os.path.join('Data', 'Positives')

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(12, 12)):
    try:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) / 255.0  # Normalize to [0,1]
        return image
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# Initialize lists
images, face_classes, bboxes, landmarks = [], [], [], []

# Read annotations and load images
for idx, line in enumerate(lines):
    if idx > 25000:  # Limit dataset size
        break

    parts = line.strip().split()
    image_path = parts[0]

    # Load image
    image = load_and_preprocess_image(image_path)
    if image is None:
        continue  # Skip corrupted images
    images.append(image)

    # Parse face classification
    face_class = 1 if int(parts[1]) >= 0 else 0
    face_classes.append(face_class)

    # Parse bounding box (fallback to (0,0,0,0) if invalid)
    try:
        bbox = tuple(map(float, parts[2].strip('()').split(',')))
    except (ValueError, IndexError):
        bbox = (0, 0, 0, 0)
    bboxes.append(bbox)

    # Parse landmarks (fallback to (0,0,...,0) if invalid)
    if len(parts) > 3:
        try:
            landmark = tuple(map(float, parts[3].strip('()').split(',')))
        except ValueError:
            landmark = (0,) * 10
        landmarks.append(landmark)
    else:
        landmarks.append((0,) * 10)

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
face_classes = np.array(face_classes, dtype=np.int32)
bboxes = np.array([list(b) for b in bboxes], dtype=np.float32)
landmarks = np.array([list(l) for l in landmarks], dtype=np.float32)

print(f"Total Training Samples: {len(images)}")

# Split dataset
X_train, X_val, y_train_face, y_val_face, y_train_bbox, y_val_bbox, y_train_landmark, y_val_landmark = train_test_split(
    images, face_classes, bboxes, landmarks, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train_face, y_train_bbox, y_train_landmark)))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, (y_val_face, y_val_bbox, y_val_landmark)))

# Batch, shuffle, and prefetch for performance
batch_size = 32
train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

print("Dataset prepared.")

# Load model
pnet_model = create_pnet()

# Loss weights for multi-task learning
loss_weights = {
    'face_class': 1.0,
    'bbox_reg_reshaped': 0.5,
    'landmark_reg_reshaped': 0.5
}

# Compile model
pnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'face_class': 'binary_crossentropy',
        'bbox_reg_reshaped': 'mean_squared_error',
        'landmark_reg_reshaped': 'mean_squared_error'
    },
    loss_weights=loss_weights,
    metrics={
        'face_class': 'accuracy',
        'bbox_reg_reshaped': 'mse',
        'landmark_reg_reshaped': 'mse'
    }
)

pnet_model.summary()

# Training loop with model checkpointing
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='Model/best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1
)

time_start = time.time()
print("Starting training...")

history = pnet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[checkpoint_callback],
    verbose=2
)

# Save final model
pnet_model.save('Model/final_model.h5')

print(f"Training completed in {time.time() - time_start:.2f} seconds.")

# Function to preprocess a test image
def preprocess_test_image(image_path, target_size=(12, 12)):
    try:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) / 255.0
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing test image {image_path}: {e}")
        return None

# Load and predict on a test image
test_image_path = 'Data/Parts/0_Parade_marchingband_1_5.jpg_6.jpg'
test_image = preprocess_test_image(test_image_path)

if test_image is not None:
    face_pred, bbox_pred, landmark_pred = pnet_model.predict(test_image)
    print(f"Face Classification: {face_pred}")
    print(f"Bounding Box: {bbox_pred}")
    print(f"Landmarks: {landmark_pred}")
