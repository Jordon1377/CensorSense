import tensorflow as tf
import pandas as pd
import numpy as np
import PIL
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow import keras  # Add this line
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from Model.PNetInputConverter import image_scaler, slide_window
import random
import cv2

import numpy as np
import numpy.random as npr
from NMS import nms


model = tf.keras.models.load_model('Model/model.h5', compile=False)
model2 = tf.keras.models.load_model('Model/rnet_model.h5', compile=False)

#print(model.summary())
cap = cv2.VideoCapture(0)
print("Press SPACE to capture an image.")
while True:
    ret, frame = cap.read()
    cv2.imshow("Live Feed - Press SPACE to Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        image_sample = frame.copy()
        break
cap.release()
cv2.destroyAllWindows()

cv2.imshow("Captured Image", image_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_pyra = image_scaler(image_sample, scaleFactor=0.75) 


crops = []
for im in image_pyra:
    h, w = im.image.shape[:2]
    if h > 300 or w > 300:
        continue
    crops.extend(slide_window(im.image, im.current_scale, 4))

predicted_Image = image_sample.copy()
totalImage = predicted_Image.copy()

def load_and_preprocess_image_cv2(image, target_size=(12, 12), normalize=True):
    # Resize the image to the target size (if necessary)
    image = cv2.resize(image, target_size)
    
    # Convert image to RGB (because OpenCV loads images as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize image to [0, 1] by dividing by 255 (only if required)
    if normalize:
        image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)  # shape will be (1, 12, 12, 3)
    
    return image

print("Crops total: " + str(len(crops)))
printed = False

bboxes = []
confidences = []

#Predict in batch is way faster!!!
batch_images = np.array([load_and_preprocess_image_cv2(c.image) for c in crops], dtype=np.float32)
batch_images = np.squeeze(batch_images, axis=1)  # Remove dimension at index 1
output = model.predict(batch_images)

for i in range(len(output[0])):

    face_pred, bbox_pred, landmark_pred = output
    face_pred = face_pred[i]
    bbox_pred = bbox_pred[i]
    landmark_pred = landmark_pred[i]
    crop = crops[i]

    # reformated_image = load_and_preprocess_image_cv2(c.image)
    # output = model.predict(reformated_image)
    # face_pred, bbox_pred, landmark_pred = output

    #If prediction is over x add box to predicted_image
    if face_pred[0] > 0.5:
        
        #print("Face prediction:", face_pred[0])
        #print("Bounding box prediction:", bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3])
        inverse_scale = 1 / crop.scale  # Inverse scale to expand the coordinates

        # Scale the bounding box coordinates and dimensions back to the original size
        xPos1 = int((crop.x + bbox_pred[0]) * inverse_scale)  # Multiply by inverse scale to get original size coordinates
        yPos1 = int((crop.y + bbox_pred[1]) * inverse_scale)  # Multiply by inverse scale
        w = int(bbox_pred[2] * inverse_scale)  # Multiply width by inverse scale
        h = int(bbox_pred[3] * inverse_scale)  # Multiply height by inverse scale

        totalImage = cv2.rectangle(totalImage, (xPos1, yPos1), (xPos1 + w, yPos1 + h), (0, 255, 0), 1)

        bboxes.append([xPos1, yPos1, xPos1+w, yPos1+h])
        confidences.append(face_pred[0])
        
    
cv2.imshow("Image with Box", totalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

bboxes_refined = nms.nms_regression(bboxes, confidences, 0.4)
predicted_Image = image_sample.copy()

for box in bboxes_refined:
    xPos1, yPos1, xPos2, yPos2 = map(int, box)
    predicted_Image = cv2.rectangle(predicted_Image, (xPos1, yPos1), (xPos2, yPos2), (0, 255, 0), 1)
cv2.imshow("Image with Box", predicted_Image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#RNET Computation
predicted_Image2 = image_sample.copy()

bboxes = []
confidences = []
bboxSizeVals = []

#Predict in batch is way faster!!!
bboxSizeVals = np.array([(box[0],box[1], box[2], box[3]) for box in bboxes_refined])

h, w = image_sample.shape[:2]
valid_crops = []

for box in bboxes_refined:
    x1, y1, x2, y2 = map(int, box)

    # Clamp coordinates within image bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    # Ensure non-zero area
    if x2 <= x1 or y2 <= y1:
        continue

    crop = image_sample[y1:y2, x1:x2]

    if crop.size == 0:
        continue  # still empty somehow

    processed = load_and_preprocess_image_cv2(crop, target_size=(24, 24))
    valid_crops.append(processed)

batch_images = np.array(valid_crops, dtype=np.float32)

batch_images = np.squeeze(batch_images, axis=1)  # Remove dimension at index 1
output = model2.predict(batch_images)

print(len(output[0]))

for i in range(len(output[0])):

    face_pred, bbox_pred, landmark_pred = output
    face_pred = face_pred[i]
    bbox_pred = bbox_pred[i]
    landmark_pred = landmark_pred[i]
    box = bboxSizeVals[i]

    if face_pred[0] > 0.5:

        bboxes.append([box[0], box[1], box[2], box[3]])
        confidences.append(face_pred[0])


for box in bboxes:
    x1, y1, x2, y2 = map(int, box)  # Make sure coordinates are integers
    predicted_Image2 = cv2.rectangle(predicted_Image2, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv2.imshow("Image with Box", predicted_Image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

bboxes_refined = nms.nms_regression(bboxes, confidences, 0.4)
predicted_Image = image_sample.copy()

for box in bboxes_refined:
    xPos1, yPos1, xPos2, yPos2 = map(int, box)
    predicted_Image = cv2.rectangle(predicted_Image, (xPos1, yPos1), (xPos2, yPos2), (0, 255, 0), 1)
cv2.imshow("Image with Box", predicted_Image)
cv2.waitKey(0)
cv2.destroyAllWindows()


