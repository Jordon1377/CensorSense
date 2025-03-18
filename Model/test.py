import tensorflow as tf
import pandas as pd
import numpy as np
import PIL
import os
import sys
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras  # Add this line
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_pnet
from PNetInputConverter import image_scaler, slide_window
import random
import cv2

import numpy as np
import numpy.random as npr

anno_file = "TrainingData/wider_face_annotations.txt"

annotations = []
with open(anno_file, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    path_name = lines[i].strip()  # First line is the image path
    i += 1
    num_faces = int(lines[i].strip())  # Second line is the number of faces
    i += 1

    boxes = []
    if num_faces > 0:  # Only parse boxes if there are faces in the image
        for _ in range(num_faces):
            parts = list(map(int, lines[i].strip().split()))
            x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = parts
            i += 1
    else:
        i += 1
        continue
    annotations.append(path_name)

model = tf.keras.models.load_model('Model/model.h5', compile=False)

#print(model.summary())

anno_sample = random.choice(annotations)
anno_sample = "TrainingData/WIDER_train/images/"+anno_sample
image_sample = cv2.imread(anno_sample)
cv2.imshow("Original Image", image_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_pyra = image_scaler(image_sample, scaleFactor=0.5) 


crops = []
for im in image_pyra:
    h, w = im.image.shape[:2]
    if h > 75 or w > 75:
        continue
    crops.extend(slide_window(im.image, im.current_scale, 4))

predicted_Image = image_sample.copy()

print("Crops total: " + str(len(crops)))
printed = False
for c in crops:
    reformated_image = img_to_array(c.image) / 255.0  # Normalize
    reformated_image = np.expand_dims(reformated_image, axis=0)
    output = model.predict(reformated_image)
    face_pred, bbox_pred, landmark_pred = output
    if not printed:
        print("Face classification prediction:", face_pred[0][0])
        print("Bounding box prediction:", bbox_pred[0][0], bbox_pred[0][1], bbox_pred[0][2], bbox_pred[0][3])
        #print("Landmark prediction:", landmark_pred)
        #printed = True

    #If prediction is over x add box to predicted_image
    if face_pred[0][0] > 0.4:
        # xPos1 = int((c.x + bbox_pred[0][0]) / c.scale)
        # yPos1 = int((c.y + bbox_pred[0][1]) / c.scale)
        # w = int((bbox_pred[0][2]*12) / c.scale)
        # h = int((bbox_pred[0][3]*12) / c.scale)

        xPos1 = int((c.x) / c.scale)
        yPos1 = int((c.y) / c.scale)
        w = int(12 / c.scale)
        h = int(12 / c.scale)
        

        predicted_Image = cv2.rectangle(predicted_Image, (xPos1, yPos1), (xPos1 + w, yPos1 + h), (0, 255, 0), 1)
    
cv2.imshow("Image with Box", predicted_Image)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
    


