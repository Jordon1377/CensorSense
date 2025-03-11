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

model = keras.models.load_model('Model/model.h5')

print(model.summary())

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
    if h > 100 or w > 100:
        continue
    crops.extend(slide_window(im.image, im.current_scale, 4))

printed = False
for c in crops:
    c = np.expand_dims(c.image, axis=0)
    output = model.predict(c)
    if not printed:
        print(output)
        printed = True

