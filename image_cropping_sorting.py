import os
import cv2
import numpy as np
import numpy.random as npr

from Utils.annotation_class import Annotation
from Utils.boundingbox_class import BoundingBox

anno_file = "TrainingData/wider_face_annotations.txt"
im_dir = "../../DATA/WIDER_train/images"
pos_save_dir = "../../DATA/12/positive"
part_save_dir = "../../DATA/12/part"
neg_save_dir = '../../DATA/12/negative'
save_dir = "../../DATA/12"

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
            boxes.append(BoundingBox(x1, y1, w, h, invalid, blur, expression, illumination, occlusion, pose))
            i += 1
    else:
        i += 1
        continue

    annotations.append(Annotation(path_name, num_faces, boxes))

# Print the parsed annotations
for anno in annotations[:5]:  # Print the first 5 entries as a sample
    print(anno)