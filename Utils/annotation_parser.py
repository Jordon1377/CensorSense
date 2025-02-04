import os
import cv2
import numpy as np
import numpy.random as npr

from Utils.annotation_class import Annotation
from Utils.boundingbox_class import BoundingBox

def GetAnnotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as f:
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
    return annotations