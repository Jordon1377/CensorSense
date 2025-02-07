import os
import cv2
import numpy as np
import numpy.random as npr

import Utils.helpers
from Utils.annotation_class import Annotation
from Utils.boundingbox_class import BoundingBox

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
            boxes.append(BoundingBox(x1, y1, w, h, invalid, blur, expression, illumination, occlusion, pose))
            i += 1
    else:
        i += 1
        continue

    annotations.append(Annotation(path_name, num_faces, boxes))


image_directory = "TrainingData/WIDER_train/images/"
f_annotations = "Data/filtered_annotations.txt"
f_n = open(f_annotations, "a")

# Print the parsed annotations
index = 0
for anno in annotations:
    if index == 1:
        exit()
    image_name = anno.path_name
    num_faces = anno.num_faces
    bboxes = anno.boxes
    image_path = image_directory+image_name
    path = os.path.join(image_directory, image_name)
    print(path)
    if not os.path.exists(path):
        print(f"File not found: {path}")
    img = cv2.imread(path)

    # TrainingData/WIDER_train/images/0--Parade/0_Parade_marchingband_1_625.jpg
    height, width, channel = img.shape
    print(height, width, channel)

    negatives = 0
    positives = 0
    neg_save_dir = "Data/Negatives/"
    while negatives < 50:
        size = npr.randint(12, min(width, height) / 2)

        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)

        nb = BoundingBox(nx, ny, size, size, False)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        for box in bboxes:
            i = Utils.helpers.IoU(nb, box)
            if i < 0.3:
                save_file = os.path.join(neg_save_dir, f"{os.path.basename(image_name)}_{negatives}.jpg")
                negatives+=1
                cv2.imwrite(save_file, resized_im)
                f_n.write(f"{save_file} -1 {(ny, nx)}\n")  # Ensure newline for readability

    ### POSITIVE FILTERING

    for box in bboxes:
        if max(box.w, box.h) < 20: continue
        if box.x1 < 0 or box.x1 > width or box.y1 < 0 or box.y1 > height: continue

        n = 0
        while n < 20:
            neg_size = npr.randint(12, min(width, height) / 2)

            nx = npr.randint(min(neg_size, box.x1), width)
            ny = npr.randint(min(neg_size, box.y1), height)




    index=index+1


f_n.close()
