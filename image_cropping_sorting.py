import os
import cv2
import numpy as np
import numpy.random as npr

import Utils.helpers
from Utils.annotation_class import Annotation
from Utils.boundingbox_class import BoundingBox

anno_file = "TrainingData/wider_face_annotations.txt"

os.makedirs("Data", exist_ok=True)
os.makedirs("Data/Negatives", exist_ok=True)
os.makedirs("Data/Positives", exist_ok=True)
os.makedirs("Data/Parts", exist_ok=True)


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

neg_save_dir = "Data/Negatives/"
pos_save_dir = "Data/Positives/"
parts_save_dir = "Data/Parts/"

# Print the parsed annotations
index = 0
for anno in annotations:
    #if index == 1:
    #    exit()
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
    parts = 0
    depthLimit = 10000
    counter = 0
    while negatives < 30:
        counter+=1
        if(depthLimit < counter):
            break
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
                f_n.write(f"{save_file} -1 {(nx, ny, size)}\n")  # Ensure newline for readability

    ### POSITIVE FILTERING
    ### Part Filtering
    ### Close to face Negative Filtering

    for box in bboxes:
        if max(box.w, box.h) < 20: continue
        if box.x1 < 0 or box.x1 > width or box.y1 < 0 or box.y1 > height: continue

        n = 0
        positiveCounter = 0
        while positives < 20 or n < 5:
            positiveCounter+=1
            if(depthLimit < positiveCounter):
                break

            if n < 5:
                neg_size = npr.randint(12, int(2 * max(box.w, box.h))) #play with size here
                x_offset = npr.randint(-neg_size+1, box.w-1)
                y_offset = npr.randint(-neg_size+1, box.h-1)

                x1pos = box.x1 + x_offset
                x2pos = box.x1 + x_offset + neg_size
                y1pos = box.y1 + y_offset
                y2pos = box.y1 + y_offset + neg_size

                if x2pos > width or y2pos > height or y1pos < 0 or x1pos < 0:
                    continue
                nb = BoundingBox(x1pos, y1pos, neg_size, neg_size, False)

                cropped_im = img[y1pos: y2pos, x1pos: x2pos, :]
                if cropped_im is None or cropped_im.size == 0:
                    print(f"Error: Cropped image is empty at ({x1pos}, {y1pos}) to ({x2pos}, {y2pos}) + ({0}, {0}) to ({width}, {height})")
                    continue  # Skip this iteration
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                i = Utils.helpers.IoU(nb, box)
                if i < 0.3:
                    save_file = os.path.join(neg_save_dir, f"{os.path.basename(image_name)}_{negatives}.jpg")
                    n+=1
                    negatives+=1
                    cv2.imwrite(save_file, resized_im)
                    f_n.write(f"{save_file} -1 {(x1pos, y1pos, neg_size)}\n")  # Ensure newline for readability

            if positives < 20:
                if box.w < 5 or box.h < 5:
                    continue

                pos_size = max(12, npr.randint(int(min(box.w, box.h) * 0.8), int(1.15 * max(box.w, box.h)))) #smaller!!!

                delta_x = npr.randint(-box.w * 0.2, box.w * 0.2) #Play around with values
                delta_y = npr.randint(-box.h * 0.2, box.h * 0.2) #Play around with values

                nx1 = int(max(box.x1 + delta_x, 0)) #fix!
                ny1 = int(max(box.y1 + delta_y, 0))

                nx2 = nx1 + pos_size
                ny2 = ny1 + pos_size

                if nx2 > width or ny2 > height or nx1 < 0 or ny1 < 0:
                    continue 

                offset_x1 = (box.x1 - nx1) / (float(pos_size)/12)
                offset_y1 = (box.y1 - ny1) / (float(pos_size)/12)
                offset_x2 = (box.x1 + box.w - nx1) / (float(pos_size)/12)
                offset_y2 = (box.y1 + box.h - ny1) / (float(pos_size)/12)



                nb = BoundingBox(nx1, ny1, pos_size, pos_size, False)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                if cropped_im is None or cropped_im.size == 0:
                    print(f"Error: Cropped image is empty at ({nx1}, {ny1}) to ({nx2}, {ny2}) + ({0}, {0}) to ({width}, {height})")
                    continue  # Skip this iteration
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                i = Utils.helpers.IoU(nb, box)
                if i >= 0.65:
                    save_file = os.path.join(pos_save_dir, f"{os.path.basename(image_name)}_{positives}.jpg")
                    positives+=1
                    cv2.imwrite(save_file, resized_im)
                    f_n.write(f"{save_file} 1 {(nx1, ny1, pos_size)} {offset_x1, offset_y1, offset_x2-offset_x1, offset_y2-offset_y1}\n")  # Ensure newline for readability
                elif i >= 0.4:
                    save_file = os.path.join(parts_save_dir, f"{os.path.basename(image_name)}_{parts}.jpg") #Maybe do parts?
                    positives+=1
                    parts+=1
                    cv2.imwrite(save_file, resized_im)
                    f_n.write(f"{save_file} 0 {(nx1, ny1, pos_size)} {offset_x1, offset_y1, offset_x2-offset_x1, offset_y2-offset_y1}\n")  # Ensure newline for readability
    
    index=index+1


f_n.close()
