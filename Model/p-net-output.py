import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import random
from model import create_pnet
from PNetInputConverter import image_scaler, slide_window
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NMS import nms
# Load the model
model = tf.keras.models.load_model('Model/Models/model_augmented.h5', compile=False)

# Load annotations
anno_file = "TrainingData/wider_face_annotations.txt"
new_anno_file = "RNetData/annotations.txt"

annotations = []
with open(anno_file, 'r') as f:
    lines = f.readlines()    

i = 0
while i < len(lines):
    path_name = lines[i].strip()
    i += 1
    num_faces = int(lines[i].strip())
    i += 1

    boxes = []
    for _ in range(num_faces):
        parts = list(map(int, lines[i].strip().split()))
        x1, y1, w, h, *_ = parts
        i += 1
        boxes.append([x1, y1, x1 + w, y1 + h])  # Convert to (x1, y1, x2, y2)

    if num_faces == 0:
        i+=1

    annotations.append((path_name, boxes))
    # add to annotation file
        # image file path
    

# Ensure output directories exist
os.makedirs("RNetData/Positives", exist_ok=True)
os.makedirs("RNetData/Negatives", exist_ok=True)

# IoU Calculation Function
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

im_count = 0
# Process each Wider-Face image
with open(new_anno_file, 'a') as anno_f:
    for image_path, gt_boxes in annotations:
        og_image_path = image_path
        image_path = "TrainingData/WIDER_train/images/" + image_path
        image_sample = cv2.imread(image_path)

        if image_sample is None:
            continue

        image_pyra = image_scaler(image_sample, scaleFactor=0.75)
        
        # Generate candidate windows
        crops = []
        for im in image_pyra:
            h, w = im.image.shape[:2]
            if h > 300 or w > 300:
                continue
            crops.extend(slide_window(im.image, im.current_scale, 4))

        # Run P-Net
        batch_images = np.array([cv2.resize(c.image, (12, 12)) / 255.0 for c in crops], dtype=np.float32)
        batch_images = np.expand_dims(batch_images, axis=-1)
        output = model.predict(batch_images)

        # Extract bounding boxes
        bboxes = []
        confidences = []

        for i in range(len(output[0])):
            face_pred, bbox_pred, _ = output
            face_pred = face_pred[i]
            bbox_pred = bbox_pred[i]
            crop = crops[i]

            if face_pred[0] > 0.5:
                inverse_scale = 1 / crop.scale
                x1 = int((crop.x + bbox_pred[0]) * inverse_scale)
                y1 = int((crop.y + bbox_pred[1]) * inverse_scale)
                x2 = x1 + int(bbox_pred[2] * inverse_scale)
                y2 = y1 + int(bbox_pred[3] * inverse_scale)

                bboxes.append([x1, y1, x2, y2])
                confidences.append(face_pred[0])

        # Apply NMS
        bboxes_refined = nms.nms_regression(bboxes, confidences, 0.4)

        # Compare with ground truth and save crops
        im_index = 0
        for box in bboxes_refined:
            x1, y1, x2, y2 = map(int, box)
            crop = image_sample[y1:y2, x1:x2]
            h, w, c = crop.shape

            if h < 24 or w < 24:
                continue

            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (24,24))
            # Compute IoU for each GT box
            max_iou, best_gt_box = max(
                ((calculate_iou(box, gt), gt) for gt in gt_boxes),
                key=lambda x: x[0],
                default=(0, None)  # If no ground truth boxes exist
            )
            

            if max_iou > 0.4:
                sanitized_path = og_image_path.replace("/", "_").replace("\\", "_")
                saved_path = f"RNetData/Positives/{sanitized_path}_{im_index}.jpg"
                cv2.imwrite(saved_path, crop)

                w = x2 - x1
                h = y2 - y1
                scaledW = 24/w
                scaledH = 24/h

                gt_w = best_gt_box[2] - best_gt_box[0]
                gt_h = best_gt_box[3] - best_gt_box[1]

                xoffset = (best_gt_box[0] - x1) * scaledW
                yoffset = (best_gt_box[1] - y1) * scaledH
                
                woffset = (gt_w) * scaledW
                hoffset = (gt_h) * scaledH

                # Append to annotation file
                anno_f.write(f"{saved_path} 1 ({[x1, y1, x2, y2]}) ({[xoffset, yoffset, woffset, hoffset]})\n")
            
            else:
                sanitized_path = og_image_path.replace("/", "_").replace("\\", "_")
                saved_path = f"RNetData/Negatives/{sanitized_path}_{im_index}.jpg"
                cv2.imwrite(saved_path, crop)

                # Append to annotation file
                anno_f.write(f"{saved_path} -1 ({[x1, y1, x2, y2]})\n")
            
            im_index += 1
            
        if im_count % 100 == 0:
            print(f"Image Index: {im_count}, {len(lines) - im_count} remaining")
        im_count += 1
        

print("Data collection for R-Net is complete.")
