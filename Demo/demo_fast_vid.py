import tensorflow as tf
import cv2
import numpy as np
import os
import sys
from time import time  # For performance measurement

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.PNetInputConverter import image_scaler, slide_window
from NMS import nms

# Load models
model = tf.keras.models.load_model('Model/Models/model_augmented.h5', compile=False)
model2 = tf.keras.models.load_model('Model/Models/rnet_model.h5', compile=False)

# === Parameters ===
VIDEO_DURATION_SECONDS = 10
FRAME_RATE = 30
OUTPUT_VIDEO_PATH = "output_detected_faces.avi"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# === Helpers ===
def load_and_preprocess_image_cv2(image, target_size=(12, 12), normalize=True):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if normalize:
        image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# === Record video ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print(f"Recording for {VIDEO_DURATION_SECONDS} seconds...")
frames = []
frame_count = 0
max_frames = FRAME_RATE * VIDEO_DURATION_SECONDS

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame.copy())
    cv2.imshow("Recording...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Captured {len(frames)} frames.")

# === New: Batch Processing Approach ===
start_time = time()

# Phase 1: Collect all crops from all frames with tracking information
all_crops = []
frame_indices = []  # Tracks which frame each crop belongs to
crop_info = []      # Stores original position and scale information

for frame_idx, frame in enumerate(frames):
    image_pyra = image_scaler(frame, scaleFactor=0.65)
    
    for im in image_pyra:
        h, w = im.image.shape[:2]
        if h > 200 or w > 200:
            continue
        crops = slide_window(im.image, im.current_scale, 4)
        
        for crop in crops:
            all_crops.append(crop.image)
            frame_indices.append(frame_idx)
            crop_info.append({
                'x': crop.x,
                'y': crop.y,
                'scale': crop.scale,
                'original_frame': frame
            })

# Convert all crops to batch format
if all_crops:
    batch_images = np.array([load_and_preprocess_image_cv2(crop) for crop in all_crops], dtype=np.float32)
    batch_images = np.squeeze(batch_images, axis=1)
    
    # Single P-Net call for all crops from all frames
    output = model.predict(batch_images)
    
    # Process P-Net outputs
    face_preds, bbox_preds, _ = output
    
    # Organize results by frame
    frame_results = {i: {'bboxes': [], 'confs': []} for i in range(len(frames))}
    
    for i in range(len(face_preds)):
        if face_preds[i][0] > 0.5:  # Face confidence threshold
            frame_idx = frame_indices[i]
            info = crop_info[i]
            bbox_pred = bbox_preds[i]
            
            scale = 1 / info['scale']
            x1 = int((info['x'] + bbox_pred[0]) * scale)
            y1 = int((info['y'] + bbox_pred[1]) * scale)
            w = int(bbox_pred[2] * scale)
            h = int(bbox_pred[3] * scale)
            
            frame_results[frame_idx]['bboxes'].append([x1, y1, x1 + w, y1 + h])
            frame_results[frame_idx]['confs'].append(face_preds[i][0])
    
    # Apply NMS per frame
    for frame_idx in frame_results:
        bboxes = frame_results[frame_idx]['bboxes']
        confs = frame_results[frame_idx]['confs']
        if bboxes:
            frame_results[frame_idx]['refined_boxes'] = nms.nms_regression(bboxes, confs, 0.4)
        else:
            frame_results[frame_idx]['refined_boxes'] = []

# Phase 2: R-Net processing in batch
# Collect all R-Net candidates across all frames
rnet_candidates = []
rnet_frame_indices = []
rnet_box_indices = []

for frame_idx in frame_results:
    refined_boxes = frame_results[frame_idx]['refined_boxes']
    original_frame = frames[frame_idx]
    
    for box_idx, box in enumerate(refined_boxes):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, original_frame.shape[1] - 1))
        y1 = max(0, min(y1, original_frame.shape[0] - 1))
        x2 = max(0, min(x2, original_frame.shape[1]))
        y2 = max(0, min(y2, original_frame.shape[0]))

        if x2 > x1 and y2 > y1:
            crop = original_frame[y1:y2, x1:x2]
            if crop.size > 0:
                processed = load_and_preprocess_image_cv2(crop, target_size=(24, 24))
                rnet_candidates.append(processed)
                rnet_frame_indices.append(frame_idx)
                rnet_box_indices.append(box_idx)

# Batch process all R-Net candidates
if rnet_candidates:
    rnet_batch = np.array(rnet_candidates, dtype=np.float32)
    rnet_batch = np.squeeze(rnet_batch, axis=1)
    rnet_output = model2.predict(rnet_batch)
    
    rnet_face_preds, _, _ = rnet_output
    
    # Process R-Net results
    for i in range(len(rnet_face_preds)):
        if rnet_face_preds[i][0] > 0.5:  # R-Net confidence threshold
            frame_idx = rnet_frame_indices[i]
            box_idx = rnet_box_indices[i]
            
            # Mark this box as confirmed by R-Net
            if 'rnet_confirmed' not in frame_results[frame_idx]:
                frame_results[frame_idx]['rnet_confirmed'] = []
            frame_results[frame_idx]['rnet_confirmed'].append(box_idx)

# Final NMS and frame processing
processed_frames = []

for frame_idx in range(len(frames)):
    frame = frames[frame_idx].copy()
    
    if 'rnet_confirmed' in frame_results[frame_idx]:
        # Get only R-Net confirmed boxes
        confirmed_boxes = [frame_results[frame_idx]['refined_boxes'][i] 
                          for i in frame_results[frame_idx]['rnet_confirmed']]
        confirmed_confs = [frame_results[frame_idx]['confs'][i] 
                          for i in frame_results[frame_idx]['rnet_confirmed']]
        
        # Final NMS
        final_boxes = nms.nms_regression(confirmed_boxes, confirmed_confs, 0.4)
        
        # Draw final boxes
        for box in final_boxes:
            x1, y1, x2, y2 = map(int, box)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    processed_frames.append(frame)

end_time = time()
print(f"Processing time: {end_time - start_time:.2f} seconds")

# === Save processed video ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

for f in processed_frames:
    out.write(f)

out.release()
print(f"Processed video saved to: {OUTPUT_VIDEO_PATH}")

# === Playback ===
for frame in processed_frames:
    cv2.imshow("Detected Faces Video", frame)
    if cv2.waitKey(int(1000 / FRAME_RATE)) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()