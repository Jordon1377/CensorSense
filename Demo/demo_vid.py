import tensorflow as tf
import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Model.PNetInputConverter import image_scaler, slide_window
from NMS import nms


# Load models
model = tf.keras.models.load_model('Model/Models/model_augmented.h5', compile=False)
model2 = tf.keras.models.load_model('Model/Models/rnet_model.h5', compile=False)

# === Parameters ===
VIDEO_DURATION_SECONDS = 5
FRAME_RATE = 10
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

# === Face detection and draw on each frame ===
processed_frames = []

for frame in frames:
    image_pyra = image_scaler(frame, scaleFactor=0.75)
    crops = []

    for im in image_pyra:
        h, w = im.image.shape[:2]
        if h > 300 or w > 300:
            continue
        crops.extend(slide_window(im.image, im.current_scale, 4))

    batch_images = np.array([load_and_preprocess_image_cv2(c.image) for c in crops], dtype=np.float32)
    batch_images = np.squeeze(batch_images, axis=1)
    output = model.predict(batch_images)

    bboxes = []
    confidences = []

    for i in range(len(output[0])):
        face_pred, bbox_pred, _ = output
        face_pred = face_pred[i]
        bbox_pred = bbox_pred[i]
        crop = crops[i]

        if face_pred[0] > 0.5:
            scale = 1 / crop.scale
            x1 = int((crop.x + bbox_pred[0]) * scale)
            y1 = int((crop.y + bbox_pred[1]) * scale)
            w = int(bbox_pred[2] * scale)
            h = int(bbox_pred[3] * scale)
            bboxes.append([x1, y1, x1 + w, y1 + h])
            confidences.append(face_pred[0])

    bboxes_refined = nms.nms_regression(bboxes, confidences, 0.4)

    # === R-Net processing ===
    final_boxes = []
    final_confs = []
    valid_crops = []

    for box in bboxes_refined:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(0, min(x2, frame.shape[1]))
        y2 = max(0, min(y2, frame.shape[0]))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        processed = load_and_preprocess_image_cv2(crop, target_size=(24, 24))
        valid_crops.append(processed)

    if valid_crops:
        batch_images = np.array(valid_crops, dtype=np.float32)
        batch_images = np.squeeze(batch_images, axis=1)
        output = model2.predict(batch_images)

        for i in range(len(output[0])):
            face_pred, _, _ = output
            face_pred = face_pred[i]
            if face_pred[0] > 0.5:
                final_boxes.append(bboxes_refined[i])
                final_confs.append(face_pred[0])

    final_boxes = nms.nms_regression(final_boxes, final_confs, 0.4)

    # === Draw boxes ===
    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    processed_frames.append(frame)

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
