#Converts an inputed image into a series of 12x12 frames representing the image as a whole.
#Uses different image scaling
import cv2
import os
import numpy as np
import random
import sys

#Function takes in an input image and creates various scaling pyramids of images ranging from as small as 12x12 to as large as the original size
#Play with the scaleFactor. 0.5 might be to small to get faces of all sizes. Test later with training set wider face.
def image_scaler(image, scaleFactor = 0.5):
    imagePyramid = []
    current_image = image.copy()

    while min(current_image.shape[:2]) >= 12:  # Stop when smallest dimension is 12 pixels
        imagePyramid.append(current_image)
        height, width = current_image.shape[:2]
        current_image = cv2.resize(current_image, (int(width * scaleFactor), int(height * scaleFactor)))

    #Image pyramid 
    #Currently added temp case to force 12x12 window at the end.
    # current_image = cv2.resize(current_image, (12, 12))
    # imagePyramid.append(current_image)
    return imagePyramid

def get_valid_image_path(default_path=r"TrainingData\WIDER_train\images\9--Press_Conference\9_Press_Conference_Press_Conference_9_16.jpg"):
    image_path = input(f"Enter image path (Press Enter for default path): ").strip()
    if not os.path.exists(image_path):
        print(f"File not found. Using default: {default_path}")
        image_path = default_path
    return image_path


def imagePyramidToFile():
    output_folder = "Model\TestingSpace"
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    image_path = get_valid_image_path()
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image.")
        return

    pyramid = image_scaler(image)

    for index, img in enumerate(pyramid):
        output_path = os.path.join(output_folder, f"scaled_{index}.png")
        cv2.imwrite(output_path, img)
        height, width = img.shape[:2]
        print(f"Saved: {output_path} | Size: {width}x{height}")

imagePyramidToFile()

