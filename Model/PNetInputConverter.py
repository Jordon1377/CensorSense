#Converts an inputed image into a series of 12x12 frames representing the image as a whole.
#Uses different image scaling
import cv2
import os
import numpy as np
import random
import sys

#Scaled image class keeps track of the scaling of images so we can go from one scale to the original scale when computing bounding boxes
class ScaledImage:
    def __init__(self, image, original_scale, current_scale):
        self.image = image
        self.original_scale = original_scale
        self.current_scale = current_scale

#This Class stores a 12x12 crop and it's locations in a relative image. A scale factor is included so that the crop could be scaled up to the same size as the reference image.
class ImageCrop:
    def __init__(self, image, scale, x, y):
        self.image = image
        self.scale = scale
        self.x = x
        self.y = y

# Function slides a 12x12 window across an image with a given step size
def slide_window(image, scale, step=3):
    crops = []
    height, width = image.shape[:2]
    for x in range(0, height - 12 + 1, step):
        for y in range(0, width - 12 + 1, step):
            crop = image[x:x+12, y:y+12]
            crops.append(ImageCrop(crop, scale, x, y))
    return crops

#Function takes in an input image and creates various scaling pyramids of images ranging from as small as 12x12 to as large as the original size
#Play with the scaleFactor. 0.5 might be to small to get faces of all sizes. Test later with training set wider face.
def image_scaler(image, scaleFactor = 0.5):
    imagePyramid = []
    current_image = image.copy()
    current_scale = 1.0

    while min(current_image.shape[:2]) >= 12:  # Stop when smallest dimension is 12 pixels
        imagePyramid.append(ScaledImage(current_image, 1.0, current_scale))
        height, width = current_image.shape[:2]
        current_scale *= scaleFactor
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

def save_images_to_folder(images, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for index, img in enumerate(images):
        output_path = os.path.join(folder, f"{prefix}_{index}.png")
        cv2.imwrite(output_path, img.image)
        print(f"Saved: {output_path} | Scale: {img.scale if hasattr(img, 'scale') else img.current_scale}")



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
        cv2.imwrite(output_path, img.image)
        height, width = img.image.shape[:2]
        print(f"Saved: {output_path} | Size: {width}x{height} | Scale: {img.current_scale}")

    process_image_and_save(pyramid)

def process_image_and_save(pyramid, maxheight=300, maxwidth=300):
    output_folder_crops = "Model\TestingSpace\TestingCrops"

    # Generate and save sliding window crops
    all_crops = []
    for image in pyramid:
        height, width = image.image.shape[:2]
        if height > maxheight or width > maxwidth:
            continue
        all_crops.extend(slide_window(image.image, 1.0, step=4))
    save_images_to_folder(all_crops, output_folder_crops, "crop")
    
    print(f"Total crops saved: {len(all_crops)}")

imagePyramidToFile()

