import cv2
import os
import numpy as np
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add the project root to sys.path


# Function that Draws bounding boxes on a copy of an image and returns it. Uses user input for bounding box coordinates.
def DrawBBoxManual(image_path: str):
    image = cv2.imread(image_path)  # Load image

    # Error handling
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image from {image_path}")

    # Get bounding box coordinates from user
    x = int(input("Enter x coordinate: ").strip())
    y = int(input("Enter y coordinate: ").strip())
    w = int(input("Enter width: ").strip())
    h = int(input("Enter height: ").strip())

    # Apply rectangle to image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
    return image


# Function that outputs image to a folder
def SaveImageToFolder(image, folder_path: str = "DataVisualization/BoundingBoxVisualization",
                      filename: str = "image.jpg"):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Construct the full path
    image_path = os.path.normpath(os.path.join(folder_path, filename))

    # Save the image
    success = cv2.imwrite(image_path, image)

    if not success:
        raise IOError(f"Error: Could not save image to {image_path}")

    print(f"Image successfully saved at: {image_path}")


# Function that outputs image to a window. Closes window when key is pressed
def OutputImageToWindow(image, window_name: str = "Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # Wait until a key is pressed to close the window
    cv2.destroyAllWindows()


# Interface
def Interface():
    print("Welcome to the Image Bounding Box Drawer!")

    choice = input(
        "Do you want to visualize a single image or process multiple images? (single/multiple): ").strip().lower()

    if choice == "single":
        image_path = input("Enter the path of the image: ")
        if not os.path.exists(image_path):
            print("Error: The specified image does not exist!")
            return

        image = DrawBBoxManual(image_path)
        action = input("Do you want to display the image or save it? (display/save): ").strip().lower()

        if action == "save":
            save_folder = input("Enter the folder path to save images: ").strip()
            if not os.path.exists(save_folder) or save_folder == "":
                print("Error: The specified save folder does not exist, using the default!")
                save_folder = "DataVisualization/BoundingBoxVisualization"
            SaveImageToFolder(image, save_folder, filename="manual_image.jpg")
        else:
            OutputImageToWindow(image, "Manual Image")

    elif choice == "multiple":
        file_path = input("Enter the file path containing image annotations: ").strip()

        if (not os.path.exists(file_path)) or file_path == "":
            print("Error: The specified file path does not exist, using the default!")
            file_path = "TrainingData/wider_face_annotations.txt"

        folder_path = input("Enter the directory path containing images: ").strip()

        if (not os.path.exists(folder_path)) or folder_path == "":
            print("Error: The specified folder does not exist, using the default!")
            folder_path = "TrainingData/WIDER_train/images"

        from Utils.annotation_parser import GetAnnotations
        annotations = GetAnnotations(file_path)

        if not annotations:
            print("No annotations found to process.")
            return

        use_random = input("Do you want to use random images? (y/n): ").strip().lower()

        if use_random == 'y':
            x = int(input("How many random images do you want to use?: ").strip())
            annotations = random.sample(annotations,
                                        min(x, len(annotations)))  # Ensure we don't exceed available images

        action = input(
            "Do you want to display images in a window or save them to a folder? (display/save): ").strip().lower()

        if action == 'save':
            save_folder = input("Enter the folder path to save images: ").strip()
            if (not os.path.exists(save_folder)) or save_folder == "":
                print("Error: The specified save folder does not exist, using the default!")
                save_folder = "DataVisualization/BoundingBoxVisualization"

        from Utils.annotation_class import Annotation
        i = 0
        for annotation in annotations:
            from Utils.boundingbox_class import BoundingBox
            image_path = os.path.normpath(os.path.join(folder_path, annotation.path_name))
            image = DrawBBoxManual(image_path)

            if action == 'save':
                SaveImageToFolder(image, save_folder, filename=f"image_{i}.jpg")
            else:
                if i < 20:
                    print("Press 0 for next image!")
                    OutputImageToWindow(image, "image_" + str(i))
            i += 1

    print("Operation complete!")
    return


def main():
    Interface()


if __name__ == "__main__":
    main()
