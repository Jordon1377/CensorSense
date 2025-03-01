import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def load_and_preprocess_image(image_path, target_size=(12, 12)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict(model, image_path):
    image = load_and_preprocess_image(image_path)
    face_class, bbox_reg, landmark_reg = model.predict(image)
    
    face_class = np.argmax(face_class, axis=-1)  # Get predicted class
    return face_class[0], bbox_reg[0], landmark_reg[0]

def main():
    model_path = input("Enter the path to the saved model: ")
    if not os.path.exists(model_path):
        print("Error: Model path does not exist.")
        return
    
    model = tf.keras.models.load_model(model_path)
    
    mode = input("Enter 'single' for one image or 'batch' for multiple images: ").strip().lower()
    
    if mode == 'single':
        image_path = input("Enter the path to the image: ")
        if not os.path.exists(image_path):
            print("Error: Image path does not exist.")
            return
        face_class, bbox, landmarks = predict(model, image_path)
        print(f"Face Classification: {face_class}\nBounding Box: {bbox}\nLandmarks: {landmarks}")
    
    elif mode == 'batch':
        image_paths = input("Enter the paths to images (comma-separated): ").split(',')
        image_paths = [path.strip() for path in image_paths if os.path.exists(path.strip())]
        
        if not image_paths:
            print("Error: No valid image paths provided.")
            return
        
        for image_path in image_paths:
            face_class, bbox, landmarks = predict(model, image_path)
            print(f"\nImage: {image_path}\nFace Classification: {face_class}\nBounding Box: {bbox}\nLandmarks: {landmarks}")
    else:
        print("Invalid mode. Please enter 'single' or 'batch'.")

if __name__ == "__main__":
    main()
