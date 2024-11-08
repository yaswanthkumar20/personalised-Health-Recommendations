# import pickle
# from model import FaceAnalyzer

# # Save the model
# model = FaceAnalyzer()
# with open('test_face_analyzer_model.pkl', 'wb') as file:
#     pickle.dump(model, file)
# print("Model saved successfully.")

# # Load the model
# try:
#     with open('test_face_analyzer_model.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import pickle

# model_path = "face_analyzer_model.pkl"

def analyze_face(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Analyze the face
    result = DeepFace.analyze(img_rgb, actions=['age', 'gender'], enforce_detection=False)

    # Extract age and gender predictions
    age = result[0]['age']
    gender = result[0]['dominant_gender']
    gender_probability = result[0]['gender'][gender]

    return age, gender, gender_probability

if __name__ == "__main__":
    # with open(model_path, 'rb') as file:
    #     face_analyzer = pickle.load(file)
    age, gender, gender_probability = analyze_face("upload/Myface.jpg")
    print(age)
    print(gender)
    print(gender_probability)
    print("I am here")