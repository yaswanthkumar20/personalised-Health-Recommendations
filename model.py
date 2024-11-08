import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import pickle

def analyze_face(image_path):
    try:
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

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None

def display_result(image_path, age, gender, gender_probability):
    # Read and resize the image for display
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Resize image for display, keeping aspect ratio

    # Convert PIL Image to numpy array
    img_np = np.array(img)

    # Convert RGB to BGR (for OpenCV display)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, f"Age: {age}", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(img_bgr, f"Gender: {gender} ({gender_probability:.2f})", (10, 60), font, 0.8, (0, 255, 0), 2)

    return img_bgr

def generate_recommendations(age):
    recommendations = {
        "Food Recommendations": "",
        "Exercise Recommendations": "",
        "Books Recommendations": "",
        "Music Recommendations": ""
    }

    if age <= 10:
        recommendations["Food Recommendations"] = "Fruits, vegetables, whole grains, and proteins."
        recommendations["Exercise Recommendations"] = "Running, playground activities, Jumping rope, playing basketball"
        recommendations["Books Recommendations"] = "Goodnight Moon by Margaret Wise Brown, The Very Hungry Caterpillar by Eric Carle"
        recommendations["Music Recommendations"] = "Twinkle Twinkle Little Star, The Wheels on the Bus, Baby Shark"
    elif 11 <= age <= 20:
        recommendations["Food Recommendations"] = "Lean meats, beans, nuts, fruits, vegetables, and whole grains"
        recommendations["Exercise Recommendations"] = "Running, swimming, cycling, Weight lifting, resistance exercises, soccer, basketball"
        recommendations["Books Recommendations"] = "Harry Potter series by J.K. Rowling, The Hunger Games by Suzanne Collins"
        recommendations["Music Recommendations"] = "The Beatles, Queen, Imagine Dragons"
    elif 21 <= age <= 30:
        recommendations["Food Recommendations"] = "Lean meats, eggs, legumes, Avocados, nuts, olive oil, Include a variety of foods from all food groups"
        recommendations["Exercise Recommendations"] = "Stretching exercises, yoga, swimming, cycling, Lifting weights"
        recommendations["Books Recommendations"] = "The Night Circus by Erin Morgenstern, Normal People by Sally Rooney"
        recommendations["Music Recommendations"] = "Taylor Swift, Ed Sheeran, Hozier"
    elif 31 <= age <= 40:
        recommendations["Food Recommendations"] = "Lean meats, eggs, legumes, Avocados, nuts, olive oil, Include a variety of foods from all food groups"
        recommendations["Exercise Recommendations"] = "Lifting weights, yoga, swimming, cycling, Stretching exercises, yoga"
        recommendations["Books Recommendations"] = "Big Little Lies by Liane Moriarty, The Goldfinch by Donna Tartt"
        recommendations["Music Recommendations"] = "Coldplay, Radiohead"
    elif 41 <= age <= 50:
        recommendations["Food Recommendations"] = "Eggs, legumes, Avocados, nuts, olive oil, Lean meats"
        recommendations["Exercise Recommendations"] = "Lifting weights, yoga, Stretching exercises, walking, swimming, cycling"
        recommendations["Books Recommendations"] = "The Nightingale by Kristin Hannah, The Girl on the Train by Paula Hawkins"
        recommendations["Music Recommendations"] = "U2, Madonna, Maroon 5, P!nk"
    elif 51 <= age <= 60:
        recommendations["Food Recommendations"] = "Include a variety of foods from all food groups, Avocados, nuts, olive oil, Lean meats, eggs, legumes"
        recommendations["Exercise Recommendations"] = "Yoga, Stretching exercises, Lifting weights, yoga, walking, swimming, cycling"
        recommendations["Books Recommendations"] = "The Help by Kathryn Stockett, Where the Crawdads Sing by Delia Owens"
        recommendations["Music Recommendations"] = "Fleetwood Mac, Bruce Springsteen, Adele, Ed Sheeran"
    elif 61 <= age <= 70:
        recommendations["Food Recommendations"] = "Focus on nutrient-dense foods to meet nutritional needs without excess calories, Whole grains, fruits, vegetables for digestive health, Lean meats, dairy, and legumes to maintain muscle mass"
        recommendations["Exercise Recommendations"] = "Aim for at least 150 minutes of moderate-intensity aerobic activity each week, Include activities that improve balance to reduce the risk of falls, Resistance training to maintain bone density"
        recommendations["Books Recommendations"] = "All the Light We Cannot See by Anthony Doerr, The Immortal Life of Henrietta Lacks by Rebecca Skloot"
        recommendations["Music Recommendations"] = "The Beatles, Simon & Garfunkel, Elton John"
    elif age > 70:
        recommendations["Food Recommendations"] = "Focus on nutrient-dense foods to meet nutritional needs without excess calories, Whole grains, fruits, vegetables for digestive health, Lean meats, dairy, and legumes to maintain muscle mass"
        recommendations["Exercise Recommendations"] = "Activities that improve balance to reduce the risk of falls, Resistance training to maintain bone density, Aim for at least 150 minutes of moderate-intensity aerobic activity each week"
        recommendations["Books Recommendations"] = "The Help by Kathryn Stockett, Where the Crawdads Sing by Delia Owens"
        recommendations["Music Recommendations"] = "Elvis Presley, Frank Sinatra, Aretha Franklin"
    else:
        recommendations["Food Recommendations"] = "General balanced diet with all necessary nutrients."
        recommendations["Exercise Recommendations"] = "Regular moderate exercise tailored to individual capacity."
        recommendations["Books Recommendations"] = "Books based on personal interests."
        recommendations["Music Recommendations"] = "Music based on personal preferences."

    return recommendations

# Save the model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load the model
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Create and save the model
if __name__ == "__main__":
    print("\nI am here\n\n")
    

