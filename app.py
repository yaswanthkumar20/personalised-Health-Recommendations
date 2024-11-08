from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import logging
from model import *

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MODEL_PATH = 'face_analyzer_model.pkl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_prediction(image_path):

    try:
        age, gender, gender_probability = analyze_face(image_path)
        
        if age is None or gender is None:
            return "Face analysis failed", 0.0, None
        
        result = f"Age: {age}, Gender: {gender}"
        confidence =   f"{gender_probability:.2f}"
        rec = generate_recommendations(age)  # Convert probability to percentage
        
        # Generate and save the result image
        result_image = display_result(image_path, age, gender, gender_probability)
        result_filename = f'result_{os.path.basename(image_path)}'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        
        return result, confidence, result_filename, rec
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return f"Error during prediction: {str(e)}", 0.0, None, None


@app.route('/')
@app.route('/home')
def home():
    return render_template('main.html')

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return render_template('result.html', prediction='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file")
        return render_template('result.html', prediction='No file uploaded')

    if not allowed_file(file.filename):
        app.logger.error("Invalid file type")
        return render_template('result.html', prediction='Invalid file type')

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prediction, confidence, result_filename, rec = get_prediction(file_path)
    image_url = f'/upload/{filename}'
    result_image_url = f'/upload/{result_filename}' if result_filename else None

    return render_template('result.html', recommendations=rec, prediction=prediction, prediction_percentage=confidence, image_url=image_url, result_image_url=result_image_url)

if __name__ == '__main__':
    app.run(debug=True)






