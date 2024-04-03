from __future__ import print_function  # Only import for Python 2 compatibility
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model path
MODEL_PATH = 'model.h5'

# Assuming the model was compiled with appropriate loss and optimizer during training
model = load_model(MODEL_PATH)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    # Replace with your actual class name mapping function (modify as needed)
    class_names = {
        0: 'Speed Limit 20 km/h',
        1: 'Speed Limit 30 km/h',
        2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h' ,
        4: 'Speed Limit 70 km/h',
        5: 'Speed Limit 80 km/h',
        6: 'Speed Limit 90 km/h',
        7: 'Speed Limit 100 km/h',
        8: 'Speed Limit 120 km/h',
        35: 'Go straight',
        37 : 'Compulsory ahead or turn left',
        36 : 'Compulsory ahead or turn right',
        34 : 'Compulsory turn left',
        33 : 'Compulsory turn Right',
        32 : 'Restirction End',
        31 : 'Wild zone',
        24 : 'Narrow ahead (right) ',
        20:   'Turn right ',
        19: 'Turn left ',
        18 : 'Avoid alert',

        
        # ... (add all your class names)
    }
    return class_names.get(classNo, "Unknown")  # Handle unknown classes gracefully

def model_predict(img_path):
    """Predicts the class of an image using the loaded model.

    Args:
        img_path: Path to the image file.

    Returns:
        The predicted class name (string).
    """

    img = image.load_img(img_path, target_size=(224, 224))  # Assuming target size from training
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)

    # Make predictions (consider using predict_classes for multi-class)
    predictions = model.predict(img.reshape(1, 32, 32, 1))
    classIndex = np.argmax(predictions)

    return getClassName(classIndex)

@app.route('/', methods=['GET'])
def index():
    """Renders the main page template."""

    return render_template('index.html')  # Assuming index.html exists

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    """Handles file upload and prediction logic."""

    if request.method == 'POST':
        f = request.files['file']
        if f:
            # Secure filename handling
            filename = secure_filename(f.filename)
            # Create uploads folder if it doesn't exist
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)  # Create parent dir if needed

            f.save(upload_path)

            # Make prediction
            prediction = model_predict(upload_path)

            # Return the prediction (consider using a render template if needed)
            return prediction
        else:
            return "No file selected"  # Handle case where no file is uploaded

    return None

if __name__ == '__main__':
    # Set upload folder configuration (optional)
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # Run the app (debug mode for development)
    app.run(debug=True)
