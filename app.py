import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Constants
MODEL_PATH = r"COVID_19_Detection\covid_model.h5"
IMG_SIZE = (150, 150)
UPLOAD_FOLDER = r"COVID_19_Detection\uploads"

# Create necessary directories
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model ONCE at startup
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input
    return img_array

@app.route('/')
def home():
    """Render the file upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    file = request.files['file']
    if not file:
        return "No file uploaded!", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess the image
    img_array = preprocess_image(filepath)

    # **Reload the model before every prediction**
    model = load_model(MODEL_PATH)  # Reloads the model

    # Predict using the reloaded model
    prediction = model.predict(img_array)[0][0]
    predicted_label = "non-covid" if prediction > 0.5 else "covid"

    return render_template("result.html", prediction=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)