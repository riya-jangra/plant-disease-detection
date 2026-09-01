from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model(3).keras"
)

# Temporary upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model from:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# DISEASE CLASSES
# ==========================================

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    print("================================")
    print("PREDICT REQUEST RECEIVED")
    print("Files received:", request.files)
    print("File field names:", list(request.files.keys()))
    print("================================")

    # Check whether any file was received
    if not request.files:
        return """
        <h2>No file received by Flask.</h2>
        <p>Please go back and select an image.</p>
        <a href="/">Go Back</a>
        """

    # Get image
    file = request.files.get("image")

    # If image field is missing
    if file is None:

        return f"""
        <h2>Image field not found.</h2>

        <p>The uploaded file field is missing.</p>

        <p><b>Received file fields:</b>
        {list(request.files.keys())}</p>

        <p><b>Received form fields:</b>
        {list(request.form.keys())}</p>

        <br>

        <a href="/">Go Back</a>
        """

    # Check filename
    if file.filename == "":
        return """
        <h2>No image selected.</h2>
        <a href="/">Go Back</a>
        """

    try:

        print("Image received:", file.filename)

        # ======================================
        # OPEN IMAGE
        # ======================================

        image = Image.open(file).convert("RGB")

        print("Image opened successfully.")

        # ======================================
        # RESIZE
        # ======================================

        image = image.resize((224, 224))

        # ======================================
        # CONVERT TO NUMPY
        # ======================================

        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Normalize pixels
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        print("Starting prediction...")

        # ======================================
        # MODEL PREDICTION
        # ======================================

        predictions = model.predict(
            image_array,
            verbose=0
        )

        print("Prediction completed.")

        # ======================================
        # GET RESULT
        # ======================================

        predicted_index = int(
            np.argmax(predictions[0])
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[0][predicted_index]
        ) * 100

        print("Disease:", predicted_class)
        print("Confidence:", confidence)

        # ======================================
        # RESULT PAGE
        # ======================================

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>Prediction Result</title>

            <style>

                body {{
                    font-family: Arial;
                    text-align: center;
                    padding-top: 50px;
                }}

                h1 {{
                    color: green;
                }}

                .result {{
                    margin: auto;
                    padding: 
