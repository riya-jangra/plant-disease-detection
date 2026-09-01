from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
print("Loading model from:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------
# Disease classes
# -----------------------------
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

# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    print("Received files:", request.files)

    if not request.files:
        return """
        <h2>No file was received by the server.</h2>
        <p>Please go back and select an image again.</p>
        <a href="/">Go Back</a>
        """

    file = request.files.get("image")

    if file is None:
        return """
        <h2>Image field not found.</h2>
        <p>The uploaded file field is missing.</p>
        <a href="/">Go Back</a>
        """

    if file.filename == "":
        return """
        <h2>No image selected.</h2>
        <a href="/">Go Back</a>
        """

    try:
        # Read uploaded image
        image = Image.open(file).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        print("Starting prediction...")

        predictions = model.predict(image_array, verbose=0)

        print("Prediction completed!")

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
        </head>

        <body>

            <h1>🌿 Plant Disease Detection Result</h1>

            <h2>Disease: {predicted_class}</h2>

            <h2>Confidence: {confidence:.2f}%</h2>

            <br>

            <a href="/">Check another image</a>

        </body>
        </html>
        """

    except Exception as e:

        print("Prediction error:", repr(e))

        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <br>
        <a href="/">Go Back</a>
        """, 500
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
