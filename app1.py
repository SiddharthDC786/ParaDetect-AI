from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===============================
# Load Trained Model
# ===============================
model = load_model("best_malaria_model_fixed.h5")

# ===============================
# Upload Folder Setup
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Image Preprocessing
# ===============================
def prepare_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0   # normalize
    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# Home Route
# ===============================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Open & preprocess
    image = Image.open(filepath).convert("RGB")
    processed = prepare_image(image)

    # Measure processing time
    start_time = time.time()
    predictions = model.predict(processed, verbose=0)[0]
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    # Get class index
    class_index = np.argmax(predictions)

    # Probabilities
    parasitized_prob = round(float(predictions[0] * 100), 2)
    uninfected_prob = round(float(predictions[1] * 100), 2)

    # Final Result Logic
    if class_index == 0:
        result = "Parasitized"
        color = "#e74c3c"        # red
        status_icon = "⚠️"
        confidence = parasitized_prob
    else:
        result = "Uninfected"
        color = "#2ecc71"        # green
        status_icon = "✅"
        confidence = uninfected_prob

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        parasitized_prob=parasitized_prob,
        uninfected_prob=uninfected_prob,
        processing_time=processing_time,
        image_path=filepath,
        color=color,
        status_icon=status_icon
    )

# ===============================
# Health Check Route
# ===============================
@app.route("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None
    }

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)

