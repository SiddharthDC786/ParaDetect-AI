import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("best_malaria_model_fixed.h5")

# Image path (change this)
IMG_PATH = "test_image.jpg"   # put any cell image here

# Load & preprocess image
img = image.load_img(IMG_PATH, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

prediction = model.predict(img_array)[0][0]
confidence = prediction * 100

if prediction > 0.5:
    print(f"🦠 Parasitized (Malaria Detected)")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"✅ Uninfected (Healthy Cell)")
    print(f"Confidence: {(100 - confidence):.2f}%")