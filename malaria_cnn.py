"""
Malaria Blood Smear CNN Classifier
==================================
A simple CNN in TensorFlow/Keras to classify malaria blood smear images
into: Parasitized vs Uninfected.

Expects data in: data/cell_images/
  - data/cell_images/Parasitized/
  - data/cell_images/Uninfected/

(e.g. from the Malaria Cell Images Dataset on Kaggle / NIH)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ---------------------------------------------------------------------------
# BLOCK 0: Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = (64, 64)   # Smaller images = faster training; 64x64 is common for this dataset
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "dataset"  # Parent folder containing Parasitized/ and Uninfected/


def build_malaria_cnn(input_shape=(64, 64, 3), num_classes=2):
    """
    Build a simple CNN for binary classification (Parasitized / Uninfected).
    Each block is explained below.
    """
    model = keras.Sequential([

        # -------------------------------------------------------------------
        # BLOCK 1: Input + First Convolutional Block
        # -------------------------------------------------------------------
        # Input: (height, width, channels). We use 64x64 RGB images.
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),



        # Conv2D: Learns local patterns (edges, textures). 32 filters, 3x3 kernels.
        # Same padding keeps spatial size; we'll reduce it with MaxPooling.
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv1"),
        # BatchNorm: Normalizes activations per channel → faster, more stable training.
        layers.BatchNormalization(name="bn1"),
        # MaxPooling: Downsample 2x → less computation, slight translation invariance.
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        # Dropout: Randomly zero 25% of activations to reduce overfitting.
        layers.Dropout(0.25, name="drop1"),

        # -------------------------------------------------------------------
        # BLOCK 2: Second Convolutional Block
        # -------------------------------------------------------------------
        # More filters (64) to capture higher-level features (e.g. cell shapes).
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # -------------------------------------------------------------------
        # BLOCK 3: Third Convolutional Block
        # -------------------------------------------------------------------
        # 128 filters for more abstract, class-relevant features.
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        layers.Dropout(0.25, name="drop3"),

        # -------------------------------------------------------------------
        # BLOCK 4: Flatten + Dense Classifier
        # -------------------------------------------------------------------
        # Flatten: Convert 3D feature maps to 1D vector for the dense layers.
        layers.Flatten(name="flatten"),
        # Dense: Fully connected layer; 128 units with ReLU.
        layers.Dense(128, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn4"),
        layers.Dropout(0.5, name="drop4"),
        # Output: num_classes units + softmax for class probabilities (Parasitized, Uninfected).
        layers.Dense(num_classes, activation="softmax", name="output"),
    ])
    return model


def get_model(num_classes=2):
    """Build model with 2-class softmax (Parasitized, Uninfected)."""
    return build_malaria_cnn(
        input_shape=(*IMG_SIZE, 3),
        num_classes=num_classes,
    )


def load_data(data_dir=DATA_DIR, validation_split=0.2, seed=42):
    """Load images from Parasitized/ and Uninfected/ subfolders using Keras image_dataset_from_directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please download the malaria cell images dataset and place it so that:\n"
            f"  {os.path.join(data_dir, 'Parasitized')} and\n"
            f"  {os.path.join(data_dir, 'Uninfected')}\n"
            "exist and contain .png images."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )
    return train_ds, val_ds


def main():
    print("TensorFlow version:", tf.__version__)
    train_ds, val_ds = load_data()
    num_classes = 2  # Parasitized, Uninfected
    model = get_model(num_classes=num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )
    return model, history


if __name__ == "__main__":
    model, history = main()

    # SAVE TRAINED MODEL
    model.save("best_malaria_model_fixed.h5")
    print("✅ Model saved as best_malaria_model_fixed.h5")
