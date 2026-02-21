# Malaria Blood Smear CNN (ParaDetect-AI)

A simple CNN in TensorFlow/Keras that classifies malaria blood smear images into **Parasitized** vs **Uninfected**.

## Data

Place the malaria cell images so that:

- `data/cell_images/Parasitized/` — parasitized images
- `data/cell_images/Uninfected/` — uninfected images

You can use the [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (e.g. from Kaggle/NIH).

## Setup and run

```bash
pip install -r requirements.txt
python malaria_cnn.py
```

## CNN architecture – block-by-block explanation

| Block | Layers | Role |
|-------|--------|------|
| **0. Config** | `IMG_SIZE`, `BATCH_SIZE`, etc. | Image size (64×64), batch size, and paths. |
| **1. First conv block** | `Input` → `Conv2D(32, 3×3)` → `BatchNorm` → `MaxPool(2×2)` → `Dropout(0.25)` | Learns low-level features (edges, textures). Pool reduces size; dropout regularizes. |
| **2. Second conv block** | `Conv2D(64, 3×3)` → `BatchNorm` → `MaxPool(2×2)` → `Dropout(0.25)` | More filters capture mid-level patterns (e.g. cell shapes). |
| **3. Third conv block** | `Conv2D(128, 3×3)` → `BatchNorm` → `MaxPool(2×2)` → `Dropout(0.25)` | Higher-level, more abstract features for parasitized vs uninfected. |
| **4. Classifier** | `Flatten` → `Dense(128)` → `BatchNorm` → `Dropout(0.5)` → `Dense(2, softmax)` | Converts feature maps to a vector, then two-class probabilities (Parasitized / Uninfected). |

- **Conv2D**: learns local patterns via 3×3 filters; more filters in deeper blocks.
- **BatchNorm**: normalizes activations for stable, faster training.
- **MaxPooling**: halves spatial size and adds slight invariance to small shifts.
- **Dropout**: reduces overfitting by randomly zeroing activations.
- **Flatten + Dense**: global decision from the extracted features.

All of this is reflected in the comments inside `malaria_cnn.py`.
