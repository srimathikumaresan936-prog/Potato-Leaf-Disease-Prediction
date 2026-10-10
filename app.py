import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join("models", "potato_leaf_model.keras")
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run `python train.py --dataset-dir PATH_TO_POTATO_DATASET` first."
        )
    return tf.keras.models.load_model(MODEL_PATH)


model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)


def predict_leaf(image):
    if image is None:
        return {"Please upload a potato leaf image.": 1.0}

    if model is None:
        return {"Model not ready — train first": 1.0}

    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The model already includes Rescaling(1./255).
    probabilities = model.predict(image_array, verbose=0)[0]
    return {
        CLASS_NAMES[i].replace("Potato___", "").replace("_", " "): float(
            probabilities[i]
        )
        for i in range(len(CLASS_NAMES))
    }


demo = gr.Interface(
    fn=predict_leaf,
    inputs=gr.Image(type="pil", label="Upload a potato leaf image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction and confidence"),
    title="Potato Leaf Disease Prediction",
    description=(
        "Upload a potato leaf image to classify it as Early Blight, "
        "Late Blight, or Healthy. This is an educational model, not a "
        "professional agricultural diagnosis."
    ),
)

if __name__ == "__main__":
    demo.launch()
