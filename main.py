import streamlit as st
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ────────────────────────── paths ──────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # folder where main.py lives
DATA_DIR = BASE_DIR                                 # adjust if .pkl files live elsewhere
UPLOAD_DIR = BASE_DIR / "Uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ─────────────────── cache expensive objects ───────────────
@st.cache_resource
def load_model_and_data():
    # model
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

    # data
    feature_list = np.array(pickle.load(open(DATA_DIR / "embeddings.pkl", "rb")))
    filenames    = np.array(pickle.load(open(DATA_DIR / "filenames.pkl",  "rb")))

    nbrs = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="euclidean")
    nbrs.fit(feature_list)

    return model, nbrs, filenames

model, neighbors, filenames = load_model_and_data()

# ─────────────────── streamlit UI ──────────────────────────
st.title("Fashion Recommender System")

def save_uploaded_file(uploaded_file):
    try:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features  = model.predict(img_array).flatten()
    return features / norm(features)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path:
        st.image(Image.open(saved_path), caption="Uploaded image", use_column_width=True)

        query_feat = extract_features(saved_path, model)
        distances, indices = neighbors.kneighbors([query_feat])

        st.subheader("Similar items:")
        for idx in indices[0]:
            sim_path = Path(filenames[idx])
            if sim_path.exists():
                st.image(sim_path, width=150)
