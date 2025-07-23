import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load model and data
@st.cache_resource
def load_model_and_data():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = np.array(pickle.load(open('filenames.pkl', 'rb')))
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    return model, neighbors, filenames

model, neighbors, filenames = load_model_and_data()

# Feature extractor
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)
    return normalized

# Save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs("Uploads", exist_ok=True)
        with open(os.path.join("Uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
        return os.path.join("Uploads", uploaded_file.name)
    except:
        return None

# UI
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("👗 Fashion Recommender System")
st.markdown("Upload a fashion image and get visually similar recommendations!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = save_uploaded_file(uploaded_file)
    if image_path:
        col1, col2 = st.columns(2)

        with col1:
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔎 Finding recommendations..."):
            features = extract_features(image_path, model)
            distances, indices = neighbors.kneighbors([features])

        with col2:
            st.subheader("Recommended Items:")
            for idx in indices[0]:
                image_file = filenames[idx]
                if os.path.exists(image_file):
                    st.image(image_file, width=200, caption=os.path.basename(image_file))
                else:
                    st.warning(f"Missing: {image_file}")
    else:
        st.error("Failed to save uploaded image.")
