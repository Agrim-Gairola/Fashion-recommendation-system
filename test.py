import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import os
from pathlib import Path
import pickle
import numpy as np

base_dir = Path(__file__).resolve().parent
data_dir = base_dir  # not base_dir / "images"

feature_list = np.array(pickle.load(open(data_dir / "embeddings.pkl", "rb")))
filenames    = np.array(pickle.load(open(data_dir / "filenames.pkl",  "rb")))


model= ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])
img = image.load_img('pexels-skgphotography-2270078.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result=result / norm(result)
neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)
print("filenames.shape:", filenames.shape)
print("feature_list.shape:", feature_list.shape)


for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512,512)))
    cv2.waitKey(0)

