
import os
import pickle
from glob import glob

image_dir = "images"  # Adjust path if needed

filenames = glob(os.path.join(image_dir, "*.jpg"))
print("Found", len(filenames), "images")

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)
