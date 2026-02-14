import os

# DO THIS BEFORE ANY OTHER IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU to avoid Metal/GPU conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1' # Limit threads to prevent the mutex crash


import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Use a decorator to load the model only once and prevent memory leaks
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("indian_food_classifier.h5")

model = load_my_model()

with open("label_map.json") as f:
    class_names = json.load(f)

st.title("🍛 Indian Food Classifier")

uploaded = st.file_uploader("Upload a food image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    # Show the image immediately so the user knows it worked
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner('Identifying dish...'):
        img_resized = img.resize((224,224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        idx = str(np.argmax(preds)) # JSON keys are often strings
        
        # Check if class_names is a list or a dict
        label = class_names.get(idx, "Unknown") if isinstance(class_names, dict) else class_names[int(idx)]
        confidence = np.max(preds)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")


    #python -m streamlit run app.py - to run 