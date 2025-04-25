import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Load model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("bloodgroup.h5")
    return model

class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

model = load_model()

# Title
st.title("Fingerprint-Based Blood Group Prediction")
st.markdown("Upload a fingerprint image and enter your personal details to predict your blood group.")

# Sidebar inputs
st.sidebar.header("Enter Your Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=300.0, step=0.5)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Fingerprint Image", type=["jpg", "jpeg", "png", "bmp"])

# Predict function
def predict_image(uploaded_image, model):
    img = uploaded_image.convert('RGB')  # Convert to RGB
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    return predicted_label

# Handle upload and prediction
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Fingerprint", width=200)

    # Predict
    predicted_group = predict_image(Image.open(uploaded_file), model)

    # Display result and details below image
    st.success(f"Predicted Blood Group: **{predicted_group}**")

    st.markdown("###  Prediction Summary")
    data = {
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Weight (kg)": [weight],
        "Predicted Blood Group": [predicted_group]
    }
    st.table(pd.DataFrame(data))
