import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model using tf.keras.models.load_model
MODEL_PATH = "model.savedmodel"
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("Helmet Compliance Detector ⛑️")

uploaded_file = st.file_uploader("Upload the data of a monitoring device", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Must match model input
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")  # Ensure it's in a displayable format
        st.image(image, caption="Uploaded Image", width=400)
    except Exception as e:
        st.error(f"Failed to load image. Error: {e}")

    try:
        img_array = preprocess_image(image)

        with st.spinner("Analyzing worker compliance..."):
            prediction = model.predict(img_array)
            class_names = ["ON Helmet", "NO Helmet"]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        if predicted_class == "ON Helmet":
            st.markdown(
                f"<h3 style='color: green;'>✅ Verdict: Worker in Compliance ({confidence:.2f}% confidence)</h3>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h3 style='color: red;'>❌ Verdict: Incompliant Worker Detected ({confidence:.2f}% confidence)</h3>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
