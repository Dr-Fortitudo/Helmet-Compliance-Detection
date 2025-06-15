import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import TFSMLayer
from keras import Sequential

# Load model using TFSMLayer
MODEL_PATH = "model.savedmodel"
layer = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
model = Sequential([layer])

# Streamlit UI
st.title("Helmet Compliance Detector ⛑️")

uploaded_file = st.file_uploader("Upload the data of a monitoring device", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Must match model's expected input
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        img_array = preprocess_image(image)

        with st.spinner("Analyzing worker compliance..."):
            output = model(img_array)  # returns dict
            prediction = output["sequential_11"].numpy()  # extract tensor
    
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
