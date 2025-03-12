import streamlit as st
import requests
from PIL import Image
import io

# API Endpoint (for deployment)
API_URL = "http://backend:5000/predict" 
# if only dockerized
# API_URL = "http://backend:5000/predict"
# if running locally
# API_URL = "http://localhost:5000/predict"


# Streamlit UI
st.set_page_config(page_title="Fashion MNIST Classifier", layout="centered")

st.title("Fashion MNIST Classifier")
st.write("Upload an image of a clothing item to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes for sending to API
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Authentication credentials
    auth = ("admin", "password")

    if st.button("Predict"):
        try:
            # Send request to API
            response = requests.post(API_URL, files={"file": image_bytes}, auth=auth)
            result = response.json()

            if "class" in result:
                # Display uploaded image again next to prediction
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption="Uploaded Image", width=100)
                with col2:
                    st.success(f"Predicted Class: {result['class']}")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {e}")
