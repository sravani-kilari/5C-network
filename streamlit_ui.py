import streamlit as st
import requests
from PIL import Image

st.title('Brain MRI Metastasis Segmentation')

uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI.', use_column_width=True)
    st.write("Classifying...")
    files = {'file': uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict/", files=files)
    st.write(response.json())
