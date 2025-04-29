import streamlit as st
from PIL import Image
from app.model import load_model, predict

st.set_page_config(page_title="Points Checker")

st.title("👀 Are the Points from the Same Figure?")

model = load_model()

uploaded_file = st.file_uploader("Upload a 224x224 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        result = predict(model, image)
        if result:
            st.success("✅ The points belong to the same figure!")
        else:
            st.error("❌ The points belong to *different* figures.")
