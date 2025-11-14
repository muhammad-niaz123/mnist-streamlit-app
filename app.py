import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 Handwritten Digit Recognition App")
st.write("Upload an image of a handwritten digit (0–9) to see the model's prediction.")

model = tf.keras.models.load_model("digit_model.h5")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    #img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction[0])
    st.subheader(f"✨ Predicted Digit: {predicted_digit}")
