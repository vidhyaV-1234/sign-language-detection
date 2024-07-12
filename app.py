import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model with .h5 extension
model = tf.keras.models.load_model("sign_minst_cnn.h5")

# Define the class labels
classlabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def get_letter(result):
    return classlabels[int(result)]

# Streamlit app
st.title("Sign Language MNIST Classifier")

st.write("Upload an image of a hand sign and the model will predict the corresponding letter.")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input shape
    image = image / 255.0  # Normalize

    # Predict the class
    prediction = model.predict(image)
    result = np.argmax(prediction)
    letter = get_letter(result)

    st.write(f'The model predicts this sign represents: {letter}')
