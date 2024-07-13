import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image

# Load the model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess the image
def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0  # Normalize to [0, 1]
    img = img[tf.newaxis, :]
    return img

# Streamlit app
st.title("Neural Style Transfer")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Load and display content image
    content_image = load_image(content_file)
    st.subheader("Content Image")
    st.image(np.squeeze(content_image), use_column_width=True, clamp=True)

    # Load and display style image
    style_image = load_image(style_file)
    st.subheader("Style Image")
    st.image(np.squeeze(style_image), use_column_width=True, clamp=True)

    # Perform style transfer
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    # Normalize and clip the stylized image to ensure it's within the [0.0, 1.0] range
    stylized_image = np.squeeze(stylized_image)
    stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0).numpy()

    # Display stylized image
    st.subheader("Stylized Image")
    st.image(stylized_image, use_column_width=True, clamp=True)

    # Save stylized image
    result_image_path = "generated_img.jpg"
    cv2.imwrite(result_image_path, cv2.cvtColor(stylized_image * 255, cv2.COLOR_RGB2BGR))
    st.write("Stylized image saved as `generated_img.jpg`")

