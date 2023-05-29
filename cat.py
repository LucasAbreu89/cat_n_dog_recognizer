import streamlit as st
from PIL import Image
from annoy import AnnoyIndex
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import time

# Download the MobileNetV2 model and remove the last layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')

# This is your vectorizer
image_vectorizer = Model(inputs=base_model.input, outputs=base_model.output)

def vectorize_image(img):
    # Resize image
    img = img.resize((224, 224))

    # Preprocess it for MobileNetV2
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Create a vector
    vector = image_vectorizer.predict(x)

    return vector

# Assume each vector has 1280 dimensions (MobileNetV2 with pooling='max')
cat_index = AnnoyIndex(1280, 'angular') 
cat_index.load('cat_images.ann')  # load the cat Annoy index

dog_index = AnnoyIndex(1280, 'angular') 
dog_index.load('dog_images.ann')  # load the dog Annoy index

# Adding some CSS styles
st.markdown(
    """
    <style>
    .reportview-container {
        background: #ffffe6;
    }
    .big-font {
        font-size:50px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Cat or Dog Recognizer')
st.markdown("**Upload an image and I'll tell you if it's a cat or a dog!**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# Set a threshold for distance
distance_threshold = 0.4

if uploaded_file is not None:
    # Convert the file to an image
    img = Image.open(uploaded_file)

    # Show the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Adding a spinner for calculating part
    with st.spinner('Analyzing the image...'):
        time.sleep(2)  # Just to mimic some calculation, you can remove it

        # Vectorize the image
        vector = vectorize_image(img)

        # Search the cat Annoy index
        cat_indices, cat_distances = cat_index.get_nns_by_vector(vector[0], 1, include_distances=True)  # find the 1 nearest neighbor

        # Search the dog Annoy index
        dog_indices, dog_distances = dog_index.get_nns_by_vector(vector[0], 1, include_distances=True)  # find the 1 nearest neighbor

        # If the closest image in the index is close enough, say it's a cat or a dog
        if cat_distances[0] < distance_threshold and cat_distances[0] < dog_distances[0]:
            st.markdown('<p class="big-font">It\'s a cat! 🐱</p>', unsafe_allow_html=True)
        elif dog_distances[0] < distance_threshold and dog_distances[0] < cat_distances[0]:
            st.markdown('<p class="big-font">It\'s a dog! 🐶</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="big-font">I am not sure what it is. 🤔</p>', unsafe_allow_html=True)

