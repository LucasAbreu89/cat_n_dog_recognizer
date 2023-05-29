import os
from annoy import AnnoyIndex
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Download the MobileNetV2 model and remove the last layer
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='max')

# This is your vectorizer
image_vectorizer = Model(inputs=base_model.input, outputs=base_model.output)

def vectorize_image(img_path):
    # Load image and convert to RGB
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))

    # Preprocess it for MobileNetV2
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Create a vector
    vector = image_vectorizer.predict(x)

    return vector


# Assuming that the images are in the folder "cat_images"
folder_path = "dog_images"
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Assume each vector has 1280 dimensions (MobileNetV2 with pooling='max')
index = AnnoyIndex(1280, 'angular')  

# Iterate over your images, vectorize them and add to the Annoy index
image_vectors = []
for i, image_path in enumerate(image_paths):
    vector = vectorize_image(image_path)
    index.add_item(i, vector[0])
    image_vectors.append((image_path, vector))

# Build the Annoy index
index.build(10)  # 10 trees for good balance between speed and accuracy

# Save the index for later use
index.save('dog_images.ann')
