# Cat or Dog Recognizer

A simple web application that uses machine learning to determine whether an uploaded image is a cat or a dog.

## Overview

This application utilizes the MobileNetV2 model and the Annoy library to perform image recognition. It allows users to upload an image and provides instant feedback on whether the image contains a cat or a dog.

## Requirements

Make sure you have the following dependencies installed:

- streamlit
- Pillow
- annoy
- numpy
- TensorFlow

You can install them by running the following command:

```shell
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/LucasAbreu89/cat_n_dog_recognizer.git
   ```

## How It Works

1. The MobileNetV2 model is used as a feature extractor. The last layer of the model is removed to obtain a feature vector of 1280 dimensions.
2. The uploaded image is resized and preprocessed to match the input requirements of MobileNetV2.
3. The preprocessed image is passed through the model to obtain its feature vector.
4. The feature vector is compared with the cat and dog Annoy indexes.
5. The nearest neighbors in each index are retrieved along with their distances.
6. If the closest image in the index is below the defined distance threshold, the application identifies the image as a cat or a dog accordingly.
7. The result is displayed on the web interface.

## Customization

You can customize the distance threshold by modifying the `distance_threshold` variable in the `app.py` file.

Additionally, you can replace the cat and dog Annoy indexes with your own indexes to perform recognition on different categories of images. Make sure the indexes are saved as `cat_images.ann` and `dog_images.ann` in the project directory.

## Acknowledgements

- The MobileNetV2 model is used from the TensorFlow Keras library.
- The Annoy library is used for efficient approximate nearest neighbor search.
