import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from skimage.feature import hog, local_binary_pattern
from mahotas.features import zernike_moments
from tensorflow.keras.datasets import mnist
import joblib
import random
import cv2

# Feature extraction function
def extract_features(image):
    # Pixel intensities
    pixel_intensities = image.flatten()
    
    # HOG features
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Corner detection
    corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04).flatten()
    
    # Edge detection
    edges = cv2.Canny(image, 100, 200).flatten()
    
    # Texture (LBP)
    lbp_features = local_binary_pattern(image, P=8, R=1, method="uniform").flatten()
    
    # Zernike moments
    zm_features = zernike_moments(image, radius=15)
    
    # Combine all features
    combined_features = np.hstack([pixel_intensities, hog_features, corners, edges, lbp_features, zm_features])
    return combined_features

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Load the pre-trained model
model = joblib.load('digit_recognition_model.pkl')

# Randomly select 10 test images
random_indices = random.sample(range(0, len(x_test)), 10)
selected_images = x_test[random_indices]
selected_labels = y_test[random_indices]

# Streamlit interface
st.title("Handwritten Digit Recognition")
st.write("This app classifies handwritten digits from the MNIST dataset.")

# Display the randomly selected 10 images in a matrix (2x5 grid)
cols = st.columns(5)  # Create 5 columns for the grid

# Generate a list of image options for selection
image_options = []
for i in range(10):
    with cols[i % 5]:  # Alternate the images across columns
        st.image(selected_images[i], caption=f"True: {selected_labels[i]}", use_column_width=True)
    image_options.append(f"Image {i + 1} (True Label: {selected_labels[i]})")

# User selects one image to make a prediction
selected_image_index = st.selectbox("Select an image to predict:", options=image_options)

# Extract the index of the selected image
selected_index = image_options.index(selected_image_index)

# Get the selected image and its true label
test_image = selected_images[selected_index]
test_label = selected_labels[selected_index]

# Extract features from the selected test image
test_image_features = extract_features(test_image).reshape(1, -1)

# Make a prediction
prediction = model.predict(test_image_features)

# Display the selected image and its prediction
st.image(test_image, caption=f"Selected Image - True Label: {test_label}", use_column_width=True)
st.write(f"Predicted Label: {prediction[0]}")


