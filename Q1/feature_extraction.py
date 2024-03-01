import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


input_folder = 'preprocessed_images'
output_folder = 'features'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the pre-trained ResNet50 model without the top layers
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# List all files in the input folder
files = os.listdir(input_folder)

# Iterate through each file
for file in files:
    if file.endswith('.jpg'):  # Consider only JPG files, modify if needed
        try:
            # Read the preprocessed image
            image_path = os.path.join(input_folder, file)
            image = cv2.imread(image_path)
            
            # Preprocess the input image
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            # image = cv2.resize(image, (224, 224))          # Resize to 224x224
            image = preprocess_input(image)                # Preprocess for ResNet50
            
            # Expand dimensions to create a batch of size 1
            image = np.expand_dims(image, axis=0)
            
            # Extract features from the image using the pre-trained ResNet50 model
            features = model.predict(image)
            
            # Flatten the feature tensor to a 1D vector
            features = features.flatten()
            
            # Save the features to a numpy file
            features_filename = f"{file[:-4]}_features.npy"  # Remove the file extension from the original filename
            features_path = os.path.join(output_folder, features_filename)
            np.save(features_path, features)
            
            print(f"Extracted features from {file} -> {features_filename}")
        except Exception as e:
            print(f"Error processing image {file}: {e}")
