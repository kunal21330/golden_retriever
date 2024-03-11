import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the input folder for preprocessed images
input_folder_images = 'preprocessed_images'

# Set the input folder for extracted features
input_folder_features = 'normalized_features'

# List all files in the input folder for images
files_images = os.listdir(input_folder_images)

# List all files in the input folder for features
files_features = os.listdir(input_folder_features)

# Calculate statistics for image sizes
image_sizes = []
for file in files_images:
    if file.endswith('.jpg'):  # Consider only JPG files, modify if needed
        image_path = os.path.join(input_folder_images, file)
        image = cv2.imread(image_path)
        image_sizes.append(image.shape[:2])  # Store image dimensions (height, width)

image_sizes = np.array(image_sizes)

# Plot histogram of image widths
plt.hist(image_sizes[:, 1], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Image Widths')
plt.xlabel('Width')
plt.ylabel('Frequency')
plt.show()

# Plot histogram of image heights
plt.hist(image_sizes[:, 0], bins=20, color='green', alpha=0.7)
plt.title('Histogram of Image Heights')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

# Calculate statistics for extracted features
feature_lengths = []
for file in files_features:
    if file.endswith('.npy'):  # Consider only numpy files, modify if needed
        features_path = os.path.join(input_folder_features, file)
        features = np.load(features_path)
        feature_lengths.append(len(features))

# Plot histogram of feature lengths
plt.hist(feature_lengths, bins=20, color='orange', alpha=0.7)
plt.title('Histogram of Feature Lengths')
plt.xlabel('Feature Length')
plt.ylabel('Frequency')
plt.show()

# Calculate summary statistics for feature lengths
mean_feature_length = np.mean(feature_lengths)
median_feature_length = np.median(feature_lengths)
std_feature_length = np.std(feature_lengths)

print(f"Mean Feature Length: {mean_feature_length}")
print(f"Median Feature Length: {median_feature_length}")
print(f"Standard Deviation of Feature Length: {std_feature_length}")
