import os
import numpy as np

# Set the input and output folders
input_folder = 'features'
output_folder = 'normalized_features'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Iterate through each file
for file in files:
    if file.endswith('.npy'):  
        try:
            # Load the extracted features
            features_path = os.path.join(input_folder, file)
            features = np.load(features_path)
            
            # Normalize the features
            normalized_features = (features - np.mean(features)) / np.std(features)
            
            # Save the normalized features to a numpy file
            normalized_features_filename = f"{file[:-4]}_normalized.npy"  # Remove the file extension from the original filename
            normalized_features_path = os.path.join(output_folder, normalized_features_filename)
            np.save(normalized_features_path, normalized_features)
            
            print(f"Normalized features saved to {normalized_features_filename}")
        except Exception as e:
            print(f"Error processing features {file}: {e}")
