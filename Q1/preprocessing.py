import os
import cv2
import numpy as np

input_folder = 'images'
output_folder = 'preprocessed_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Iterate through each file
for file in files:
    if file.endswith('.jpg'):  # Consider only JPG files, modify if needed
        try:
            # Read the image
            image_path = os.path.join(input_folder, file)
            image = cv2.imread(image_path)
            
            # Apply preprocessing steps
            # 1. Adjust contrast
            adjusted_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
            
            # 2. Resize the image to 224x224
            resized_image = cv2.resize(image, (224, 224))
            
            # # 3. Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # 4. Normalize pixel values to be within the range [0, 1]
            normalized_image = resized_image / 255.0
            
            # 5. Mean subtraction
            # You may need to compute the mean pixel values of your dataset
            # and subtract them from each pixel in the image
            # mean = np.array([0.485, 0.456, 0.406])  # Mean values for ImageNet dataset
            # std = np.array([0.229, 0.224, 0.225])   # Standard deviation values for ImageNet dataset
            # normalized_image = (normalized_image - mean) / std
            
            # Save the preprocessed image to the output folder
            new_filename = f"{file[:-4]}_preprocessed.jpg"  # Remove the file extension from the original filename
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, normalized_image * 255)  # Multiply by 255 to convert back to [0, 255] range before saving
            
            print(f"Preprocessed {file} -> {new_filename}")
        except Exception as e:
            print(f"Error processing image {file}: {e}")
