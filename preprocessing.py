import os
import cv2

input_folder = 'images'
output_folder = 'preprocessed_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Iterate through each file
for file in files:
    if file.endswith('.jpg'):  # Consider only JPG files, modify if needed
        try:# Read the image
            image_path = os.path.join(input_folder, file)
            image = cv2.imread(image_path)

            adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # Resize the image
            resized_image = cv2.resize(adjusted_image, (224,224))
            

            # Save the resized image to the output folder
            new_filename = f"{file}_preprocessed{'.jpg'}"

            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, resized_image) 

            print(f"Resized image saved: {output_path}")
        except Exception as e:
            print(f"Error processing image {file}: {e}")


