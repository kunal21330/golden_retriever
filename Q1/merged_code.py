import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

input_folder = 'Q1\images'
output_folder = 'Q1\\normalized_features'

if not os.path.exists(output_folder):
     os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Loading the pre-trained ResNet50 model without the top layers
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for file in files:
    if file.endswith('.jpg'):  
        try:
            image_path = os.path.join(input_folder, file)
            image = cv2.imread(image_path)
            
            # Applying preprocessing steps
            # 1. Adjust contrast
            adjusted_image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
            
            # 2. Resize the image to 224x224
            resized_image = cv2.resize(image, (224, 224))
            
            # # 3. Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # 4. Normalize pixel values to be within the range [0, 1]
            normalized_image = rgb_image / 255.0
            
        except Exception as e:
            print(f"Error processing image {file}: {e}")


         #i am trying to continue code here only without saving image into a separate folder to save space and computation 
            

#____________________________________________________________________________________________________________________________________________________
            


        try:

            image_for_extraction = preprocess_input(normalized_image) # Preprocess for ResNet50

            # Expand dimensions to create a batch of size 1
            image_for_extraction = np.expand_dims(normalized_image, axis=0)

            # Extract features from the image using the pre-trained ResNet50 model
            features = model.predict(image_for_extraction)
            
            # Flatten the feature tensor to a 1D vector
            features = features.flatten()

        except Exception as e:
            print(f"Error extracting features from image {file}:{e}")


#____________________________________________________________________________________________________________________________________________________
            
            #normalizing extracted features and saving them in folder
        try:

             # Normalize the features
            normalized_features = (features - np.mean(features)) / np.std(features)
            normalized_features_filename = f"{file[:-4]}_normalized.npy"  # Remove the file extension from the original filename
            normalized_features_path = os.path.join(output_folder, normalized_features_filename)
            np.save(normalized_features_path, normalized_features)
            
            print(f"Normalized features saved to {normalized_features_filename}")
        except Exception as e:
            print(f"Error processing features {file}: {e}")
        
       