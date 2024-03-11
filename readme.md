Q1:
first i tried to make 4 different codes and 4 different folder for 
# downloading image
        import pandas as pd
        import requests
        import os


        csv_file = 'A2_Data.csv'
        image_folder = 'images'

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        df = pd.read_csv(csv_file)

        for index, row in df.iterrows():
            image_id = row['id']
            image_urls = eval(row['Image'])  # Convert string representation of list to actual list
            for i, image_url in enumerate(image_urls):
                image_path = os.path.join(image_folder, f'{image_id}_{i}.jpg')  # Include index in filename
                try:
                    # Download image from URL
                    response = requests.get(image_url)
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Image {image_id}_{i} downloaded successfully.")
                except Exception as e:
                    print(f"Error downloading image {image_id}_{i}: {e}")


# preprocessing image

        import os
        import cv2
        import numpy as np


        input_folder = 'images'
        output_folder = 'preprocessed_images'

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # List all files in the input folder
        files = os.listdir(input_folder)



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

                    # Save the preprocessed image to the output folder
                    new_filename = f"{file[:-4]}_preprocessed.jpg"  # Remove the file extension from the original filename
                    output_path = os.path.join(output_folder, new_filename)
                    cv2.imwrite(output_path, normalized_image * 255)  # Multiply by 255 to convert back to [0, 255] range before saving
                    print(f"Preprocessed {file} -> {new_filename}")
                    
                except Exception as e:
                    print(f"Error processing image {file}: {e}")

# feature extraction

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
            if file.endswith('.jpg'):
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


# normalizing features

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
            if file.endswith('.npy'):  # Consider only numpy files, modify if needed
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


then i merged all codes in 1 to save time and storage

# merged
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    input_folder = 'images'
    output_folder = 'normalized_features'

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

            #     # Save the preprocessed image to the output folder
            #     new_filename = f"{file[:-4]}_preprocessed.jpg"  # Remove the file extension from the original filename
            #     output_path = os.path.join(output_folder, new_filename)
            #     cv2.imwrite(output_path, normalized_image * 255)  # Multiply by 255 to convert back to [0, 255] range before saving
            #     print(f"Preprocessed {file} -> {new_filename}")
                
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
            
        