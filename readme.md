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
            
        
---------------------------------------------------------------------------------------------------------------------------------------


for Q3 too i have first made different codes for part a and b then merged them 

# part a image retrieval

import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def load_normalized_features(normalized_features_dir):
    """Load normalized features for all images from the specified directory."""
    features = {}
    for feature_file in os.listdir(normalized_features_dir):
        if feature_file.endswith('.npy'):
            # Image ID is derived from the filename by removing '_normalized.npy'
            image_id = os.path.splitext(feature_file)[0].replace('_normalized', '')
            features_path = os.path.join(normalized_features_dir, feature_file)
            features[image_id] = np.load(features_path)
    return features

def reviews(reviews_dir):
    """Load reviews for all images from the specified directory."""
    reviews = {}
    for review_file in os.listdir(reviews_dir):
        if review_file.endswith('.txt'):
            # Extract image ID from the filename
            image_id = os.path.splitext(review_file)[0]
            review_path = os.path.join(reviews_dir, review_file)
            with open(review_path, 'r') as file:
                reviews[image_id] = file.read().strip()
    return reviews


def find_top_similar_images(image_id, features, top_k=3):
    """Find and return the top K similar images for the given image ID."""
    if image_id not in features:
        print(f"Features for image ID {image_id} are not found.")
        return []
    
    # Reshape the feature vector of the given image for cosine similarity computation
    given_image_features = features[image_id].reshape(1, -1)
    
    # Exclude the given image's features from the comparison set
    other_images = {id_: feat for id_, feat in features.items() if id_ != image_id}
    other_features = np.array(list(other_images.values()))
    other_ids = list(other_images.keys())
    
    # Compute the cosine similarity between the given image and all others
    similarity_scores = cosine_similarity(given_image_features, other_features)[0]
    
    # Identify the indices of the top K similarity scores
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    
    # Retrieve the top K similar image IDs and their similarity scores
    top_similar_images = [(other_ids[index], similarity_scores[index]) for index in top_indices]
    
    return top_similar_images

#Path to the directory containing the normalized feature files
normalized_features_dir = 'Q1/normalized_features'

#Load the normalized features from the specified directory
features = load_normalized_features(normalized_features_dir)

#The image ID for which you want to find similar images
image_id = input("enter image id: ")  # The '.jpg' is omitted because the feature files are named without the file extension

#Path to the directory containing the reviews
reviews_dir = 'Q2/text_files'

review_id = input("review: ")


#Retrieve the top 3 similar images and their similarity scores
top_similar_images = find_top_similar_images(image_id, features, top_k=3)
print("Top 3 similar images and their similarity scores:")
for img_id, score in top_similar_images:
    print(f"Image ID: {img_id}, Similarity Score: {score}")
    review_id = img_id.split('_')[0]  # Extracting the review ID from the image ID
    if review_id in reviews:
        print("Review:")
        print(reviews[review_id])
    else:
        print("Review not available.")


# part b text retrieval

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

preprocessed_dir = 'Q2/preprocessed_text_files' 

#Load preprocessed texts and their IDs
preprocessed_texts = []
ids = []
for filename in os.listdir(preprocessed_dir):
    if filename.startswith('preprocessed_'):
        with open(os.path.join(preprocessed_dir, filename), 'r', encoding='utf-8') as file:
            preprocessed_texts.append(file.read())
            ids.append(filename.replace('preprocessed_', '').replace('.txt', ''))
#initialize TF-IDF Vectorizer and compute TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
def find_similar_reviews(input_id, ids, tfidf_matrix, top_k=3):
    if input_id not in ids:
        print(f"Review ID {input_id} not found.")
        return []
    
    input_index = ids.index(input_id)
    input_vector = tfidf_matrix[input_index]

    # Compute cosine similarity scores between the input vector and all others
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # Get indices of the top_k similar reviews, excluding the input review itself
    top_indices = similarity_scores.argsort()[-top_k-1:-1][::-1]  # Exclude the last one (itself)

    # Map indices back to IDs
    similar_reviews = [(ids[i], similarity_scores[i]) for i in top_indices]
    
    return similar_reviews

input_id = input("Enter review ID: ")  
similar_reviews = find_similar_reviews(input_id, ids, tfidf_matrix, top_k=3)

print("Similar reviews:")
for review_id, score in similar_reviews:
    print(f"Review ID: {review_id}, Similarity Score: {score}")


results = {'image_similarity': top_similar_images, 'review_similarity': similar_reviews}
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved in 'results.pkl' file.")


# merged 

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#First Part: Image Similarity

def load_normalized_features(normalized_features_dir):
    features = {}
    for feature_file in os.listdir(normalized_features_dir):
        if feature_file.endswith('.npy'):
            image_id = os.path.splitext(feature_file)[0].replace('_normalized', '')
            features_path = os.path.join(normalized_features_dir, feature_file)
            features[image_id] = np.load(features_path)
    return features

def find_top_similar_images(image_id, features, top_k=3):
    if image_id not in features:
        print(f"Features for image ID {image_id} are not found.")
        return []
    
    given_image_features = features[image_id].reshape(1, -1)
    other_images = {id_: feat for id_, feat in features.items() if id_ != image_id}
    other_features = np.array(list(other_images.values()))
    other_ids = list(other_images.keys())
    
    similarity_scores = cosine_similarity(given_image_features, other_features)[0]
    top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    top_similar_images = [(other_ids[index], similarity_scores[index]) for index in top_indices]
    
    return top_similar_images

#Adjust these paths according to your directory structure
normalized_features_dir = 'Q1/normalized_features'
features = load_normalized_features(normalized_features_dir)
image_id = input("enter image id: ")
top_similar_images = find_top_similar_images(image_id, features)

#Second Part: Review Similarity

preprocessed_dir = 'Q2/preprocessed_text_files'
preprocessed_texts = []
ids = []
for filename in os.listdir(preprocessed_dir):
    if filename.startswith('preprocessed_'):
        with open(os.path.join(preprocessed_dir, filename), 'r', encoding='utf-8') as file:
            preprocessed_texts.append(file.read())
            ids.append(filename.replace('preprocessed_', '').replace('.txt', ''))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

def find_similar_reviews(input_id, ids, tfidf_matrix, top_k=3):
    if input_id not in ids:
        print(f"Review ID {input_id} not found.")
        return []
    
    input_index = ids.index(input_id)
    input_vector = tfidf_matrix[input_index]
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-top_k-1:-1][::-1]
    similar_reviews = [(ids[i], similarity_scores[i]) for i in top_indices]
    
    return similar_reviews

input_id = input("Enter review ID: ")
similar_reviews = find_similar_reviews(input_id, ids, tfidf_matrix)

#Saving Results

results = {
    'image_similarity': top_similar_images,
    'review_similarity': similar_reviews
}

print(results)
with open('Q3/results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved in 'results.pkl' file.")
