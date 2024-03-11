import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Function to load normalized image features
def load_normalized_image_features(features_directory):
    features = {}
    files=os.listdir(features_directory)
    for file in files:
        if file.endswith('_normalized.npy'):
            image_id = file.replace('_normalized.npy', '')
            feature_path = os.path.join(features_directory, file)
            features[image_id] = np.load(feature_path)
    return features

# Load your normalized image features (adjust the directory as needed)
features_directory = 'Q1\\normalized_features'
image_features = load_normalized_image_features(features_directory)




def retrieve_similar_images(input_image_id, image_features, top_n=3):
    if input_image_id not in image_features:
        print("Input image ID not found in the dataset.")
        return []
    
    input_features = image_features[input_image_id]
    input_features = input_features.reshape(1, -1)  # Reshape for compatibility with sklearn's function
    all_features_matrix = np.array(list(image_features.values()))
    similarity_scores = cosine_similarity(input_features, all_features_matrix)
    
    # Get sorted indices based on similarity scores, excluding the first one (highest score) as it's the input image
    sorted_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    
    # Retrieve the image IDs and their scores for the top matches
    similar_images = [(list(image_features.keys())[index], similarity_scores[index]) for index in sorted_indices]
    
    return similar_images
# Print out the available image IDs
# print(list(image_features.keys()))

# Example usage (ensure you replace 'your_input_image_id_here' with an actual image ID from your dataset)
input_image_id = input("enter image id: ")
similar_images = retrieve_similar_images(input_image_id, image_features, top_n=3)

# Display or save the results
print("Top 3 similar images:")
for image_id, score in similar_images:
    print(f"Image ID: {image_id}, Similarity Score: {score}")

# # Optionally, save results with pickle
# with open('image_retrieval_results.pkl', 'wb') as f:
#     pickle.dump(similar_images, f)
