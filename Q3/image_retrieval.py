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

# Path to the directory containing the normalized feature files
normalized_features_dir = 'Q1/normalized_features'

# Load the normalized features from the specified directory
features = load_normalized_features(normalized_features_dir)

# The image ID for which you want to find similar images
image_id = input("enter image id: ")  # The '.jpg' is omitted because the feature files are named without the file extension

# Retrieve the top 3 similar images and their similarity scores
top_similar_images = find_top_similar_images(image_id, features, top_k=3)
print("Top 3 similar images and their similarity scores:")
for img_id, score in top_similar_images:
    print(f"Image ID: {img_id}, Similarity Score: {score}")
