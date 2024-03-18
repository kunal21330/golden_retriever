import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# First Part: Image Similarity

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

# Adjust these paths according to your directory structure
normalized_features_dir = 'Q1/normalized_features'
features = load_normalized_features(normalized_features_dir)
image_id = input("enter image id: ")
top_similar_images = find_top_similar_images(image_id, features)

# Second Part: Review Similarity

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

# Saving Results

results = {
    'image_similarity': top_similar_images,
    'review_similarity': similar_reviews
}

print(results)
with open('Q3/results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved in 'results.pkl' file.")
