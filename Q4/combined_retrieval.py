import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


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
image_id = input("enter image id: ") # example 11_0
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

input_id = input("Enter review ID: ") # example: 11
similar_reviews = find_similar_reviews(input_id, ids, tfidf_matrix)


def extract_base_review_id(image_id):
    # Extract the base review ID from the image ID
    return image_id.split('_')[0]

def combine_and_rank(top_similar_images, similar_reviews):
    combined_scores = {}
    
    # Initialize combined scores with review scores
    for review_id, review_score in similar_reviews:
        combined_scores[review_id] = {'image': [], 'review': review_score, 'average': 0}
    
    # Aggregate image similarity scores by base review ID
    for img_id, img_score in top_similar_images:
        base_review_id = extract_base_review_id(img_id)
        if base_review_id in combined_scores:
            combined_scores[base_review_id]['image'].append(img_score)
        else:
            combined_scores[base_review_id] = {'image': [img_score], 'review': 0, 'average': 0}
    
    # Calculate average image score for each base review ID, then compute the composite score
    for id_, scores in combined_scores.items():
        image_score_avg = np.mean(scores['image']) if scores['image'] else 0
        scores['average'] = np.mean([image_score_avg, scores['review']])
    
    # Rank based on the composite score
    ranked_results = sorted(combined_scores.items(), key=lambda x: x[1]['average'], reverse=True)
    return ranked_results

# Assuming you've executed the parts for finding top similar images and reviews
ranked_results = combine_and_rank(top_similar_images, similar_reviews)

# Displaying the combined and ranked results
print("Combined and Ranked Results:")
print(ranked_results)
