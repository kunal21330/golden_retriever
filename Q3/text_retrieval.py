import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

preprocessed_dir = 'Q2/preprocessed_text_files' 

# Load preprocessed texts and their IDs
preprocessed_texts = []
ids = []
for filename in os.listdir(preprocessed_dir):
    if filename.startswith('preprocessed_'):
        with open(os.path.join(preprocessed_dir, filename), 'r', encoding='utf-8') as file:
            preprocessed_texts.append(file.read())
            ids.append(filename.replace('preprocessed_', '').replace('.txt', ''))
# Initialize TF-IDF Vectorizer and compute TF-IDF matrix
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
