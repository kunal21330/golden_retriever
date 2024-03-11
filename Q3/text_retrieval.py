# import numpy as np
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load precomputed TF-IDF scores (assuming it's a matrix) and the TfidfVectorizer's vocabulary
# with open('tfidf_matrix.pkl', 'rb') as f:
#     tfidf_matrix = pickle.load(f)

# with open('tfidf_vocab.pkl', 'rb') as f:
#     tfidf_vocab = pickle.load(f)

# # Initialize a TfidfVectorizer with the loaded vocabulary
# vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab)

# # Function to process and vectorize the input review
# def vectorize_input_review(review, vectorizer):
#     return vectorizer.transform([review])

# # Example input review
# input_review = "Example text of the input review here."

# # Vectorize the input review
# input_tfidf_vector = vectorize_input_review(input_review, vectorizer)

# # Compute cosine similarity between the input review TF-IDF vector and the dataset TF-IDF matrix
# cosine_similarities = cosine_similarity(input_tfidf_vector, tfidf_matrix).flatten()

# # Find the indices of the top 3 most similar reviews
# top_indices = np.argsort(cosine_similarities)[-3:][::-1]

# # Print out the indices and their corresponding similarity scores
# print("Indices of Top 3 Similar Reviews:", top_indices)
# print("Similarity Scores:", cosine_similarities[top_indices])
