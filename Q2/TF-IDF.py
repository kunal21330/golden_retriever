import os
import math
import numpy as np
import pickle

# Directory containing all your preprocessed text files
preprocessed_dir = 'Q2/preprocessed_text_files'

# Step 1: Read documents and build a corpus
documents = []

files=os.listdir(preprocessed_dir)
for file in files:
    if file.endswith('.txt'):
        filepath = os.path.join(preprocessed_dir, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append(f.read())


# Step 2: Calculate TF (Term Frequency)
def compute_tf(text):
    tf_dict = {}
    words = text.split()
    word_count = len(words)
    for word in words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / word_count
    return tf_dict

tfs = [compute_tf(doc) for doc in documents]

#after this i got a list of dictionaries with each dictionary contains tf score of each word in a document

# Step 3: Calculate IDF (Inverse Document Frequency)
def compute_idf(docs):
    N = len(docs)
    idf_dict = {}
    for doc in docs:
        words=set(doc.split())
        for word in words:
            idf_dict[word] = idf_dict.get(word, 0) + 1
    for word in idf_dict:
        idf_dict[word] = math.log(N / float(idf_dict[word]))
    return idf_dict

idf = compute_idf(documents)

# Step 4: Calculate TF-IDF
def compute_tfidf(tf, idfs):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val * idfs.get(word, 0)
    return tfidf

tfidfs = [compute_tfidf(tf, idf) for tf in tfs]

# Assuming `tfidfs` contains the TF-IDF scores for your documents
tfidf_filename = 'Q2/tfidf_scores.pickle'

# Write TF-IDF scores to a pickle file
with open(tfidf_filename, 'wb') as f:
    pickle.dump(tfidfs, f)

print(f"TF-IDF scores saved to {tfidf_filename}")
