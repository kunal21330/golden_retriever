import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

dataset_path = 'Q2/text_files'
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define a translator for removing punctuation
translator = str.maketrans('', '', string.punctuation)

files=os.listdir(dataset_path)
for file in files:
    if file.endswith('.txt'):
        file_path = os.path.join(dataset_path, file)

        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = file.read()

        # Convert text to lowercase
        lowercase = raw_data.lower()

        # Tokenize text
        tokenized = nltk.word_tokenize(lowercase)

        # Filter out tokens that are in the list of stop words
        filtered_tokens = [token for token in tokenized if token not in stop_words]

        # Remove punctuation from tokens
        no_punctuation_tokens = [word.translate(translator) for word in filtered_tokens]

        # Apply stemming to tokens
        stemmed_tokens = [stemmer.stem(token) for token in no_punctuation_tokens if token not in stop_words]

        # Extract file name from file path
        file_name = os.path.basename(file_path)

        # Construct preprocessed file path using file name
        preprocessed_file_path = os.path.join('Q2/preprocessed_text_files', 'preprocessed_' + file_name)

        # Write preprocessed tokens back to a new file
        with open(preprocessed_file_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(stemmed_tokens))

        print(f"Preprocessed file saved to {preprocessed_file_path}")
