import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

dataset_path = 'Q2/text_files'
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

translator = str.maketrans('', '', string.punctuation)

for file in os.listdir(dataset_path):
    if file.endswith('.txt'):
        file_path = os.path.join(dataset_path, file)

        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = file.read()

        lowercase = raw_data.lower()

        tokenized = nltk.word_tokenize(lowercase)

        filtered_tokens = [token for token in tokenized if token not in stop_words]

        removepanctuation=[word.translate(translator) for word in filtered_tokens]
        

        preprocessed_tokens = []
        for token in removepanctuation:
            # Stemming
            stemmed_token = stemmer.stem(token)
            # Lemmatization
            lemmatized_token = lemmatizer.lemmatize(token)
            if stemmed_token not in stop_words and lemmatized_token not in stop_words:
                preprocessed_tokens.append(stemmed_token)

        

        # Extract file name from file path
        file_name = os.path.basename(file_path)

        # Construct preprocessed file path using file name
        preprocessed_file_path = os.path.join('Q2/preprocessed_text_files', 'preprocessed_' + file_name)
        with open(preprocessed_file_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(preprocessed_tokens))
